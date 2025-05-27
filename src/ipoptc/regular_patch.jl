function get_eta(solver::MadNLPSolver)
    # TODO mu thresh to options?
    if solver.mu ≤ 5e-6
        return 0.1*solver.mu/(1+maximum(solver.y))
    else
        return 0
    end
end

function madnlp_homotopy(model::MadMPEC.ScholtesRelaxation; kwargs...)
    solver = MadNLP.MadNLPSolver(model; kwargs...)
    return solve_homotopy!(solver)
end

function solve_homotopy!(
    nlp::MadMPEC.ScholtesRelaxation,
    solver::MadNLP.MadNLPSolver{
        T,
        VT,
        VI,
        KKTSystem,
        MadMPEC.ScholtesRelaxation{T, VT},
        CB,
        Iterator,
        IC,
        KKTVec,
    };
    kwargs...,
) where {T, VT, VI, KKTSystem, CB, Iterator, IC, KKTVec}
    return solve_homotopy!(nlp, solver, MadNLP.MadNLPExecutionStats(solver); kwargs...)
end

function solve_homotopy!(
    solver::MadNLP.MadNLPSolver{
        T,
        VT,
        VI,
        KKTSystem,
        MadMPEC.ScholtesRelaxation{T, VT},
        CB,
        Iterator,
        IC,
        KKTVec,
    };
    kwargs...,
) where {T, VT, VI, KKTSystem, CB, Iterator, IC, KKTVec}
    return solve_homotopy!(solver.nlp, solver; kwargs...)
end

function solve_homotopy!(
    nlp::MadMPEC.ScholtesRelaxation,
    solver::MadNLP.MadNLPSolver{
        T,
        VT,
        VI,
        KKTSystem,
        MadMPEC.ScholtesRelaxation{T, VT},
        CB,
        Iterator,
        IC,
        KKTVec,
    },
    stats::MadNLP.MadNLPExecutionStats;
    x=nothing,
    y=nothing,
    zl=nothing,
    zu=nothing,
    kwargs...,
) where {T, VT, VI, KKTSystem, CB, Iterator, IC, KKTVec}
    if x != nothing
        MadNLP.full(solver.x)[1:get_nvar(nlp)] .= x
    end
    if y != nothing
        solver.y[1:get_ncon(nlp)] .= y
    end
    if zl != nothing
        MadNLP.full(solver.zl)[1:get_nvar(nlp)] .= zl
    end
    if zu != nothing
        MadNLP.full(solver.zu)[1:get_nvar(nlp)] .= zu
    end

    if !isempty(kwargs)
        @warn(solver.logger, "The options set during resolve may not have an effect")
        set_options!(solver.opt, kwargs)
    end

    try
        if solver.status == MadNLP.INITIAL
            MadNLP.@notice(
                solver.logger,
                "This is $(MadNLP.introduce()), using MadMPEC extension, running with $(MadNLP.introduce(solver.kkt.linear_solver))\n"
            )
            MadNLP.print_init(solver)
            solver.status = MadNLP.initialize!(solver)
        else # resolving the problem
            solver.status = MadNLP.reinitialize!(solver)
        end

        while solver.status >= MadNLP.REGULAR
            solver.status == MadNLP.REGULAR && (solver.status = MadMPEC.homotopy!(solver))
            solver.status == MadNLP.RESTORE && (solver.status = MadNLP.restore!(solver))
            solver.status == MadNLP.ROBUST && (solver.status = MadNLP.robust!(solver))
        end
    catch e
        if e isa MadNLP.InvalidNumberException
            if e.callback == :obj
                solver.status=MadNLP.INVALID_NUMBER_OBJECTIVE
            elseif e.callback == :grad
                solver.status=MadNLP.INVALID_NUMBER_GRADIENT
            elseif e.callback == :cons
                solver.status=MadNLP.INVALID_NUMBER_CONSTRAINTS
            elseif e.callback == :jac
                solver.status=MadNLP.INVALID_NUMBER_JACOBIAN
            elseif e.callback == :hess
                solver.status=MadNLP.INVALID_NUMBER_HESSIAN_LAGRANGIAN
            else
                solver.status=MadNLP.INVALID_NUMBER_DETECTED
            end
        elseif e isa MadNLP.NotEnoughDegreesOfFreedomException
            solver.status=MadNLP.NOT_ENOUGH_DEGREES_OF_FREEDOM
        elseif e isa MadNLP.LinearSolverException
            solver.status=MadNLP.ERROR_IN_STEP_COMPUTATION;
            solver.opt.rethrow_error && rethrow(e)
        elseif e isa InterruptException
            solver.status=MadNLP.USER_REQUESTED_STOP
            solver.opt.rethrow_error && rethrow(e)
        else
            solver.status=INTERNAL_ERROR
            solver.opt.rethrow_error && rethrow(e)
        end
    finally
        solver.cnt.total_time = time() - solver.cnt.start_time
        if !(solver.status < MadNLP.SOLVE_SUCCEEDED)
            MadNLP.print_summary(solver)
        end
        MadNLP.@notice(
            solver.logger,
            "EXIT: $(MadNLP.get_status_output(solver.status, solver.opt))"
        )
        solver.opt.disable_garbage_collector && (
            GC.enable(true);
            @warn(solver.logger, "Julia garbage collector is turned back on")
        )
        MadNLP.finalize(solver.logger)

        MadNLP.update!(stats, solver)
    end

    return stats
end

function homotopy!(
    solver::MadNLP.MadNLPSolver{
        T,
        VT,
        VI,
        KKTSystem,
        MadMPEC.ScholtesRelaxation{T, VT},
        CB,
        Iterator,
        IC,
        KKTVec,
    },
) where {T, VT, VI, KKTSystem, CB, Iterator, IC, KKTVec}
    while true
        if (solver.cnt.k!=0 && !solver.opt.jacobian_constant)
            MadNLP.eval_jac_wrapper!(solver, solver.kkt, solver.x)
        end
        MadNLP.jtprod!(solver.jacl, solver.kkt, solver.y)
        sd = MadNLP.get_sd(solver.y, solver.zl_r, solver.zu_r, T(solver.opt.s_max))
        sc = MadNLP.get_sc(solver.zl_r, solver.zu_r, T(solver.opt.s_max))
        solver.inf_pr = MadNLP.get_inf_pr(solver.c)
        solver.inf_du = MadNLP.get_inf_du(
            MadNLP.full(solver.f),
            MadNLP.full(solver.zl),
            MadNLP.full(solver.zu),
            solver.jacl,
            sd,
        )
        solver.inf_compl = MadNLP.get_inf_compl(
            solver.x_lr,
            solver.xl_r,
            solver.zl_r,
            solver.xu_r,
            solver.x_ur,
            solver.zu_r,
            zero(T),
            sc,
        )
        inf_compl_mu = MadNLP.get_inf_compl(
            solver.x_lr,
            solver.xl_r,
            solver.zl_r,
            solver.xu_r,
            solver.x_ur,
            solver.zu_r,
            solver.mu,
            sc,
        )

        MadNLP.print_iter(solver)

        # evaluate termination criteria
        MadNLP.@trace(solver.logger, "Evaluating termination criteria.")
        max(solver.inf_pr, solver.inf_du, solver.inf_compl) <= solver.opt.tol &&
            return MadNLP.SOLVE_SUCCEEDED
        max(solver.inf_pr, solver.inf_du, solver.inf_compl) <= solver.opt.acceptable_tol ?
        (
            solver.cnt.acceptable_cnt < solver.opt.acceptable_iter ?
            solver.cnt.acceptable_cnt+=1 : return MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL
        ) : (solver.cnt.acceptable_cnt = 0)
        max(solver.inf_pr, solver.inf_du, solver.inf_compl) >=
        solver.opt.diverging_iterates_tol && return MadNLP.DIVERGING_ITERATES
        solver.cnt.k>=solver.opt.max_iter && return MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
        time()-solver.cnt.start_time>=solver.opt.max_wall_time &&
            return MadNLP.MAXIMUM_WALLTIME_EXCEEDED

        # update the barrier parameter
        MadNLP.@trace(solver.logger, "Updating the barrier parameter.")
        while solver.mu != max(solver.opt.mu_min, solver.opt.tol/10) &&
            max(solver.inf_pr, solver.inf_du, inf_compl_mu) <=
            solver.opt.barrier_tol_factor*solver.mu
            mu_new = MadNLP.get_mu(
                solver.mu,
                solver.opt.mu_min,
                solver.opt.mu_linear_decrease_factor,
                solver.opt.mu_superlinear_decrease_power,
                solver.opt.tol,
            )
            solver.inf_pr = MadNLP.get_inf_pr(solver.c)
            solver.inf_du = MadNLP.get_inf_du(
                MadNLP.full(solver.f),
                MadNLP.full(solver.zl),
                MadNLP.full(solver.zu),
                solver.jacl,
                sd,
            )
            inf_compl_mu = MadNLP.get_inf_compl(
                solver.x_lr,
                solver.xl_r,
                solver.zl_r,
                solver.xu_r,
                solver.x_ur,
                solver.zu_r,
                solver.mu,
                sc,
            )
            solver.tau = MadNLP.get_tau(solver.mu, solver.opt.tau_min)
            solver.mu = mu_new
            solver.nlp.𝜎[] = mu_new
            MadNLP.@info(solver.logger, "Updating Scholtes relaxation parameter: $(mu_new)")
            empty!(solver.filter)
            push!(solver.filter, (solver.theta_max, -Inf))
        end

        # TODO(@anton) update the scholtes relaxation parameter here!
        MadNLP.@trace(solver.logger, "Get eta.")
        eta_k = get_eta(solver)

        # compute the newton step
        MadNLP.@trace(solver.logger, "Computing the newton step.")
        if (solver.cnt.k!=0 && !solver.opt.hessian_constant)
            MadNLP.eval_lag_hess_wrapper!(solver, solver.kkt, solver.x, solver.y)
        end

        # TODO(@anton) update solver.x solver.zl, solver.zu
        MadNLP.set_aug_diagonal!(solver.kkt, solver)
        MadNLP.set_aug_rhs!(solver, solver.kkt, solver.c)
        MadNLP.dual_inf_perturbation!(
            MadNLP.primal(solver.p),
            solver.ind_llb,
            solver.ind_uub,
            solver.mu,
            solver.opt.kappa_d,
        )

        MadNLP.inertia_correction!(solver.inertia_corrector, solver) || return MadNLP.ROBUST

        MadNLP.@trace(solver.logger, "Backtracking line search initiated.")
        status = MadNLP.filter_line_search!(solver)
        if status != MadNLP.LINESEARCH_SUCCEEDED
            return status
        end

        MadNLP.@trace(solver.logger, "Updating primal-dual variables.")
        copyto!(MadNLP.full(solver.x), MadNLP.full(solver.x_trial))
        copyto!(solver.c, solver.c_trial)
        solver.obj_val = solver.obj_val_trial
        MadNLP.adjust_boundary!(
            solver.x_lr,
            solver.xl_r,
            solver.x_ur,
            solver.xu_r,
            solver.mu,
        )

        MadNLP.axpy!(solver.alpha, MadNLP.dual(solver.d), solver.y)

        solver.zl_r .+= solver.alpha_z .* MadNLP.dual_lb(solver.d)
        solver.zu_r .+= solver.alpha_z .* MadNLP.dual_ub(solver.d)
        MadNLP.reset_bound_dual!(
            MadNLP.primal(solver.zl),
            MadNLP.primal(solver.x),
            MadNLP.primal(solver.xl),
            solver.mu,
            solver.opt.kappa_sigma,
        )
        MadNLP.reset_bound_dual!(
            MadNLP.primal(solver.zu),
            MadNLP.primal(solver.xu),
            MadNLP.primal(solver.x),
            solver.mu,
            solver.opt.kappa_sigma,
        )

        MadNLP.eval_grad_f_wrapper!(solver, solver.f, solver.x)

        solver.cnt.k+=1
        MadNLP.@trace(solver.logger, "Proceeding to the next interior point iteration.")
    end
end
