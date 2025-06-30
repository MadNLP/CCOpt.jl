include("project.jl")

function get_eta_heuristic(solver::MadNLPCSolver)
    if solver.ipm.mu ≤ solver.opts.mu_thresh
        return solver.opts.eta_factor*solver.ipm.mu/(
            1+max(
                maximum(MadNLP.slack(solver.ipm.zu)),
                maximum(MadNLP.slack(solver.ipm.zl)),
            )
        )
    else
        return 0.0
    end
end

function MadNLP.set_aug_diagonal!(
    kkt::MadNLP.AbstractKKTSystem{T},
    solver::MadNLPCSolver{T, VT},
    eta::T,
) where {T, VT}
    ipm = solver.ipm
    n = length(ipm.x_ur)
    ncc = ipm.nlp.mpcc.meta.ncc

    fill!(kkt.reg, zero(T))
    fill!(kkt.du_diag, zero(T))
    kkt.l_diag .= ipm.xl_r .- ipm.x_lr   # (Xˡ - X)
    kkt.u_diag .= ipm.x_ur .- ipm.xu_r   # (X - Xᵘ)
    copyto!(kkt.l_lower, ipm.zl_r)
    copyto!(kkt.u_lower, ipm.zu_r)

    # Regularize with 𝜂 using vicente-wright
    if solver.opts.kkt_regularization == :vicente_wright
        MadNLP.@debug(
            solver.logger,
            "Applying regularization to complementarity slacks eta = $(eta)"
        )
        kkt.u_diag[(n-ncc+1):n] .= @views min.(kkt.u_diag[(n-ncc+1):n], -eta)
        kkt.u_lower[(n-ncc+1):n] .= @views max.(kkt.u_lower[(n-ncc+1):n], eta)
    end

    MadNLP._set_aug_diagonal!(kkt)
    return
end

function MadNLP.set_aug_diagonal!(
    kkt::MadNLP.ScaledSparseKKTSystem{T},
    solver::MadNLPCSolver{T, VT},
    eta::T,
) where {T, VT}
    ipm = solver.ipm
    n = length(ipm.x_ur)
    ncc = ipm.nlp.mpcc.meta.ncc

    fill!(kkt.reg, zero(T))
    fill!(kkt.du_diag, zero(T))
    # Ensure l_diag and u_diag have only non negative entries
    kkt.l_diag .= ipm.x_lr .- ipm.xl_r   # (X - Xˡ)
    kkt.u_diag .= ipm.xu_r .- ipm.x_ur   # (Xᵘ - X)
    copyto!(kkt.l_lower, ipm.zl_r)
    copyto!(kkt.u_lower, ipm.zu_r)

    if solver.opts.kkt_regularization == :vicente_wright
        MadNLP.@debug(
            solver.logger,
            "Applying regularization to complementarity slacks eta = $(eta)"
        )
        kkt.u_diag[(n-ncc+1):n] .= @views max.(kkt.u_diag[(n-ncc+1):n], eta)
        kkt.u_lower[(n-ncc+1):n] .= @views max.(kkt.u_lower[(n-ncc+1):n], eta)
    end

    return MadNLP._set_aug_diagonal!(kkt)
end

function solve_homotopy!(nlp::MadMPEC.ScholtesRelaxation, solver::MadNLPCSolver; kwargs...)
    return solve_homotopy!(nlp, solver, MadNLP.MadNLPExecutionStats(solver.ipm); kwargs...)
end

function solve_homotopy!(solver::MadNLPCSolver; kwargs...)
    return solve_homotopy!(solver.scholtes, solver; kwargs...)
end

# TODO(@anton) Why do we pass things this way???
function solve_homotopy!(
    nlp::MadMPEC.ScholtesRelaxation,
    solver::MadMPEC.MadNLPCSolver,
    stats::MadNLP.MadNLPExecutionStats;
    x=nothing,
    y=nothing,
    zl=nothing,
    zu=nothing,
    kwargs...,
)
    ipm = solver.ipm
    if x != nothing
        MadNLP.full(ipm.x)[1:get_nvar(nlp)] .= x
    end
    if y != nothing
        ipm.y[1:get_ncon(nlp)] .= y
    end
    if zl != nothing
        MadNLP.full(ipm.zl)[1:get_nvar(nlp)] .= zl
    end
    if zu != nothing
        MadNLP.full(ipm.zu)[1:get_nvar(nlp)] .= zu
    end

    if !isempty(kwargs)
        MadNLP.@warn(ipm.logger, "The options set during resolve may not have an effect")
        set_options!(ipm.opt, kwargs)
    end

    try
        if ipm.status == MadNLP.INITIAL
            MadNLP.@notice(
                solver.logger,
                "This is $(MadNLP.introduce()), using MadMPEC extension, running with $(MadNLP.introduce(ipm.kkt.linear_solver))\n"
            )
            MadNLP.print_init(ipm)
            ipm.status = MadNLP.initialize!(ipm)
            # Also reset sigma
            ipm.nlp.𝜎[] = ipm.mu
        else # resolving the problem
            ipm.status = MadNLP.reinitialize!(ipm)
            # Also reset sigma
            ipm.nlp.𝜎[] = ipm.mu
        end
        # possibly fix complementarity variable upper bounds:
        if solver.opts.respect_comp_bounds
            MadNLP.variable(ipm.xl)[solver.mpcc.meta.ind_cc1] .=
                solver.mpcc.meta.lvar[solver.mpcc.meta.ind_cc1]
            MadNLP.variable(ipm.xl)[solver.mpcc.meta.ind_cc2] .=
                solver.mpcc.meta.lvar[solver.mpcc.meta.ind_cc2]
        end

        while ipm.status >= MadNLP.REGULAR
            ipm.status == MadNLP.REGULAR && (ipm.status = MadMPEC.homotopy!(solver))
            ipm.status == MadNLP.RESTORE && (ipm.status = MadNLP.restore!(ipm))
            ipm.status == MadNLP.ROBUST && (ipm.status = MadNLP.robust!(ipm))
        end
    catch e
        if e isa MadNLP.InvalidNumberException
            if e.callback == :obj
                ipm.status=MadNLP.INVALID_NUMBER_OBJECTIVE
            elseif e.callback == :grad
                ipm.status=MadNLP.INVALID_NUMBER_GRADIENT
            elseif e.callback == :cons
                ipm.status=MadNLP.INVALID_NUMBER_CONSTRAINTS
            elseif e.callback == :jac
                ipm.status=MadNLP.INVALID_NUMBER_JACOBIAN
            elseif e.callback == :hess
                ipm.status=MadNLP.INVALID_NUMBER_HESSIAN_LAGRANGIAN
            else
                ipm.status=MadNLP.INVALID_NUMBER_DETECTED
            end
        elseif e isa MadNLP.NotEnoughDegreesOfFreedomException
            ipm.status=MadNLP.NOT_ENOUGH_DEGREES_OF_FREEDOM
        elseif e isa MadNLP.LinearSolverException
            ipm.status=MadNLP.ERROR_IN_STEP_COMPUTATION;
            ipm.opt.rethrow_error && rethrow(e)
        elseif e isa MadNLP.InterruptException
            ipm.status=MadNLP.USER_REQUESTED_STOP
            ipm.opt.rethrow_error && rethrow(e)
        elseif e isa AmplException
            ipm.status=MadNLP.INVALID_NUMBER_DETECTED
        else
            ipm.status=MadNLP.INTERNAL_ERROR
            ipm.opt.rethrow_error && rethrow(e)
        end
    finally
        ipm.cnt.total_time = time() - ipm.cnt.start_time
        if !(ipm.status < MadNLP.SOLVE_SUCCEEDED)
            MadNLP.print_summary(ipm)
        end

        MadNLP.@notice(
            solver.logger,
            "EXIT: $(MadNLP.get_status_output(ipm.status, ipm.opt))"
        )
        ipm.opt.disable_garbage_collector && (
            GC.enable(true);
            MadNLP.@warn(ipm.logger, "Julia garbage collector is turned back on")
        )
        MadNLP.finalize(ipm.logger)
        finalize(solver.iterate_logger)

        MadNLP.update!(stats, ipm)
    end

    return stats
end

function homotopy!(solver::MadNLPCSolver{T, VT}) where {T, VT}
    ipm = solver.ipm
    mpcc = solver.mpcc
    c_mpcc = VT(undef, length(ipm.c))
    log_iter(solver.iterate_logger, solver) # Log initial state
    while true
        # Set 𝜎 to zero for constraint infeasibility calculations
        if (ipm.cnt.k!=0 && !ipm.opt.jacobian_constant)
            MadNLP.eval_jac_wrapper!(ipm, ipm.kkt, ipm.x)
        end

        𝜎 = ipm.nlp.𝜎[]
        ipm.nlp.𝜎[] = 0
        MadNLP.eval_cons_wrapper!(ipm, c_mpcc, ipm.x)
        ipm.nlp.𝜎[] = 𝜎
        MadNLP.jtprod!(ipm.jacl, ipm.kkt, ipm.y)
        sd = MadNLP.get_sd(ipm.y, ipm.zl_r, ipm.zu_r, T(ipm.opt.s_max))
        sc = MadNLP.get_sc(ipm.zl_r, ipm.zu_r, T(ipm.opt.s_max))
        ipm.inf_pr = MadNLP.get_inf_pr(c_mpcc)
        ipm.inf_du = MadNLP.get_inf_du(
            MadNLP.full(ipm.f),
            MadNLP.full(ipm.zl),
            MadNLP.full(ipm.zu),
            ipm.jacl,
            sd,
        )
        ipm.inf_compl = MadNLP.get_inf_compl(
            ipm.x_lr,
            ipm.xl_r,
            ipm.zl_r,
            ipm.xu_r,
            ipm.x_ur,
            ipm.zu_r,
            zero(T),
            sc,
        )

        MadNLP.print_iter(ipm)
        log_iter(solver.iterate_logger, solver)
        # evaluate termination criteria
        MadNLP.@trace(ipm.logger, "Evaluating termination criteria.")
        max(ipm.inf_pr, ipm.inf_du, ipm.inf_compl) <= ipm.opt.tol &&
            return MadNLP.SOLVE_SUCCEEDED
        max(ipm.inf_pr, ipm.inf_du, ipm.inf_compl) <= ipm.opt.acceptable_tol ?
        (
            ipm.cnt.acceptable_cnt < ipm.opt.acceptable_iter ? ipm.cnt.acceptable_cnt+=1 :
            return MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL
        ) : (ipm.cnt.acceptable_cnt = 0)
        max(ipm.inf_pr, ipm.inf_du, ipm.inf_compl) >= ipm.opt.diverging_iterates_tol &&
            return MadNLP.DIVERGING_ITERATES
        ipm.cnt.k>=ipm.opt.max_iter && return MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
        time()-ipm.cnt.start_time>=ipm.opt.max_wall_time &&
            return MadNLP.MAXIMUM_WALLTIME_EXCEEDED

        # Now go back to using relaxed inf_pr
        ipm.inf_pr = MadNLP.get_inf_pr(ipm.c)
        # update the barrier parameter
        MadNLP.@trace(ipm.logger, "Updating the barrier parameter.")
        mu_old = ipm.mu
        MadNLP.update_barrier!(ipm.opt.barrier, solver, sc)
        mu_updated = ipm.mu != mu_old
        MadNLP.@debug(
            solver.logger,
            "Updated the barrier parameter from mu=$(mu_old) to mu=$(ipm.mu)"
        )
        MadNLP.@trace(solver.logger, "Updating the relaxation parameter.")
        update_sigma!(solver.opts.relaxation_update, solver)

        if mu_updated && solver.opts.use_magic_step
            ncc = mpcc.meta.ncc
            𝜅 = solver.opts.magic_step_kappa
            try
                @views project_scholtes_explicit!(
                    MadNLP.variable(ipm.x)[mpcc.meta.ind_cc1],
                    MadNLP.variable(ipm.x)[mpcc.meta.ind_cc2],
                    MadNLP.variable(ipm.x)[mpcc.meta.ind_cc1],
                    MadNLP.variable(ipm.x)[mpcc.meta.ind_cc2],
                    MadNLP.variable(ipm.xl)[mpcc.meta.ind_cc1],
                    MadNLP.variable(ipm.xl)[mpcc.meta.ind_cc2],
                    𝜅,
                    ipm.nlp.𝜎[],
                )
                # also update multipliers by z1 = 𝜇/x1 and z2 = 𝜇/x2
                # TODO(@anton) throwing away the multiplier information is probably incorrect
                #              but doing it correctly seems nontrivial
                if solver.opts.magic_step_duals
                    MadNLP.variable(ipm.zl)[mpcc.meta.ind_cc1] = @views ipm.mu ./ (
                        MadNLP.variable(ipm.x)[mpcc.meta.ind_cc1] .-
                        MadNLP.variable(ipm.xl)[mpcc.meta.ind_cc1]
                    )
                    MadNLP.variable(ipm.zl)[mpcc.meta.ind_cc2] = @views ipm.mu ./ (
                        MadNLP.variable(ipm.x)[mpcc.meta.ind_cc2] .-
                        MadNLP.variable(ipm.xl)[mpcc.meta.ind_cc2]
                    )
                end
                if solver.opts.magic_step_slack
                    MadNLP.slack(ipm.x)[(end-ncc+1):end] .= -(1-𝜅)*ipm.mu
                end
                if solver.opts.magic_step_slack_dual
                    MadNLP.slack(ipm.zu)[(end-ncc+1):end] .= ipm.mu/((1-𝜅)*ipm.mu)
                end
            catch e
                println(MadNLP.variable(ipm.xl)[mpcc.meta.ind_cc1])
                println(MadNLP.variable(ipm.xu)[mpcc.meta.ind_cc1])
                println(MadNLP.variable(ipm.xl)[mpcc.meta.ind_cc2])
                println(MadNLP.variable(ipm.xu)[mpcc.meta.ind_cc2])
                throw(e)
            end
            log_iter(solver.iterate_logger, solver; magic=true)
        end

        MadNLP.@trace(solver.logger, "Get eta.")
        eta_k = get_eta_heuristic(solver)

        # compute the newton step
        MadNLP.@trace(ipm.logger, "Computing the newton step.")
        if (ipm.cnt.k!=0)
            MadNLP.eval_lag_hess_wrapper!(ipm, ipm.kkt, ipm.x, ipm.y)
        end

        # TODO(@anton) update ipm.x ipm.zl, ipm.zu
        MadNLP.set_aug_diagonal!(ipm.kkt, solver, eta_k)
        MadNLP.set_aug_rhs!(ipm, ipm.kkt, ipm.c, ipm.mu)
        MadNLP.dual_inf_perturbation!(
            MadNLP.primal(ipm.p),
            ipm.ind_llb,
            ipm.ind_uub,
            ipm.mu,
            ipm.opt.kappa_d,
        )

        MadNLP.inertia_correction!(ipm.inertia_corrector, ipm) || return MadNLP.ROBUST

        MadNLP.@trace(ipm.logger, "Backtracking line search initiated.")
        status = MadNLP.filter_line_search!(ipm)
        if status != MadNLP.LINESEARCH_SUCCEEDED
            return status
        end

        MadNLP.@trace(ipm.logger, "Updating primal-dual variables.")
        copyto!(MadNLP.full(ipm.x), MadNLP.full(ipm.x_trial))
        copyto!(ipm.c, ipm.c_trial)
        ipm.obj_val = ipm.obj_val_trial
        MadNLP.adjust_boundary!(ipm.x_lr, ipm.xl_r, ipm.x_ur, ipm.xu_r, ipm.mu)

        MadNLP.axpy!(ipm.alpha, MadNLP.dual(ipm.d), ipm.y)

        ipm.zl_r .+= ipm.alpha_z .* MadNLP.dual_lb(ipm.d)
        ipm.zu_r .+= ipm.alpha_z .* MadNLP.dual_ub(ipm.d)
        MadNLP.reset_bound_dual!(
            MadNLP.primal(ipm.zl),
            MadNLP.primal(ipm.x),
            MadNLP.primal(ipm.xl),
            ipm.mu,
            ipm.opt.kappa_sigma,
        )
        MadNLP.reset_bound_dual!(
            MadNLP.primal(ipm.zu),
            MadNLP.primal(ipm.xu),
            MadNLP.primal(ipm.x),
            ipm.mu,
            ipm.opt.kappa_sigma,
        )

        MadNLP.eval_grad_f_wrapper!(ipm, ipm.f, ipm.x)

        ipm.cnt.k+=1
        MadNLP.@trace(ipm.logger, "Proceeding to the next interior point iteration.")
    end
end
