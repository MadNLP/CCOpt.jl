function solve_homotopy!(nlp::MadMPEC.Ell1Relaxation, solver::ExactPenaltySolver; kwargs...)
    return solve_homotopy!(nlp, solver, MadNLP.MadNLPExecutionStats(solver.ipm); kwargs...)
end

function solve_homotopy!(solver::ExactPenaltySolver; kwargs...)
    return solve_homotopy!(solver.ell1, solver; kwargs...)
end

function solve_homotopy!(
    nlp::MadMPEC.Ell1Relaxation,
    solver::MadMPEC.ExactPenaltySolver,
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
                "This is $(MadNLP.introduce()), using MadMPEC Ell1 extension, running with $(MadNLP.introduce(ipm.kkt.linear_solver))\n"
            )
            MadNLP.print_init(ipm)
            # Also reset tau
            ipm.nlp.tau[] = solver.opts.tau_0
            ipm.status = MadNLP.initialize!(ipm)
        else # resolving the problem
            # Also reset tau
            ipm.nlp.tau[] = solver.opts.tau_0
            ipm.status = MadNLP.reinitialize!(ipm)
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

        update!(stats, solver)
    end

    return stats
end

function homotopy!(solver::ExactPenaltySolver{T, VT}) where {T, VT}
    ipm = solver.ipm
    nlp = solver.ell1
    mpcc = solver.mpcc
    while true
        # Set sigma to zero for constraint infeasibility calculations
        if (ipm.cnt.k!=0 && !ipm.opt.jacobian_constant)
            MadNLP.eval_jac_wrapper!(ipm, ipm.kkt, ipm.x)
        end
        MadNLP.jtprod!(ipm.jacl, ipm.kkt, ipm.y)
        sd = MadNLP.get_sd(ipm.y, ipm.zl_r, ipm.zu_r, T(ipm.opt.s_max))
        sc = MadNLP.get_sc(ipm.zl_r, ipm.zu_r, T(ipm.opt.s_max))
        ipm.inf_pr = MadNLP.get_inf_pr(ipm.c)
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
        inf_compl_mu = MadNLP.get_inf_compl(
            ipm.x_lr,
            ipm.xl_r,
            ipm.zl_r,
            ipm.xu_r,
            ipm.x_ur,
            ipm.zu_r,
            ipm.mu,
            sc,
        )
        inf_pr_comp = MadMPEC.comp_residual(mpcc, MadNLP.variable(ipm.x)) # Primal complementarity residual
        inf_pr_comp_prod = MadMPEC.comp_residual_product(mpcc, MadNLP.variable(ipm.x)) # Primal complementarity residual
        inf_pr_comp_sum = MadMPEC.comp_residual_sum(mpcc, MadNLP.variable(ipm.x)) # Primal complementarity residual
        push!(solver.pr_comp_hist, inf_pr_comp_sum)

        MadNLP.print_iter(ipm)

        # evaluate termination criteria
        MadNLP.@trace(ipm.logger, "Evaluating etrmination criteria.")
        max(ipm.inf_pr, ipm.inf_du, ipm.inf_compl, inf_pr_comp) <= ipm.opt.tol &&
            return MadNLP.SOLVE_SUCCEEDED
        max(ipm.inf_pr, ipm.inf_du, ipm.inf_compl, inf_pr_comp_prod) <=
        ipm.opt.acceptable_tol ?
        (
            ipm.cnt.acceptable_cnt < ipm.opt.acceptable_iter ? ipm.cnt.acceptable_cnt+=1 :
            return MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL
        ) : (ipm.cnt.acceptable_cnt = 0)
        max(ipm.inf_pr, ipm.inf_du, ipm.inf_compl, inf_pr_comp) >=
        ipm.opt.diverging_iterates_tol && return MadNLP.DIVERGING_ITERATES
        ipm.cnt.k>=ipm.opt.max_iter && return MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
        time()-ipm.cnt.start_time>=ipm.opt.max_wall_time &&
            return MadNLP.MAXIMUM_WALLTIME_EXCEEDED

        # Do dynamic penalty update:
        # First calculate primal comp epsilon
        eps_pr_comp = ipm.mu^solver.opts.gamma
        if solver.opts.dynamic_tau_update &&
           inf_pr_comp > eps_pr_comp &&
           inf_pr_comp_sum > solver.opts.eta_dynamic_update*maximum(solver.pr_comp_hist)
            nlp.tau[] = solver.opts.tau_growth_rate*nlp.tau[]
            MadNLP.@trace(
                solver.logger,
                "Updating the penalty parameter dynamically to $(nlp.tau[])."
            )
            ipm.obj_val = MadNLP.eval_f_wrapper(ipm, ipm.x)
            # Also clear the filter
            empty!(ipm.filter)
            push!(ipm.filter, (ipm.theta_max, -Inf))
        end
        # update the barrier parameter
        MadNLP.@trace(ipm.logger, "Updating the barrier parameter.")
        mu_updated = false
        mu_old = ipm.mu
        MadNLP.update_barrier!(ipm.opt.barrier, solver.ipm, sc)
        mu_updated = ipm.mu != mu_old
        # Standard check
        if mu_updated
            # check for complementarity convergence when we decrease 𝜇
            # or if we already are at smallest mu increase penalty if we are not satisfying eps_pr_comp
            if inf_pr_comp > eps_pr_comp
                nlp.tau[] = solver.opts.tau_growth_rate*nlp.tau[]
                MadNLP.@trace(
                    solver.logger,
                    "Updating the penalty parameter to $(nlp.tau[])."
                )
                ipm.obj_val = MadNLP.eval_f_wrapper(ipm, ipm.x)
            end
        elseif ipm.mu ≤ max(ipm.opt.barrier.mu_min, ipm.opt.tol/10) &&
               max(ipm.inf_pr, ipm.inf_du, inf_compl_mu) <=
               ipm.opt.barrier_tol_factor*ipm.mu
            if inf_pr_comp > ipm.opt.tol
                nlp.tau[] = solver.opts.tau_growth_rate*nlp.tau[]
                MadNLP.@trace(
                    solver.logger,
                    "Updating the penalty parameter to $(nlp.tau[])."
                )
                ipm.obj_val = MadNLP.eval_f_wrapper(ipm, ipm.x)
                empty!(ipm.filter)
                push!(ipm.filter, (ipm.theta_max, -Inf))
            end
        end

        # compute the newton step
        MadNLP.@trace(ipm.logger, "Computing the newton step.")
        if (ipm.cnt.k!=0)
            MadNLP.eval_lag_hess_wrapper!(ipm, ipm.kkt, ipm.x, ipm.y)
        end
        MadNLP.set_aug_diagonal!(ipm.kkt, ipm)
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

# evaluate mpcc objective instead of ell1 objective (though they should be the same)
function update!(stats::MadNLP.MadNLPExecutionStats, solver::ExactPenaltySolver)
    MadNLP.update!(stats, solver.ipm)
    stats.objective = MadMPEC.obj(solver.mpcc, stats.solution)
    return stats
end
