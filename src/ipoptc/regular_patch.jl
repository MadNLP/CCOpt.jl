function get_eta(solver::MadNLPSolver)
    # TODO mu thresh to options?
    if solver.mu ≤ 5e-6
        return 0.1*solver.mu/(1+maximum(solver.y))
    else
        return 0
    end
end

function MadNLP.regular!(
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
            MadNLP.@info(solver.logger, "Updating Scholtes relaxation parameter. $(mu_new)")
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
