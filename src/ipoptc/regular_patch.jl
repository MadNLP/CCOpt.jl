function get_eta(solver::MadNLPSolver)
    # TODO mu thresh to options?
    if solver.mu ≤ 5e-6
        return 0.1*solver.mu/(1+maximum(solver.y))
    else
        return 0
    end
end
"""
MadNLPSolver{Float64,
             Vector{Float64},
             Vector{Int64},
             MadNLP.SparseKKTSystem{Float64, Vector{Float64}, SparseArrays.SparseMatrixCSC{Float64, Int32}, MadNLP.ExactHessian{Float64, Vector{Float64}}, UmfpackSolver{Float64}, Vector{Int64}, Vector{Int32}},
             MadMPEC.ScholtesRelaxation{Float64, Vector{Float64}},
             MadNLP.SparseCallback{Float64, Vector{Float64}, Vector{Int64}, MadMPEC.ScholtesRelaxation{Float64, Vector{Float64}}, MadNLP.MakeParameter{Vector{Float64}, Vector{Int64}}, MadNLP.EnforceEquality},
             MadNLP.RichardsonIterator{Float64, MadNLP.SparseKKTSystem{Float64, Vector{Float64}, SparseArrays.SparseMatrixCSC{Float64, Int32}, MadNLP.ExactHessian{Float64, Vector{Float64}}, UmfpackSolver{Float64}, Vector{Int64}, Vector{Int32}}},
             MadNLP.InertiaFree{Float64, Vector{Float64}, MadNLP.UnreducedKKTVector{Float64, Vector{Float64}, Vector{Int64}}},
             MadNLP.UnreducedKKTVector{Float64, Vector{Float64}, Vector{Int64}}}
"""

function MadNLP.regular!(
    solver::MadNLPSolver{
        T,
        VT,
        VI,
        KKTSystem,
        ScholtesRelaxation,
        CB,
        Iterator,
        IC,
        KKTVec,
    },
) where {T, VT, VI, KKTSystem, CB, Iterator, IC, KKTVec}
    while true
        println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if (solver.cnt.k!=0 && !solver.opt.jacobian_constant)
            eval_jac_wrapper!(solver, solver.kkt, solver.x)
        end

        jtprod!(solver.jacl, solver.kkt, solver.y)
        sd = get_sd(solver.y, solver.zl_r, solver.zu_r, T(solver.opt.s_max))
        sc = get_sc(solver.zl_r, solver.zu_r, T(solver.opt.s_max))
        solver.inf_pr = get_inf_pr(solver.c)
        solver.inf_du =
            get_inf_du(full(solver.f), full(solver.zl), full(solver.zu), solver.jacl, sd)
        solver.inf_compl = get_inf_compl(
            solver.x_lr,
            solver.xl_r,
            solver.zl_r,
            solver.xu_r,
            solver.x_ur,
            solver.zu_r,
            zero(T),
            sc,
        )
        inf_compl_mu = get_inf_compl(
            solver.x_lr,
            solver.xl_r,
            solver.zl_r,
            solver.xu_r,
            solver.x_ur,
            solver.zu_r,
            solver.mu,
            sc,
        )

        print_iter(solver)

        # evaluate termination criteria
        MadNLP.@trace(solver.logger, "Evaluating termination criteria.")
        max(solver.inf_pr, solver.inf_du, solver.inf_compl) <= solver.opt.tol &&
            return SOLVE_SUCCEEDED
        max(solver.inf_pr, solver.inf_du, solver.inf_compl) <= solver.opt.acceptable_tol ?
        (
            solver.cnt.acceptable_cnt < solver.opt.acceptable_iter ?
            solver.cnt.acceptable_cnt+=1 : return SOLVED_TO_ACCEPTABLE_LEVEL
        ) : (solver.cnt.acceptable_cnt = 0)
        max(solver.inf_pr, solver.inf_du, solver.inf_compl) >=
        solver.opt.diverging_iterates_tol && return DIVERGING_ITERATES
        solver.cnt.k>=solver.opt.max_iter && return MAXIMUM_ITERATIONS_EXCEEDED
        time()-solver.cnt.start_time>=solver.opt.max_wall_time &&
            return MAXIMUM_WALLTIME_EXCEEDED

        # update the barrier parameter
        MadNLP.@trace(solver.logger, "Updating the barrier parameter.")
        while solver.mu != max(solver.opt.mu_min, solver.opt.tol/10) &&
            max(solver.inf_pr, solver.inf_du, inf_compl_mu) <=
            solver.opt.barrier_tol_factor*solver.mu
            mu_new = get_mu(
                solver.mu,
                solver.opt.mu_min,
                solver.opt.mu_linear_decrease_factor,
                solver.opt.mu_superlinear_decrease_power,
                solver.opt.tol,
            )
            inf_compl_mu = get_inf_compl(
                solver.x_lr,
                solver.xl_r,
                solver.zl_r,
                solver.xu_r,
                solver.x_ur,
                solver.zu_r,
                solver.mu,
                sc,
            )
            solver.tau = get_tau(solver.mu, solver.opt.tau_min)
            solver.mu = mu_new
            MadNLP.@trace(solver.logger, "Updating Scholtes relaxation parameter.")
            solver.model.sigma[] = mu_new
            empty!(solver.filter)
            push!(solver.filter, (solver.theta_max, -Inf))
        end

        # TODO(@anton) update the scholtes relaxation parameter here!
        MadNLP.@trace(solver.logger, "Get eta.")
        eta_k = get_eta(solver)

        # compute the newton step
        MadNLP.@trace(solver.logger, "Computing the newton step.")
        if (solver.cnt.k!=0 && !solver.opt.hessian_constant)
            eval_lag_hess_wrapper!(solver, solver.kkt, solver.x, solver.y)
        end

        # TODO(@anton) update solver.x solver.zl, solver.zu
        set_aug_diagonal!(solver.kkt, solver)
        set_aug_rhs!(solver, solver.kkt, solver.c)
        dual_inf_perturbation!(
            primal(solver.p),
            solver.ind_llb,
            solver.ind_uub,
            solver.mu,
            solver.opt.kappa_d,
        )

        inertia_correction!(solver.inertia_corrector, solver) || return ROBUST

        MadNLP.@trace(solver.logger, "Backtracking line search initiated.")
        status = filter_line_search!(solver)
        if status != LINESEARCH_SUCCEEDED
            return status
        end

        MadNLP.@trace(solver.logger, "Updating primal-dual variables.")
        copyto!(full(solver.x), full(solver.x_trial))
        copyto!(solver.c, solver.c_trial)
        solver.obj_val = solver.obj_val_trial
        adjust_boundary!(solver.x_lr, solver.xl_r, solver.x_ur, solver.xu_r, solver.mu)

        axpy!(solver.alpha, dual(solver.d), solver.y)

        solver.zl_r .+= solver.alpha_z .* dual_lb(solver.d)
        solver.zu_r .+= solver.alpha_z .* dual_ub(solver.d)
        reset_bound_dual!(
            primal(solver.zl),
            primal(solver.x),
            primal(solver.xl),
            solver.mu,
            solver.opt.kappa_sigma,
        )
        reset_bound_dual!(
            primal(solver.zu),
            primal(solver.xu),
            primal(solver.x),
            solver.mu,
            solver.opt.kappa_sigma,
        )

        eval_grad_f_wrapper!(solver, solver.f, solver.x)

        solver.cnt.k+=1
        MadNLP.@trace(solver.logger, "Proceeding to the next interior point iteration.")
    end
end
