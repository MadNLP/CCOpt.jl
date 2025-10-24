@kwdef struct ExactPenaltyOptions{T}
    # complementarity homotopy options
    tau_0::T = 1.0
    tau_max::T = 1e8
    tau_growth_rate::T = 10.0
    gamma::T = 0.4

    # Algorithm options
    dynamic_tau_update::Bool = false # Switch between classic and dynamic algorithm from
    # Leyffer2006 paper
    comp_history_length::Int = 5 # Length of history buffer (default from Leyffer2006)
    eta_dynamic_update::T = 0.99 # "sufficient decrease" parameter (default from Leyffer2006)

    # regularization
    kkt_regularization::Symbol = :none
    min_eig_value::T = 1e-4
    max_eig_value::T = Inf
    critical_rho_factor::T = 0.9
    min_reg_mu::T = 1e-5

    # Output options
    output_file::String = ""
    print_level::MadNLP.LogLevels = MadNLP.INFO
    file_print_level::MadNLP.LogLevels = MadNLP.INFO
end

mutable struct ExactPenaltySolver{T, VT}
    mpcc::AbstractMPCCModel{T, VT}
    ell1::Ell1Relaxation{T, VT}
    ipm::MadNLP.MadNLPSolver{T, VT}
    logger::MadNLP.MadNLPLogger
    opts::ExactPenaltyOptions{T}

    inf_pr_cc::T

    pr_comp_hist::CircularBuffer{T} # Complementarity history
end

"""
  This solver implements the two algorithms in the paper:
  Interior Methods for Mathematical Programs with Complementarity Constraints
  Sven Leyffer, Gabriel López-Calva, and Jorge Nocedal
  SIAM Journal on Optimization Vol. 17, Issue 1 (2006)
  epubs.siam.org/doi/10.1137/040621065
"""
function ExactPenaltySolver(
    mpcc::AbstractMPCCModel{T, VT};
    solver_opts=ExactPenaltyOptions{Float64}(),
    ipm_options...,
) where {T, VT}
    ell1 = Ell1Relaxation(mpcc)
    ipm = MadNLP.MadNLPSolver(ell1; ipm_options...)

    logger = MadNLP.MadNLPLogger(
        print_level=solver_opts.print_level,
        file_print_level=solver_opts.file_print_level,
        file=solver_opts.output_file == "" ? nothing : open(solver_opts.output_file, "w+"),
    )

    pr_comp_hist = CircularBuffer{T}(solver_opts.comp_history_length)
    return ExactPenaltySolver(mpcc, ell1, ipm, logger, solver_opts, pr_comp_hist)
end

function MadNLP.set_aug_diagonal!(
    kkt::MadNLP.AbstractKKTSystem{T},
    solver::ExactPenaltySolver{T, VT},
) where {T, VT}
    ipm = solver.ipm
    n = length(ipm.x_ur)
    ncc = solver.mpcc.meta.ncc
    nnzh = solver.mpcc.meta.nnzh

    fill!(kkt.reg, zero(T))
    fill!(kkt.du_diag, zero(T))
    kkt.l_diag .= ipm.xl_r .- ipm.x_lr   # (Xˡ - X)
    kkt.u_diag .= ipm.x_ur .- ipm.xu_r   # (X - Xᵘ)
    copyto!(kkt.l_lower, ipm.zl_r)
    copyto!(kkt.u_lower, ipm.zu_r)

    MadNLP._set_aug_diagonal!(kkt)

    return
end

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
        solver.inf_pr_cc = inf_pr_comp
        push!(solver.pr_comp_hist, inf_pr_comp_sum)

        MadNLP.print_iter(solver)

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
           isfull(solver.pr_comp_hist) &&
           inf_pr_comp > eps_pr_comp &&
           inf_pr_comp_sum > solver.opts.eta_dynamic_update*maximum(solver.pr_comp_hist) &&
           nlp.tau[] < solver.opts.tau_max
            nlp.tau[] = solver.opts.tau_growth_rate*nlp.tau[]
            MadNLP.@trace(
                solver.logger,
                "Updating the penalty parameter dynamically to $(nlp.tau[])."
            )
            ipm.obj_val = MadNLP.eval_f_wrapper(ipm, ipm.x)
            # Also clear the filter
            #empty!(solver.pr_comp_hist)
            empty!(ipm.filter)
            push!(ipm.filter, (ipm.theta_max, -Inf))
        end

        MadNLP.@trace(solver.logger, "Evaluating the lagrangian hessian.")
        if (ipm.cnt.k!=0)
            MadNLP.eval_lag_hess_wrapper!(ipm, ipm.kkt, ipm.x, ipm.y)
        end
        # update the barrier parameter
        MadNLP.@trace(ipm.logger, "Updating the barrier parameter.")
        mu_updated = false
        mu_old = ipm.mu
        MadNLP.update_barrier!(ipm.opt.barrier, solver.ipm, sc)
        mu_updated = ipm.mu != mu_old
        # Standard check
        # FIXME(@anton) This update happening _after_ the matrix has been factorized means
        #               we are still using the old $\tau$ for one iteration. This in principle
        #               does not impact convergence guarantees, but is inaccurate.
        #               In principle this also means we need to possibly factorize twice in the case
        #               of the QualityFunctionUpdate (though this may not even make sense and we may
        #               only use the adaptive $\tau$ update in this case).
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
                MadNLP.@trace(
                    solver.logger,
                    "Evaluating the lagrangian hessian (again because penalty was updated."
                )
                # TODO(@anton) we can do this as a vector assignment
                if (ipm.cnt.k!=0)
                    MadNLP.eval_lag_hess_wrapper!(ipm, ipm.kkt, ipm.x, ipm.y)
                end
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
                MadNLP.@trace(
                    solver.logger,
                    "Evaluating the lagrangian hessian (again because penalty was updated."
                )
                # TODO(@anton) we can do this as a vector assignment
                if (ipm.cnt.k!=0)
                    MadNLP.eval_lag_hess_wrapper!(ipm, ipm.kkt, ipm.x, ipm.y)
                end
            end
        end

        # compute the newton step
        MadNLP.@trace(ipm.logger, "Computing the newton step.")
        MadNLP.set_aug_diagonal!(ipm.kkt, solver)
        MadNLP.set_aug_rhs!(ipm, ipm.kkt, ipm.c, ipm.mu)
        MadNLP.dual_inf_perturbation!(
            MadNLP.primal(ipm.p),
            ipm.ind_llb,
            ipm.ind_uub,
            ipm.mu,
            ipm.opt.kappa_d,
        )
        MadNLP.inertia_correction!(ipm.inertia_corrector, solver) || return MadNLP.ROBUST

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

function regularize_Q!(solver::ExactPenaltySolver{T}) where {T}
    if solver.opts.kkt_regularization == :none || solver.ipm.mu < solver.opts.min_reg_mu
        return false
    end

    ipm = solver.ipm
    rnlp = solver.ell1
    kkt = solver.ipm.kkt
    n = length(ipm.x_ur)
    ncc = solver.mpcc.meta.ncc
    nnzh = solver.mpcc.meta.nnzh
    tau = solver.ell1.tau[]
    ind_cc1 = solver.mpcc.meta.ind_cc1
    ind_cc2 = solver.mpcc.meta.ind_cc2
    A = Array{T}(undef, 2, 2)
    regularized = false
    for i in 1:ncc
        cc1 = ind_cc1[i]
        cc2 = ind_cc2[i]

        # TODO(@anton) figure out numerical stability here
        # tr = kkt.pr_diag[cc1] + kkt.pr_diag[cc2]
        # det = kkt.pr_diag[cc1] * kkt.pr_diag[cc2] - tau^2
        # lam1 = (tr + sqrt(tr^2 - 4*det))/2
        # lam2 = 2*det/((tr - sqrt(tr^2 - 4*det)))
        # v11 = (lam2 - kkt.pr_diag[cc2])/tau
        # v21 = (lam1 - kkt.pr_diag[cc2])/tau
        if solver.opts.kkt_regularization == :eigenvalue_decomposition
            A[1, 1] = kkt.pr_diag[cc1]
            A[2, 2] = kkt.pr_diag[cc2]
            A[2, 1] = tau
            A[1, 2] = tau
            E = eigen(Symmetric(A))
            if E.values[1] < 0 || E.values[2] > solver.opts.max_eig_value
                E.values[1] = solver.opts.min_eig_value
                E.values[2] = min(solver.opts.max_eig_value, E.values[2])
                A .= Symmetric(Matrix(E))
                kkt.reg[cc1] = A[1, 1] - kkt.pr_diag[cc1]
                kkt.reg[cc2] = A[2, 2] - kkt.pr_diag[cc2]
                kkt.pr_diag[cc1] = A[1, 1]
                kkt.pr_diag[cc2] = A[2, 2]
                kkt.hess_raw.V[nnzh+i] = A[1, 2]
                regularized = true
            end
        elseif solver.opts.kkt_regularization == :critical_rho
            rho_max = sqrt(kkt.pr_diag[cc1]*kkt.pr_diag[cc2])
            if tau > rho_max
                kkt.hess_raw.V[nnzh+i] =
                    solver.opts.critical_rho_factor*rho_max*(
                        rnlp.meta.minimize ? one(T) : -one(T)
                    )
                #kkt.hess_raw.V[nnzh+i] = (1-ipm.mu)*rho_max*(rnlp.meta.minimize ? one(T) : -one(T))
                regularized = true
            end
        end
    end
    # We modify hess_raw so need to compress_hessian again.
    MadNLP.compress_hessian!(kkt)
    return regularized
end

function unregularize_Q!(solver::ExactPenaltySolver{T}) where {T}
    ipm = solver.ipm
    rnlp = solver.ell1
    kkt = solver.ipm.kkt
    n = length(ipm.x_ur)
    ncc = solver.mpcc.meta.ncc
    nnzh = solver.mpcc.meta.nnzh
    tau = solver.ell1.tau[]
    ind_cc1 = solver.mpcc.meta.ind_cc1
    ind_cc2 = solver.mpcc.meta.ind_cc2
    A = Array{T}(undef, 2, 2)
    regularized = false
    kkt.pr_diag[ind_cc1] .-= kkt.reg[ind_cc1]
    kkt.pr_diag[ind_cc2] .-= kkt.reg[ind_cc2]
    kkt.reg[ind_cc1] .= 0
    kkt.reg[ind_cc2] .= 0
    kkt.hess_raw.V[(nnzh+1):(nnzh+ncc)] .= rnlp.meta.minimize ? tau : -tau
    # We modify hess_raw so need to compress_hessian again.
    MadNLP.compress_hessian!(kkt)
    return regularized
end

function MadNLP.inertia_correction!(
    inertia_corrector::MadNLP.InertiaBased,
    solver::ExactPenaltySolver{T},
) where {T}
    ipm = solver.ipm
    n_trial = 0
    ipm.del_w = del_w_prev = zero(T)
    ipm.del_c = del_c_prev = zero(T)

    MadNLP.@trace(ipm.logger, "Inertia-based regularization started.")

    MadNLP.factorize_wrapper!(ipm)
    num_pos, num_zero, num_neg = MadNLP.inertia(ipm.kkt.linear_solver)

    solve_status = if MadNLP.is_inertia_correct(ipm.kkt, num_pos, num_zero, num_neg)
        # Try a backsolve. If the factorization has failed, solve_refine_wrapper returns false.
        MadNLP.solve_refine_wrapper!(ipm.d, ipm, ipm.p, ipm._w4)
    else
        false
    end

    # Try to regularize
    if !solve_status && regularize_Q!(solver)
        MadNLP.@trace(ipm.logger, "Trying to regularize out penalty.")
        MadNLP.factorize_wrapper!(ipm)
        num_pos, num_zero, num_neg = MadNLP.inertia(ipm.kkt.linear_solver)

        solve_status = if MadNLP.is_inertia_correct(ipm.kkt, num_pos, num_zero, num_neg)
            # Try a backsolve. If the factorization has failed, solve_refine_wrapper returns false.
            MadNLP.solve_refine_wrapper!(ipm.d, ipm, ipm.p, ipm._w4)
        else
            false
        end

        if !solve_status
            # undo what we did and just do inertia correction
            unregularize_Q!(solver)
        end
    end
    while !solve_status
        MadNLP.@debug(ipm.logger, "Primal-dual perturbed.")

        if n_trial == 0
            ipm.del_w =
                ipm.del_w_last==zero(T) ? ipm.opt.first_hessian_perturbation :
                max(
                    ipm.opt.min_hessian_perturbation,
                    ipm.opt.perturb_dec_fact*ipm.del_w_last,
                )
        else
            ipm.del_w *=
                ipm.del_w_last==zero(T) ? ipm.opt.perturb_inc_fact_first :
                ipm.opt.perturb_inc_fact
            if ipm.del_w>ipm.opt.max_hessian_perturbation
                ipm.cnt.k+=1
                MadNLP.@debug(
                    ipm.logger,
                    "Primal regularization is too big. Switching to restoration phase."
                )
                return false
            end
        end
        ipm.del_c =
            num_zero == 0 ? zero(T) :
            ipm.opt.jacobian_regularization_value *
            ipm.mu^(ipm.opt.jacobian_regularization_exponent)
        MadNLP.regularize_diagonal!(ipm.kkt, ipm.del_w - del_w_prev, ipm.del_c - del_c_prev)
        del_w_prev = ipm.del_w
        del_c_prev = ipm.del_c

        MadNLP.factorize_wrapper!(ipm)
        num_pos, num_zero, num_neg = MadNLP.inertia(ipm.kkt.linear_solver)

        solve_status = if MadNLP.is_inertia_correct(ipm.kkt, num_pos, num_zero, num_neg)
            MadNLP.solve_refine_wrapper!(ipm.d, ipm, ipm.p, ipm._w4)
        else
            false
        end

        n_trial += 1
    end

    ipm.del_w != 0 && (ipm.del_w_last = ipm.del_w)
    return true
end

function MadNLP.inertia_correction!(
    inertia_corrector::MadNLP.AbstractInertiaCorrector,
    solver::ExactPenaltySolver{T},
) where {T}
    return MadNLP.inertia_correction!(inertia_corrector, solver.ipm)
end
