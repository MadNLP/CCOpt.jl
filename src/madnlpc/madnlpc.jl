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

function solve_homotopy!(
    nlp::ST,
    solver::MadNLPCSolver;
    kwargs...,
) where {ST <: AbstractMPCCRelaxation}
    return solve_homotopy!(nlp, solver, MadNLPCExecutionStats(solver); kwargs...)
end

function solve_homotopy!(solver::MadNLPCSolver; kwargs...)
    return solve_homotopy!(solver.rnlp, solver; kwargs...)
end

function solve_homotopy!(
    nlp::ST,
    solver::MadMPEC.MadNLPCSolver,
    stats::MadNLPCExecutionStats;
    x=nothing,
    y=nothing,
    zl=nothing,
    zu=nothing,
    kwargs...,
) where {ST <: AbstractMPCCRelaxation}
    ipm = solver.ipm
    ipm.cnt.start_time = time()
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
            ipm.nlp.σ[] = ipm.mu
        else # resolving the problem
            ipm.status = MadNLP.reinitialize!(ipm)
            # Also reset sigma
            ipm.nlp.σ[] = ipm.mu
        end
        # possibly fix complementarity variable upper bounds:
        if solver.opts.respect_comp_bounds
            MadNLP.variable(ipm.xl)[solver.mpcc.meta.ind_cc1] .=
                solver.mpcc.meta.lvar[solver.mpcc.meta.ind_cc1]
            MadNLP.variable(ipm.xl)[solver.mpcc.meta.ind_cc2] .=
                solver.mpcc.meta.lvar[solver.mpcc.meta.ind_cc2]
        end

        # Now begin "Phase I"
        solver.status = PHASE_I
        while ipm.status >= MadNLP.REGULAR && solver.status ∉ [PHASE_II, NLP_STATIONARY]
            ipm.status == MadNLP.REGULAR &&
                ((ipm.status, solver.status) = MadMPEC.homotopy!(solver))
            ipm.status == MadNLP.RESTORE && (
                (ipm.status, solver.status) =
                    irregular_to_mpcc_status(MadNLP.restore!(ipm))
            )
            ipm.status == MadNLP.ROBUST && (
                (ipm.status, solver.status) =
                    irregular_to_mpcc_status(MadNLP.robust!(ipm))
            )
        end
        # Now we are either in NLP stationarity, failed, or proceeding to Phase II.
        if solver.status == PHASE_II
            phaseII!(solver, stats)
        else
            # Copy the primal solution from Phase I
            solver.x .= MadNLP.variable(ipm.x)
        end
    catch e
        if e isa MadNLP.InvalidNumberException
            if e.callback == :obj
                ipm.status=MadNLP.INVALID_NUMBER_OBJECTIVE
                solver.status = IPM_ERROR
            elseif e.callback == :grad
                ipm.status=MadNLP.INVALID_NUMBER_GRADIENT
                solver.status = IPM_ERROR
            elseif e.callback == :cons
                ipm.status=MadNLP.INVALID_NUMBER_CONSTRAINTS
                solver.status = IPM_ERROR
            elseif e.callback == :jac
                ipm.status=MadNLP.INVALID_NUMBER_JACOBIAN
                solver.status = IPM_ERROR
            elseif e.callback == :hess
                ipm.status=MadNLP.INVALID_NUMBER_HESSIAN_LAGRANGIAN
                solver.status = IPM_ERROR
            else
                ipm.status=MadNLP.INVALID_NUMBER_DETECTED
                solver.status = IPM_ERROR
            end
        elseif e isa MadNLP.NotEnoughDegreesOfFreedomException
            ipm.status=MadNLP.NOT_ENOUGH_DEGREES_OF_FREEDOM
            solver.status = IPM_ERROR
        elseif e isa MadNLP.LinearSolverException
            ipm.status=MadNLP.ERROR_IN_STEP_COMPUTATION;
            solver.status = IPM_ERROR
            ipm.opt.rethrow_error && rethrow(e)
        elseif e isa MadNLP.InterruptException
            ipm.status=MadNLP.USER_REQUESTED_STOP
            solver.status = IPM_ERROR
            ipm.opt.rethrow_error && rethrow(e)
        else
            ipm.status=MadNLP.INTERNAL_ERROR
            solver.status = IPM_ERROR
            ipm.opt.rethrow_error && rethrow(e)
        end
    finally
        ipm.cnt.total_time = time() - ipm.cnt.start_time
        if !(ipm.status < MadNLP.SOLVE_SUCCEEDED)
            MadNLP.print_summary(ipm)
        else
            solver.status = IPM_ERROR
        end

        MadNLP.@notice(
            solver.logger,
            "EXIT: $(get_status_output(solver.status, solver.opts, ipm.opt))"
        )
        ipm.opt.disable_garbage_collector && (
            GC.enable(true);
            MadNLP.@warn(ipm.logger, "Julia garbage collector is turned back on")
        )
        MadNLP.finalize(ipm.logger)
        MadNLP.finalize(solver.logger)
        finalize(solver.iterate_logger)

        update!(stats, solver)
    end

    return stats
end

function homotopy!(solver::MadNLPCSolver{T, VT}) where {T, VT}
    opts = solver.opts
    ipm = solver.ipm
    mpcc = solver.mpcc
    log_iter(solver.iterate_logger, solver) # Log initial state
    while true
        if (ipm.cnt.k!=0 && !ipm.opt.jacobian_constant)
            MadNLP.eval_jac_wrapper!(ipm, ipm.kkt, ipm.x)
        end

        # Set σ to zero for constraint infeasibility calculations
        σ = ipm.nlp.σ[]
        MadNLP.jtprod!(ipm.jacl, ipm.kkt, ipm.y)
        sd = MadNLP.get_sd(ipm.y, ipm.zl_r, ipm.zu_r, T(ipm.opt.s_max))
        sc = MadNLP.get_sc(ipm.zl_r, ipm.zu_r, T(ipm.opt.s_max))
        ipm.inf_pr = max(
            MadNLP.get_inf_pr(@view(ipm.c[1:mpcc.meta.ncon])),
            MadMPEC.get_inf_pr_cc(solver),
        )
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
            return MadNLP.SOLVE_SUCCEEDED, NLP_STATIONARY
        max(ipm.inf_pr, ipm.inf_du, ipm.inf_compl) <= ipm.opt.acceptable_tol ?
        (
            ipm.cnt.acceptable_cnt < ipm.opt.acceptable_iter ? ipm.cnt.acceptable_cnt+=1 :
            return MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL, NLP_STATIONARY
        ) : (ipm.cnt.acceptable_cnt = 0)
        max(ipm.inf_pr, ipm.inf_du, ipm.inf_compl) >= ipm.opt.diverging_iterates_tol &&
            return MadNLP.DIVERGING_ITERATES, DIVERGING_ITERATES
        ipm.cnt.k>=ipm.opt.max_iter &&
            return MadNLP.MAXIMUM_ITERATIONS_EXCEEDED, MAXIMUM_ITERATIONS_EXCEEDED
        time()-ipm.cnt.start_time>=ipm.opt.max_wall_time &&
            return MadNLP.MAXIMUM_WALLTIME_EXCEEDED, MAXIMUM_WALL_TIME_EXCEEDED

        # If using macmpec and we are feasible enough to try projection
        if opts.use_mpecopt && ipm.inf_pr <= solver.eps_proj
            MadNLP.@trace(ipm.logger, "Getting active set oracle.")
            solver.x .= MadNLP.variable(ipm.x)

            # get oracle
            valid, b = MadMPEC.phase_I_b_oracle(solver)
            if valid
                # Check if bnlp is feasilbe
                MadMPEC.build_bnlp_solver!(solver, b)
                ipm_stats = MadMPEC.solve_bnlp!(solver)

                # Check if BNLP succeeded
                if ipm_stats.status ∈
                   [MadNLP.SOLVE_SUCCEEDED, MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL]
                    solver.x .= ipm_stats.solution
                    @views begin
                        solver.x[mpcc.meta.ind_cc1[.!b]] .=
                            mpcc.meta.lvar[mpcc.meta.ind_cc1[.!b]]
                        solver.x[mpcc.meta.ind_cc2[b]] .=
                            mpcc.meta.lvar[mpcc.meta.ind_cc2[b]]
                        solver.b = b
                    end
                    println("lpec succeeded")
                    return MadNLP.REGULAR, PHASE_II
                else
                    println("lpec failed at eps=$(solver.eps_proj)")
                    solver.eps_proj = min(solver.eps_proj*opts.alpha_eps_proj)
                    println("trying again at eps=$(solver.eps_proj)")
                end
            else
                println("lpec failed at eps=$(solver.eps_proj)")
                solver.eps_proj = min(solver.eps_proj*opts.alpha_eps_proj)
                println("trying again at eps=$(solver.eps_proj)")
            end
        end
        # Now go back to using relaxed inf_pr
        ipm.inf_pr = MadNLP.get_inf_pr(ipm.c)

        MadNLP.@trace(solver.logger, "Get eta.")
        eta_k = get_eta_heuristic(solver)

        # Evaluate the hessian of the lagrangian
        MadNLP.@trace(ipm.logger, "Evaluating nabla L.")
        if (ipm.cnt.k!=0)
            MadNLP.eval_lag_hess_wrapper!(ipm, ipm.kkt, ipm.x, ipm.y)
        end
        MadNLP.set_aug_diagonal!(ipm.kkt, solver, eta_k)
        MadNLP.@trace(solver.logger, "Factorizing the KKT system.")
        MadNLP.inertia_correction!(ipm.inertia_corrector, ipm) ||
            return MadNLP.ROBUST, solver.status

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
            @views project_scholtes_explicit!(
                MadNLP.variable(ipm.x)[mpcc.meta.ind_cc1],
                MadNLP.variable(ipm.x)[mpcc.meta.ind_cc2],
                MadNLP.variable(ipm.x)[mpcc.meta.ind_cc1],
                MadNLP.variable(ipm.x)[mpcc.meta.ind_cc2],
                MadNLP.variable(ipm.xl)[mpcc.meta.ind_cc1],
                MadNLP.variable(ipm.xl)[mpcc.meta.ind_cc2],
                𝜅,
                ipm.nlp.σ[],
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
            log_iter(solver.iterate_logger, solver; magic=true)
        end

        MadNLP.@trace(ipm.logger, "Calculating the newton step.")
        # TODO(@anton) update ipm.x ipm.zl, ipm.zu
        MadNLP.set_aug_rhs!(ipm, ipm.kkt, ipm.c, ipm.mu)
        MadNLP.dual_inf_perturbation!(
            MadNLP.primal(ipm.p),
            ipm.ind_llb,
            ipm.ind_uub,
            ipm.mu,
            ipm.opt.kappa_d,
        )
        MadNLP.solve_refine_wrapper!(ipm.d, ipm, ipm.p, ipm._w4)

        MadNLP.@trace(ipm.logger, "Backtracking line search initiated.")
        status = MadNLP.filter_line_search!(ipm)
        if status != MadNLP.LINESEARCH_SUCCEEDED
            if status == MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL
                solver.status = SEARCH_DIRECTION_BECOMES_TOO_SMALL
            end
            return status, solver.status
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

function phaseII!(solver::MadNLPCSolver{T, VT}, stats::MadNLPCExecutionStats) where {T, VT}
    # TODO(@anton) this is unoptimized for now as we generate a new solver at each iteration :)
    tr = solver.opts.phase_II_tr0
    prev_obj = solver.bnlp_ipm.obj_val / solver.bnlp_ipm.cb.obj_scale[]
    mpcc = solver.mpcc
    opts = solver.opts
    MadMPEC.linearize_lpec!(solver, tr)

    # Check for S-stationarity
    @views begin # TODO(@anton) add tolerance as option or maybe use tr?
        if ~any(
            (
                solver.x[mpcc.meta.ind_cc1] .- mpcc.meta.lvar[mpcc.meta.ind_cc1] .<
                opts.s_stationarity_tol
            ) .& (
                solver.x[mpcc.meta.ind_cc2] .- mpcc.meta.lvar[mpcc.meta.ind_cc2] .<
                opts.s_stationarity_tol
            ),
        )
            solver.status = B_STATIONARY
            return
        end
    end
    while solver.status >= PHASE_II
        # Solve the corresponding LPCC
        optimal, d, b, obj =
            MadMPEC.solve_lpec!(solver; x0=vcat(zeros(mpcc.meta.nvar), solver.b))
        if optimal
            if norm(@view d[1:mpcc.meta.nvar]) <= opts.b_stationarity_tol  # TODO(@anton) make option
                solver.status = B_STATIONARY
                return
            elseif abs(obj) <= opts.b_stationarity_tol
                solver.status = B_STATIONARY
                return
            elseif (mpcc.meta.minimize && obj > 0) || (!mpcc.meta.minimize && obj < 0)
                # TODO(@anton) is this reasonable? this technically implies an infeasible point?
                solver.status = B_STATIONARY
                return
            elseif all(solver.b .== b) # TODO(@anton) this should maybe also check for "acceptable" tolerance
                #solver.status = B_STATIONARY
                #return
                tr = opts.phase_II_alpha_tr*tr # TODO(@anton) Options
                if tr <= opts.phase_II_tr_min
                    # Search direction too small
                    solver.status = SEARCH_DIRECTION_BECOMES_TOO_SMALL
                end
                MadMPEC.update_lpec_tr!(solver, tr)
                continue
            else
                solver.b .= b
            end
        else
            solver.status = LPCC_ERROR
            return
        end
        # Build bnlp
        MadMPEC.build_bnlp_solver!(solver, b)

        # Solve bnlp
        ipm_stats = MadMPEC.solve_bnlp!(solver, stats.stats)

        # Check if BNLP succeeded
        if ipm_stats.status ∈ [
            MadNLP.SOLVE_SUCCEEDED,
            MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL,
            MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL,
        ]
            if ipm_stats.objective < prev_obj # Accept step
                println("ACCEPTED PHASE II STEP")
                # update current values
                prev_obj = ipm_stats.objective
                solver.x .= ipm_stats.solution
                # Check if we even need to solve an LPCC by checking for biactives:
                @views begin # TODO(@anton) add tolerance as option or maybe use tr?
                    if ~any(
                        (solver.x[mpcc.meta.ind_cc1] .< 1e-8) .&
                        (solver.x[mpcc.meta.ind_cc2] .< 1e-8),
                    )
                        solver.status = B_STATIONARY
                        continue
                    end
                end
                # Reset the trust region
                tr = opts.phase_II_tr0
                # Linearize at the current point
                MadMPEC.linearize_lpec!(solver, tr)
            else # Otherwise we did not get descent in the BNLP, reuse linearization and a smaller tr
                tr = opts.phase_II_alpha_tr*tr # TODO(@anton) Options
                println("REJECTED PHASE II STEP")
                if tr <= opts.phase_II_tr_min
                    # Search direction too small
                    solver.status = SEARCH_DIRECTION_BECOMES_TOO_SMALL
                    continue
                end
                MadMPEC.update_lpec_tr!(solver, tr)
            end
        else
            if ipm_stats.status == MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
                solver.status = MAXIMUM_ITERATIONS_EXCEEDED
            elseif ipm_stats.status == MadNLP.MAXIMUM_WALLTIME_EXCEEDED
                solver.status = MAXIMUM_WALLTIME_EXCEEDED
            else
                solver.status = IPM_ERROR
            end
        end
    end
end

function update!(stats::MadNLPCExecutionStats, solver::MadNLPCSolver{T, VT}) where {T, VT}
    # TODO(@anton) we probably want to return a custom stats object which returns the correct statuses etc.
    ipm = solver.ipm
    bnlp_ipm = solver.bnlp_ipm
    ipm_stats = stats.stats
    if solver.ipm.status < MadNLP.REGULAR # We didn't stop the IPM early TODO(@anton) A cleaner way here would be good
        ipm_stats.status = ipm.status
        ipm_stats.solution .= @view(MadNLP.primal(ipm.x)[1:get_nvar(ipm.nlp)])
        ipm_stats.multipliers .= ipm.y[1:solver.mpcc.meta.ncon]
        ipm_stats.multipliers_L .= @view(MadNLP.primal(ipm.zl)[1:get_nvar(ipm.nlp)])
        ipm_stats.multipliers_U .= @view(MadNLP.primal(ipm.zu)[1:get_nvar(ipm.nlp)])
        ipm_stats.objective = ipm.obj_val / ipm.cb.obj_scale[]
        ipm_stats.constraints .=
            ipm.c[1:solver.mpcc.meta.ncon] ./ ipm.cb.con_scale[1:solver.mpcc.meta.ncon] .+
            ipm.rhs[1:solver.mpcc.meta.ncon]
        ind_ind_ineq = ipm.ind_ineq .∈ [1:solver.mpcc.meta.ncon]
        ipm_stats.constraints[ipm.ind_ineq[ind_ind_ineq]] .+=
            MadNLP.slack(ipm.x)[ind_ind_ineq]
        ipm_stats.dual_feas = ipm.inf_du
        ipm_stats.primal_feas = ipm.inf_pr
        MadNLP.update_z!(ipm.cb, ipm_stats.multipliers_L, ipm_stats.multipliers_U, ipm.jacl)
        ipm_stats.iter = ipm.cnt.k
    else # We stopped the IPM early and have solved BNLPs
        MadNLP.update!(ipm_stats, solver.bnlp_ipm)
    end
    stats.status = solver.status
    stats.solution = solver.x
    stats.counters.solver_time =
        stats.counters.counters.total_time - stats.counters.counters.linear_solver_time -
        stats.counters.counters.eval_function_time - stats.counters.lpcc_init_time -
        stats.counters.lpcc_solve_time - stats.counters.bnlp_init_time
    return stats
end

function robust_to_mpcc_status(status::MadNLP.Status)
    if status > MadNLP.INITIAL
        return status, PHASE_I
    else
        return status, IPM_ERROR
    end
end

function irregular_to_mpcc_status(status::MadNLP.Status)
    if status > MadNLP.INITIAL
        return status, PHASE_I
    else
        return status, IPM_ERROR
    end
end
