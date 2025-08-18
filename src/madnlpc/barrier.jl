function MadNLP.update_barrier!(
    barrier::MadNLP.LOQOUpdate{T},
    solver::MadNLPCSolver{T},
    sc::T,
) where {T}
    ipm = solver.ipm
    if !solver.opts.use_specialized_barrier_update
        return MadNLP.update_barrier!(barrier, ipm, sc)
    end
    mu = MadNLP.get_average_complementarity(solver) # get average complementarity.
    ncc = ipm.nlb + ipm.nub
    min_cc = MadNLP.get_min_complementarity(solver)
    xi = min_cc/mu
    sigma = barrier.gamma*min((1-barrier.r)*((1-xi)/xi), 2)^3
    new_mu = max(sigma*mu, barrier.mu_min)
    if ipm.mu != new_mu # Stop clearing if we reached mu_min
        empty!(ipm.filter)
        push!(ipm.filter, (ipm.theta_max, -Inf))
    end
    ipm.mu = max(sigma*mu, barrier.mu_min)

    return nothing
end

function MadNLP.update_barrier!(
    barrier::MadNLP.MonotoneUpdate{T},
    solver::MadNLPCSolver{T},
    sc::T,
) where {T}
    return MadNLP.update_barrier!(barrier, solver.ipm, sc)
end

# Set RHS
function set_aug_rhs_aff!(
    solver::MadNLPCSolver,
    kkt::MadNLP.AbstractKKTSystem,
    c::AbstractVector,
)
    ipm = solver.ipm
    px = MadNLP.primal(ipm.p)
    x = MadNLP.primal(ipm.x)
    f = MadNLP.primal(ipm.f)
    xl = MadNLP.primal(ipm.xl)
    xu = MadNLP.primal(ipm.xu)
    zl = MadNLP.full(ipm.zl)
    zu = MadNLP.full(ipm.zu)
    py = MadNLP.dual(ipm.p)
    pzl = MadNLP.dual_lb(ipm.p)
    pzu = MadNLP.dual_ub(ipm.p)

    ncon = solver.mpcc.meta.ncon
    ncc = solver.mpcc.meta.ncc

    px .= .-f .+ zl .- zu .- ipm.jacl
    py .= .-c
    py[(ncon+1):(ncon+ncc)] .-= solver.rnlp.σ[]
    pzl .= (ipm.xl_r .- ipm.x_lr) .* ipm.zl_r
    pzu .= (ipm.xu_r .- ipm.x_ur) .* ipm.zu_r
    return
end

function MadNLP.set_cen_aug_rhs!(solver::MadNLPCSolver, kkt::MadNLP.AbstractKKTSystem, mu)
    ipm = solver.ipm
    px = MadNLP.primal(ipm.p)
    py = MadNLP.dual(ipm.p)
    pzl = MadNLP.dual_lb(ipm.p)
    pzu = MadNLP.dual_ub(ipm.p)

    px .= 0
    py .= mu
    pzl .= mu
    pzu .= -mu
    return
end

function MadNLP.get_adaptive_mu(solver::MadNLPCSolver, barrier::MadNLP.AdaptiveUpdate)
    ipm = solver.ipm
    linear_solver = ipm.kkt.linear_solver
    step_aff = ipm._w1 # buffer 1
    step_cen = ipm._w2 # buffer 2

    # Affine step
    set_aug_rhs_aff!(solver, ipm.kkt, ipm.c)
    # Get primal and dual infeasibility directly 1from the values in RHS p
    res_primal = norm(MadNLP.primal(ipm.p))
    res_dual = norm(MadNLP.dual(ipm.p))

    # Get approximate solution without iterative refinement
    copyto!(MadNLP.full(step_aff), MadNLP.full(ipm.p))
    MadNLP.solve!(linear_solver, MadNLP.full(step_aff))

    # Get average complementarity
    mu = MadNLP.get_average_complementarity(solver)
    # Centering step
    MadNLP.set_cen_aug_rhs!(solver, ipm.kkt, mu)
    # NOTE(@anton) Ipopt also applies the dual infeasibility perturbation for some reason???
    MadNLP.dual_inf_perturbation!(
        MadNLP.primal(ipm.p),
        ipm.ind_llb,
        ipm.ind_uub,
        mu,
        ipm.opt.kappa_d,
    )
    # Get (again) approximate solution without iterative refinement
    copyto!(MadNLP.full(step_cen), MadNLP.full(ipm.p))
    MadNLP.solve!(linear_solver, MadNLP.full(step_cen))

    # Refine the search interval using Ipopt's heuristics
    # First, check if sigma is greater than 1.
    phi1 = MadNLP._evaluate_quality_function(
        ipm,
        1.0,
        step_aff,
        step_cen,
        res_primal,
        res_dual,
    )
    sigma_1m = 1.0 - 1e-4
    phi1m = MadNLP._evaluate_quality_function(
        ipm,
        sigma_1m,
        step_aff,
        step_cen,
        res_primal,
        res_dual,
    )
    # Restrict search interval
    if phi1m > phi1
        sigma_min = 1.0
        sigma_max = min(barrier.sigma_max, barrier.mu_max / mu)
    else
        sigma_min = max(barrier.sigma_min, barrier.mu_min / mu)
        sigma_max = min(max(sigma_min, sigma_1m), barrier.mu_max / mu)
    end

    # Run Golden-section search (assume the quality function is unimodal)
    sigma_opt = MadNLP._run_golden_search!(
        ipm,
        barrier,
        sigma_min,
        sigma_max,
        step_aff,
        step_cen,
        res_primal,
        res_dual,
    )
    return sigma_opt * mu
end

function MadNLP.update_barrier!(
    barrier::MadNLP.QualityFunctionUpdate{T},
    solver::MadNLPCSolver{T},
    sc::T,
) where {T}
    # TODO(@anton) also specialize this for mpcc
    ipm = solver.ipm
    kappa_1 = T(1e-5)
    kappa_2 = T(1.0)

    # TODO: implement fixed mode
    mu = MadNLP.get_adaptive_mu(solver, barrier)
    # TODO: check sufficient progress using filter line-search
    # theta = NaN
    # varphi = NaN
    # delta = NaN
    # progress = is_filter_acceptable(solver.filter, theta + delta, varphi + delta)

    # Just a sketch for now
    # if barrier.free_mode
    #     if progress
    #         mu = get_adaptive_mu(solver.mu)
    #     else
    #         barrier.free_mode = false
    #         # Get initial fixed barrier
    #         mu = barrier_fixed_mu(barrier)
    #     end
    # else
    #     if progress
    #         barrier.free_mode = true
    #     else
    #         # Monotone update
    #         # TODO
    #     end
    # end

    # Update tau
    ipm.mu = max(mu, barrier.mu_min)
    ipm.tau = MadNLP.get_tau(ipm.mu, ipm.opt.tau_min)
    # Reset filter line-search
    empty!(ipm.filter)
    return push!(ipm.filter, (ipm.theta_max, -Inf))
end
