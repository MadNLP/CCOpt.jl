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
    py .= mu # TODO AHHHHHHHHH
    pzl .= mu
    pzu .= -mu
    return
end

function MadNLP._evaluate_quality_function(
    solver::MadNLPCSolver,
    sigma,
    step_aff,
    step_cen,
    res_dual,
    res_primal,
)
    ipm = solver.ipm
    n, m = ipm.n, ipm.m
    ncc = solver.mpcc.meta.ncc
    ind_cc1 = solver.mpcc.meta.ind_cc1
    ind_cc2 = solver.mpcc.meta.ind_cc2
    nlb, nub = ipm.nlb, ipm.nub
    tau = ipm.tau
    d = ipm.d # Load buffer
    x_pr = MadNLP.variable(ipm.x)
    xl_pr = MadNLP.variable(ipm.xl)
    dx_pr = MadNLP.primal(ipm.d)

    # Δ(σ) = Δ(0) + σ (Δ(1) - Δ(0))
    MadNLP.full(d) .=
        MadNLP.full(step_aff) .+ sigma .* (MadNLP.full(step_cen) .- MadNLP.full(step_aff))
    # Primal step
    alpha_pr = MadNLP.get_alpha_max(
        MadNLP.primal(ipm.x),
        MadNLP.primal(ipm.xl),
        MadNLP.primal(ipm.xu),
        MadNLP.primal(d),
        tau,
    )
    # Dual step
    alpha_du =
        MadNLP.get_alpha_z(ipm.zl_r, ipm.zu_r, MadNLP.dual_lb(d), MadNLP.dual_ub(d), tau)

    # (x + αp Δx - xl)ᵀ (zl + αd Δzl)
    inf_compl_lb = mapreduce(
        (x, xl, dx, z, dz) -> ((x + alpha_pr * dx - xl) * (z + alpha_du * dz))^2,
        +,
        ipm.x_lr,
        ipm.xl_r,
        ipm.dx_lr,
        ipm.zl_r,
        MadNLP.dual_lb(d);
        init=0.0,
    )
    # (xu - x - αp Δx)ᵀ (zu + αd Δzu)
    inf_compl_ub = mapreduce(
        (x, xu, dx, z, dz) -> ((xu - x - alpha_pr * dx) * (z + alpha_du * dz))^2,
        +,
        ipm.x_ur,
        ipm.xu_r,
        ipm.dx_ur,
        ipm.zu_r,
        MadNLP.dual_ub(d);
        init=0.0,
    )
    @views begin
        inf_compl_pr =
            mapreduce(
                (x1, x1l, x2, x2l, dx1, dx2) ->
                    ((x1 + (alpha_pr * dx1) - x1l) * (x2 + (alpha_pr * dx2) - x2l))^2,
                +,
                x_pr[ind_cc1],
                xl_pr[ind_cc1],
                x_pr[ind_cc2],
                xl_pr[ind_cc2],
                dx_pr[ind_cc1],
                dx_pr[ind_cc2];
                init=0.0,
            ) / ncc
    end

    # Primal infeasibility
    inf_pr = (1.0 - alpha_pr)^2 * res_primal^2 / n
    # Dual infeasibility
    inf_du = (1.0 - alpha_du)^2 * res_dual^2 / (m-ncc)
    # Complementarity infeasibility
    inf_compl = (inf_compl_lb + inf_compl_ub) / (nlb + nub)

    # Quality function qL defined in Eq. (4.2)
    return inf_du + inf_pr + inf_compl + inf_compl_pr
end

function MadNLP._run_golden_search!(
    solver::MadNLPCSolver,
    barrier,
    sigma_lb,
    sigma_ub,
    step_aff,
    step_cen,
    res_primal,
    res_dual,
)
    ipm = solver.ipm
    gfac = 0.5 * (3.0 - sqrt(5.0))

    sigma_1, sigma_2 = sigma_lb, sigma_ub
    phi_1 = MadNLP._evaluate_quality_function(
        solver,
        sigma_1,
        step_aff,
        step_cen,
        res_primal,
        res_dual,
    )
    phi_2 = MadNLP._evaluate_quality_function(
        solver,
        sigma_2,
        step_aff,
        step_cen,
        res_primal,
        res_dual,
    )

    sigma_mid1 = sigma_lb + gfac * (sigma_ub - sigma_lb)
    sigma_mid2 = sigma_lb + (1.0 - gfac) * (sigma_ub - sigma_lb)
    phi_mid1 = MadNLP._evaluate_quality_function(
        solver,
        sigma_mid1,
        step_aff,
        step_cen,
        res_primal,
        res_dual,
    )
    phi_mid2 = MadNLP._evaluate_quality_function(
        solver,
        sigma_mid2,
        step_aff,
        step_cen,
        res_primal,
        res_dual,
    )

    # Golden search
    for i in 1:barrier.max_gs_iter
        if phi_mid1 > phi_mid2
            sigma_1 = sigma_mid1
            phi_1 = phi_mid1
            sigma_mid1 = sigma_mid2
            sigma_mid2 = sigma_1 + (1.0 - gfac) * (sigma_2 - sigma_1)
            phi_mid2 = MadNLP._evaluate_quality_function(
                solver,
                sigma_mid2,
                step_aff,
                step_cen,
                res_primal,
                res_dual,
            )
        else
            sigma_2 = sigma_mid2
            phi_2 = phi_mid2
            sigma_mid2 = sigma_mid1
            sigma_mid1 = sigma_1 + gfac * (sigma_2 - sigma_1)
            phi_mid1 = MadNLP._evaluate_quality_function(
                solver,
                sigma_mid1,
                step_aff,
                step_cen,
                res_primal,
                res_dual,
            )
        end

        if sigma_2 - sigma_1 < barrier.sigma_tol * sigma_2
            break
        end
    end
    # Compute final sigma
    sigma, phi = phi_mid1 < phi_mid2 ? (sigma_mid1, phi_mid1) : (sigma_mid2, phi_mid2)
    return sigma
end

function MadNLP.get_adaptive_mu(solver::MadNLPCSolver, barrier::MadNLP.AdaptiveUpdate)
    ipm = solver.ipm
    linear_solver = ipm.kkt.linear_solver
    step_aff = ipm._w1 # buffer 1
    step_cen = ipm._w2 # buffer 2
    # Affine step
    set_aug_rhs_aff!(solver, ipm.kkt, ipm.c)
    # Get primal and dual infeasibility directly 1from the values in RHS p
    res_primal = norm(@view(MadNLP.dual(ipm.p)[1:solver.mpcc.meta.ncon]))
    res_dual = norm(MadNLP.primal(ipm.p))

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
        solver,
        1.0,
        step_aff,
        step_cen,
        res_primal,
        res_dual,
    )
    sigma_1m = 1.0 - 1e-4
    phi1m = MadNLP._evaluate_quality_function(
        solver,
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
        solver,
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
