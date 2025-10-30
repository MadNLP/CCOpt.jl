function update_sigma!(
    relax::ProportionalRelaxationUpdate{T},
    rnlp::AbstractMPCCRelaxation{T},
    solver::MadNLPCSolver{T},
) where {T}
    ipm = solver.ipm
    mpcc = solver.mpcc
    ncc = mpcc.meta.ncc
    # update c
    ipm.c[(end-ncc+1):end] .+= get_relaxation(rnlp)
    # calculate new sigma
    sigma_candidate = relax.sigma_mu_ratio*(solver.ipm.mu^relax.sigma_mu_exp)
    if relax.monotone
        set_relaxation(
            rnlp,
            max(min(solver.rnlp.σ[1], sigma_candidate), solver.opts.sigma_min),
        )
    else
        set_relaxation(rnlp, max(sigma_candidate, solver.opts.sigma_min))
    end
    # update c
    ipm.c[(end-ncc+1):end] .-= get_relaxation(rnlp)
    # Here we assume the barrier update handles whether we throw out the filter.
    return nothing
end

function update_sigma!(
    relax::LOQORelaxationUpdate{T},
    rnlp::AbstractMPCCRelaxation{T},
    solver::MadNLPCSolver{T},
) where {T}
    ipm = solver.ipm
    mpcc = solver.mpcc
    ncc = mpcc.meta.ncc
    # update c
    ipm.c[(end-ncc+1):end] .+= get_relaxation(rnlp)
    # Calculate mean primal complementarity
    cc_pr = @views dot(
        MadNLP.variable(ipm.x)[mpcc.meta.ind_cc1] -
        MadNLP.variable(ipm.xl)[mpcc.meta.ind_cc1],
        MadNLP.variable(ipm.x)[mpcc.meta.ind_cc2] -
        MadNLP.variable(ipm.xl)[mpcc.meta.ind_cc2],
    )
    mean_cc = cc_pr/ncc
    # Calculate minimum primal complementarity
    min_cc_pr = @views mapreduce(
        (x1, xl1, x2, x2l) -> (x1-xl1)*(x2-x2l),
        min,
        MadNLP.variable(ipm.x)[mpcc.meta.ind_cc1],
        MadNLP.variable(ipm.xl)[mpcc.meta.ind_cc1],
        MadNLP.variable(ipm.x)[mpcc.meta.ind_cc2],
        MadNLP.variable(ipm.xl)[mpcc.meta.ind_cc2],
        init=T(Inf),
    )
    # Calculate the factor to multiply the mean complementarity by.
    xi = min_cc_pr/mean_cc
    gamma_sigma = max(relax.gamma_min, relax.gamma*min((1-relax.r)*((1-xi)/xi), 2)^3)
    # TODO(@anton) in principle we would like to not reduce this too much depending on how close we are to the KKT conds
    set_relaxation(
        rnlp,
        max(gamma_sigma*mean_cc, solver.opts.sigma_min, relax.mu_factor*ipm.mu),
    )
    # update c
    c[(end-ncc+1):end] .-= get_relaxation(rnlp)
    # Throw out the filter as the barrier problem has changed
    empty!(ipm.filter)
    push!(ipm.filter, (ipm.theta_max, -Inf))
    return nothing
end

function kkt_residual_norm(
    rnlp::ScholtesRelaxation{T},
    solver::MadNLPCSolver{T},
    δ1,
    δ2,
    σ,
) where {T}
    ipm = solver.ipm
    mpcc = solver.mpcc
    ncon = solver.mpcc.meta.ncon
    ncc = solver.mpcc.meta.ncc
    ind_cc1 = mpcc.meta.ind_cc1
    ind_cc2 = mpcc.meta.ind_cc2
    # Reset bounds:
    MadNLP.variable(ipm.xl)[ind_cc1] .= @view(rnlp.meta.lvar[ind_cc1]) .- δ1
    MadNLP.variable(ipm.xl)[ind_cc2] .= @view(rnlp.meta.lvar[ind_cc2]) .- δ2
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
    c = ipm.c

    px .= .-f .+ zl .- zu .- ipm.jacl
    py .= .-c
    pzl .= (ipm.xl_r .- ipm.x_lr) .* ipm.zl_r
    pzu .= (ipm.xu_r .- ipm.x_ur) .* ipm.zu_r

    return r
end

function update_sigma!(
    relax::TwoSidedScholtesUpdate{T},
    rnlp::ScholtesRelaxation{T},
    solver::MadNLPCSolver{T},
) where {T}
    ipm = solver.ipm
    mpcc = solver.mpcc
    ncc = mpcc.meta.ncc
    ncon = mpcc.meta.ncon
    ind_cc1 = mpcc.meta.ind_cc1
    ind_cc2 = mpcc.meta.ind_cc2
    # TODO(@anton) This is inefficient
    r = kkt_residual_norm(rnlp, solver, rnlp.δ1opt, rnlp.δ2opt, rnlp.σopt) # kkt norm
    rl = r^(1+relax.tau)
    ru = r^(1-relax.tau)
    MadNLP.variable(ipm.xl)[ind_cc1] .= @view(rnlp.meta.lvar[ind_cc1]) .- rnlp.δ1
    MadNLP.variable(ipm.xl)[ind_cc2] .= @view(rnlp.meta.lvar[ind_cc2]) .- rnlp.δ2
    updated = false
    ipm.c[(end-ncc+1):end] .+= rnlp.σ
    for ii in 1:ncc
        cc1 = ind_cc1[ii]
        cc2 = ind_cc2[ii]

        nu1 = solver.multipliers_cc1[ii]
        nu2 = solver.multipliers_cc2[ii]
        # These are the rules from
        if nu1 > ru
            rnlp.δ1[ii] = min(relax.kappa*rnlp.δ1[ii], rl)
            MadNLP.variable(ipm.xl)[cc1] = mpcc.meta.lvar[cc1] - rnlp.δ1[ii]
            rnlp.δ1opt[ii] = 0.0
            updated = true
        else
            rnlp.δ1opt[ii] = rnlp.δ1[ii]
        end

        if nu2 > ru
            rnlp.δ2[ii] = min(relax.kappa*rnlp.δ2[ii], rl)
            MadNLP.variable(ipm.xl)[cc2] = mpcc.meta.lvar[cc2] - rnlp.δ2[ii]
            rnlp.δ2opt[ii] = 0.0
            updated = true
        else
            rnlp.δ2opt[ii] = rnlp.δ2[ii]
        end

        if nu1 < -ru || nu2 < -ru
            if min(relax.kappa*rnlp.σ[ii], rl) >= solver.opts.sigma_min
                rnlp.σ[ii] = max(min(relax.kappa*rnlp.σ[ii], rl), solver.opts.sigma_min)
                rnlp.σopt[ii] = 0.0
                updated = true
            end
        else
            rnlp.σopt[ii] = rnlp.σ[ii]
        end
    end
    if !updated && ipm.mu <= 1e-4 # Standard rules make no progress
        for ii in 1:ncc
            cc1 = ind_cc1[ii]
            cc2 = ind_cc2[ii]
            x1 = MadNLP.variable(ipm.x)[cc1] - mpcc.meta.lvar[cc1]
            x2 = MadNLP.variable(ipm.x)[cc2] - mpcc.meta.lvar[cc2]

            if x1 <= 0 # we are lower bound infeasible:
                max_decrease =
                    (
                        relax.k_ftb
                    )*(MadNLP.variable(ipm.x)[cc1] - MadNLP.variable(ipm.xl)[cc1])
                rnlp.δ1[ii] = max(relax.kappa*rnlp.δ1[ii], rnlp.δ1[ii]-max_decrease)
                MadNLP.variable(ipm.xl)[cc1] = mpcc.meta.lvar[cc1] - rnlp.δ1[ii]
                updated = true
            end
            if x2 <= 0 # we are lower bound infeasible:
                max_decrease =
                    (
                        relax.k_ftb
                    )*(MadNLP.variable(ipm.x)[cc2] - MadNLP.variable(ipm.xl)[cc2])
                rnlp.δ2[ii] = max(relax.kappa*rnlp.δ2[ii], rnlp.δ2[ii]-max_decrease)
                MadNLP.variable(ipm.xl)[cc2] = mpcc.meta.lvar[cc2] - rnlp.δ2[ii]
                updated = true
            end
            if x1 >= 0 && x2 >= 0 && x1*x2 >= solver.ipm.opt.tol
                rnlp.σ[ii] = max(relax.kappa*rnlp.σ[ii], solver.opts.sigma_min)
                updated = true
            end
        end
    end
    ipm.c[(end-ncc+1):end] .-= rnlp.σ
    if updated
        empty!(ipm.filter)
        push!(ipm.filter, (ipm.theta_max, -Inf))
    end
end

function update_sigma!(
    relax::RelaxLBUpdate{T},
    rnlp::ScholtesRelaxation{T},
    solver::MadNLPCSolver{T},
) where {T}
    ipm = solver.ipm
    mpcc = solver.mpcc
    ncc = mpcc.meta.ncc
    ncon = mpcc.meta.ncon
    ind_cc1 = mpcc.meta.ind_cc1
    ind_cc2 = mpcc.meta.ind_cc2
    # update c
    ipm.c[(end-ncc+1):end] .+= get_relaxation(rnlp)
    # calculate new sigma
    sigma_candidate = relax.sigma_mu_ratio*(solver.ipm.mu^relax.sigma_mu_exp)
    if relax.monotone
        set_relaxation(
            rnlp,
            max(min(solver.rnlp.σ[1], sigma_candidate), solver.opts.sigma_min),
        )
    else
        set_relaxation(rnlp, max(sigma_candidate, solver.opts.sigma_min))
    end
    # update c
    ipm.c[(end-ncc+1):end] .-= rnlp.σ

    if ipm.mu <= relax.relax_threshold # check if we need to relax bounds
        for ii in 1:ncc
            cc1 = ind_cc1[ii]
            cc2 = ind_cc2[ii]
            nu1 = solver.multipliers_cc1[ii]
            nu2 = solver.multipliers_cc2[ii]

            if nu1 <= -((ipm.mu)^relax.tau)
                rnlp.δ1[ii] = relax.mu_factor*ipm.mu
                MadNLP.variable(ipm.xl)[cc1] = mpcc.meta.lvar[cc1] - rnlp.δ1[ii]
            elseif relax.unrelax && rnlp.δ1[ii] > 0 && nu1 >= ((ipm.mu)^relax.tau)
                max_decrease =
                    relax.k_ftb*(MadNLP.variable(ipm.x)[cc1] - MadNLP.variable(ipm.xl)[cc1])
                rnlp.δ1[ii] = max(0.0, rnlp.δ1[ii]-max_decrease)
                MadNLP.variable(ipm.xl)[cc1] = mpcc.meta.lvar[cc1] - rnlp.δ1[ii]
            end
            if nu2 <= -((ipm.mu)^relax.tau)
                rnlp.δ2[ii] = relax.mu_factor*ipm.mu
                MadNLP.variable(ipm.xl)[cc2] = mpcc.meta.lvar[cc2] - rnlp.δ2[ii]
            elseif relax.unrelax && rnlp.δ2[ii] > 0 && nu2 >= ((ipm.mu)^relax.tau)
                max_decrease =
                    relax.k_ftb*(MadNLP.variable(ipm.x)[cc2] - MadNLP.variable(ipm.xl)[cc2])
                rnlp.δ2[ii] = max(0.0, rnlp.δ2[ii]-max_decrease)
                MadNLP.variable(ipm.xl)[cc2] = mpcc.meta.lvar[cc2] - rnlp.δ2[ii]
            end
        end
    end

    # Here we assume the barrier update handles whether we throw out the filter.
    return nothing
end
