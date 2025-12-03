function update_sigma!(
    relax::ProportionalRelaxationUpdate{T},
    rnlp::AbstractMPCCRelaxation{T},
    solver::MadNLPCSolver{T},
) where {T}
    ipm = solver.ipm
    mpcc = solver.mpcc
    ncc = get_ncc(mpcc)
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
    ncc = get_ncc(mpcc)
    # update c
    ipm.c[(end-ncc+1):end] .+= get_relaxation(rnlp)
    # Calculate mean primal complementarity
    cc_pr = @views dot(
        MadNLP.variable(ipm.x)[solver.ind_cc1] - MadNLP.variable(ipm.xl)[solver.ind_cc1],
        MadNLP.variable(ipm.x)[solver.ind_cc2] - MadNLP.variable(ipm.xl)[solver.ind_cc2],
    )
    mean_cc = cc_pr/ncc
    # Calculate minimum primal complementarity
    min_cc_pr = @views mapreduce(
        (x1, xl1, x2, x2l) -> (x1-xl1)*(x2-x2l),
        min,
        MadNLP.variable(ipm.x)[solver.ind_cc1],
        MadNLP.variable(ipm.xl)[solver.ind_cc1],
        MadNLP.variable(ipm.x)[solver.ind_cc2],
        MadNLP.variable(ipm.xl)[solver.ind_cc2],
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
    ncon = get_ncon(solver.mpcc)
    ncc = get_ncc(solver.mpcc)
    ind_cc1 = solver.ind_cc1
    ind_cc2 = solver.ind_cc2
    # Reset bounds:
    # NOTE(@anton): HERE BE DRAGONS! Due to using the original lvars we need to use two different
    #               index sets, due to fixed variable treatment
    MadNLP.variable(ipm.xl)[ind_cc1] .= @view(rnlp.meta.lvar[get_ind_cc1(mpcc)]) .- δ1
    MadNLP.variable(ipm.xl)[ind_cc2] .= @view(rnlp.meta.lvar[get_ind_cc2(mpcc)]) .- δ2
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
    ncc = get_ncc(mpcc)
    ncon = get_ncon(mpcc)
    ind_cc1 = solver.ind_cc1
    ind_cc2 = solver.ind_cc2
    # TODO(@anton) This is inefficient
    r = kkt_residual_norm(rnlp, solver, rnlp.δ1opt, rnlp.δ2opt, rnlp.σopt) # kkt norm
    rl = r^(1+relax.tau)
    ru = r^(1-relax.tau)
    MadNLP.variable(ipm.xl)[ind_cc1] .= @view(rnlp.meta.lvar[get_ind_cc1(mpcc)]) .- rnlp.δ1
    MadNLP.variable(ipm.xl)[ind_cc2] .= @view(rnlp.meta.lvar[get_ind_cc2(mpcc)]) .- rnlp.δ2
    updated = false
    ipm.c[(end-ncc+1):end] .+= rnlp.σ
    for ii in 1:ncc
        cc1 = ind_cc1[ii]
        cc2 = ind_cc2[ii]
        cc1_orig = get_ind_cc1(mpcc)[ii]
        cc2_orig = get_ind_cc2(mpcc)[ii]

        nu1 = solver.multipliers_cc1[ii]
        nu2 = solver.multipliers_cc2[ii]
        # These are the rules from
        if nu1 > ru
            rnlp.δ1[ii] = min(relax.kappa*rnlp.δ1[ii], rl)
            MadNLP.variable(ipm.xl)[cc1] = get_lvar(mpcc)[cc1_orig] - rnlp.δ1[ii]
            rnlp.δ1opt[ii] = 0.0
            updated = true
        else
            rnlp.δ1opt[ii] = rnlp.δ1[ii]
        end

        if nu2 > ru
            rnlp.δ2[ii] = min(relax.kappa*rnlp.δ2[ii], rl)
            MadNLP.variable(ipm.xl)[cc2] = get_lvar(mpcc)[cc2_orig] - rnlp.δ2[ii]
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
            cc1_orig = get_ind_cc1(mpcc)[ii]
            cc2_orig = get_ind_cc2(mpcc)[ii]
            x1 = MadNLP.variable(ipm.x)[cc1] - get_lvar(mpcc)[cc1_orig]
            x2 = MadNLP.variable(ipm.x)[cc2] - get_lvar(mpcc)[cc2_orig]

            if x1 <= 0 # we are lower bound infeasible:
                max_decrease =
                    (
                        relax.k_ftb
                    )*(MadNLP.variable(ipm.x)[cc1] - MadNLP.variable(ipm.xl)[cc1])
                rnlp.δ1[ii] = max(relax.kappa*rnlp.δ1[ii], rnlp.δ1[ii]-max_decrease)
                MadNLP.variable(ipm.xl)[cc1] = get_lvar(mpcc)[cc1_orig] - rnlp.δ1[ii]
                updated = true
            end
            if x2 <= 0 # we are lower bound infeasible:
                max_decrease =
                    (
                        relax.k_ftb
                    )*(MadNLP.variable(ipm.x)[cc2] - MadNLP.variable(ipm.xl)[cc2])
                rnlp.δ2[ii] = max(relax.kappa*rnlp.δ2[ii], rnlp.δ2[ii]-max_decrease)
                MadNLP.variable(ipm.xl)[cc2] = get_lvar(mpcc)[cc2_orig] - rnlp.δ2[ii]
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

function get_delta_candidate(nu, x, sigma, delta_max)
    if nu + x < 0.0
        return delta_max
    else
        return min(delta_max, sigma/(nu+x))
    end
end

function update_sigma!(
    relax::RelaxLBUpdate{T},
    rnlp::ScholtesRelaxation{T},
    solver::MadNLPCSolver{T},
) where {T}
    ipm = solver.ipm
    cb = ipm.cb
    mpcc = solver.mpcc
    ncc = get_ncc(mpcc)
    ncon = get_ncon(mpcc)
    ind_cc1 = solver.ind_cc1
    ind_cc2 = solver.ind_cc2
    # Store previous delta
    # TODO(@anton) this is inefficient
    solver.prev_delta1 .= rnlp.δ1
    solver.prev_delta2 .= rnlp.δ2
    mu = ipm.mu
    if solver.delta_rollback
        return # Already updated everything, do nothing
    end
    # update c
    ipm.c[(end-ncc+1):end] .+= get_relaxation(rnlp)

    # TODO(@anton) is this the best trigger?
    kkt_error = max(ipm.inf_pr, ipm.inf_du, ipm.inf_compl, solver.inf_pr_cc)
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
    if kkt_error <= relax.relax_threshold
        for ii in 1:ncc
            cc1 = ind_cc1[ii]
            cc2 = ind_cc2[ii]
            cc1_orig = get_ind_cc1(mpcc)[ii]
            cc2_orig = get_ind_cc2(mpcc)[ii]
            nu1 = solver.multipliers_cc1[ii]
            nu2 = solver.multipliers_cc2[ii]

            nu1_filt = solver.multipliers_cc1_filt[ii]
            nu2_filt = solver.multipliers_cc2_filt[ii]

            x1 = MadNLP.variable(ipm.x)[cc1] - MadNLP.variable(ipm.xl)[cc1]
            z1 = MadNLP.variable(ipm.zl)[cc1]
            x2 = MadNLP.variable(ipm.x)[cc2] - MadNLP.variable(ipm.xl)[cc2]
            z2 = MadNLP.variable(ipm.zl)[cc2]
            zs = MadNLP.slack(ipm.zu)[end-ncc+ii]
            s = MadNLP.slack(ipm.x)[end-ncc+ii]

            nu1_inactive =
                relax.use_filtered ? nu1_filt <= -((ipm.mu)^relax.tau) :
                nu1 <= -((ipm.mu)^relax.tau)
            if nu1_inactive && rnlp.δ1[ii] < relax.mu_factor*ipm.mu
                delta_candidate = get_delta_candidate(nu1, x2, rnlp.σ[ii], relax.delta_max)
                println(
                    "Relaxing cc1[$(ii)] with nu1=$(nu1) and bound = $(delta_candidate)",
                )
                # Relax the lower bound, and take a magic step in the multipliers
                rnlp.δ1[ii] = delta_candidate
                MadNLP.variable(ipm.xl)[cc1] = mpcc.meta.lvar[cc1_orig] - rnlp.δ1[ii]

                # Calculate new values
                mu_r = mu + (x1*z1 - mu)
                nu_res = -MadNLP.primal(ipm.f)[cc1] - MadNLP.primal(ipm.zu)[cc1]
                z1_hat = inv(x1+rnlp.δ1[ii])*mu_r # TODO(@anton): maybe do mu+r where r is the old residual
                zs_hat = inv(x2)*(-nu1 + z1_hat) # TODO(@anton): if this doesn't work then calculate ther real residual instead of -nu1.
                z2_hat = x1*zs_hat + nu2 # TODO(@anton) same here
                delta_zs = zs_hat - zs
                println(-s*(zs) - mu)
                println(-s*(zs_hat) - mu)

                # Set new values
                ## Set the new slack?
                #MadNLP.slack(ipm.x)[end-ncc+ii] = -mu*inv(zs_hat)
                ## Set the new duals
                MadNLP.variable(ipm.zl)[cc1] = z1_hat
                MadNLP.variable(ipm.zl)[cc2] = z2_hat
                MadNLP.slack(ipm.zu)[end-ncc+ii] = zs_hat
                ipm.y[end-ncc+ii] = zs_hat

                ## Set the new J'y_c
                ipm.jacl[cc1] += x2*delta_zs*cb.con_scale[end-ncc+ii]
                ipm.jacl[cc2] += x1*delta_zs*cb.con_scale[end-ncc+ii]
                ipm.jacl[end-ncc+ii] -= delta_zs*cb.con_scale[end-ncc+ii]

                ## Set the multiplier contribution in the Hessian of the Lagrangian
                nnzh = get_nnzh(mpcc)
                ipm.kkt.hess[nnzh+ii] = zs_hat*cb.con_scale[end-ncc+ii]
            elseif relax.unrelax && rnlp.δ1[ii] > 0 && nu1 >= ((ipm.mu)^relax.tau)
                max_decrease =
                    relax.k_ftb*(MadNLP.variable(ipm.x)[cc1] - MadNLP.variable(ipm.xl)[cc1])
                rnlp.δ1[ii] = max(0.0, rnlp.δ1[ii]-max_decrease)
                println(
                    "unrelaxing cc1[$(ii)], max_decrease = $(max_decrease), violation = $(MadNLP.variable(ipm.x)[cc1] - mpcc.meta.lvar[cc1_orig])",
                )
                MadNLP.variable(ipm.xl)[cc1] = mpcc.meta.lvar[cc1_orig] - rnlp.δ1[ii]
                rnlp.σ[ii] = relax.mu_factor*ipm.mu
                # TODO(@anton) this should probably magic step as well but need to figure out how because it is a step in the primal
            end

            nu2_inactive =
                relax.use_filtered ? nu2_filt <= -((ipm.mu)^relax.tau) :
                nu2 <= -((ipm.mu)^relax.tau)
            if nu2_inactive && rnlp.δ2[ii] < relax.mu_factor*ipm.mu
                delta_candidate = get_delta_candidate(nu2, x1, rnlp.σ[ii], relax.delta_max)
                println(
                    "Relaxing cc2[$(ii)] with nu1=$(nu2) and bound = $(delta_candidate)",
                )
                rnlp.δ2[ii] = delta_candidate
                MadNLP.variable(ipm.xl)[cc2] = mpcc.meta.lvar[cc2_orig] - rnlp.δ2[ii]

                # Calculate new values
                mu_r = mu + (x2*z2 - mu)
                nu_res = -MadNLP.primal(ipm.f)[cc2] - MadNLP.primal(ipm.zu)[cc2]
                px = MadNLP.primal(ipm.p)
                z2_hat = inv(x2+rnlp.δ2[ii])*mu_r # TODO(@anton): maybe do mu+r where r is the old residual
                zs_hat = inv(x1)*(-nu2 + z2_hat) # TODO(@anton): if this doesn't work then calculate ther real residual instead of -nu1.
                z1_hat = x2*zs_hat + nu1 # TODO(@anton) same here
                delta_zs = zs_hat - zs
                # Set new values
                ## Set the new slack?
                #MadNLP.slack(ipm.x)[end-ncc+ii] = -mu*inv(zs_hat)
                ## Set the new duals
                MadNLP.variable(ipm.zl)[cc1] = z1_hat
                MadNLP.variable(ipm.zl)[cc2] = z2_hat
                MadNLP.slack(ipm.zu)[end-ncc+ii] = zs_hat
                ipm.y[end-ncc+ii] = zs_hat

                #MadNLP.jtprod!(ipm.jacl, ipm.kkt, ipm.y)
                ## Set the new J'y_c
                ipm.jacl[cc1] += x2*delta_zs*cb.con_scale[end-ncc+ii]
                ipm.jacl[cc2] += x1*delta_zs*cb.con_scale[end-ncc+ii]
                ipm.jacl[end-ncc+ii] -= delta_zs*cb.con_scale[end-ncc+ii]

                ## Set the multiplier contribution in the Hessian of the Lagrangian
                nnzh = get_nnzh(mpcc)
                ipm.kkt.hess[nnzh+ii] = zs_hat*cb.con_scale[end-ncc+ii]
            elseif relax.unrelax && rnlp.δ2[ii] > 0 && nu2 >= ((ipm.mu)^relax.tau)
                max_decrease =
                    relax.k_ftb*(MadNLP.variable(ipm.x)[cc2] - MadNLP.variable(ipm.xl)[cc2])
                rnlp.δ2[ii] = max(0.0, rnlp.δ2[ii]-max_decrease)
                println(
                    "unrelaxing cc2[$(ii)], max_decrease = $(max_decrease), violation = $(MadNLP.variable(ipm.x)[cc2] - mpcc.meta.lvar[cc2_orig])",
                )
                MadNLP.variable(ipm.xl)[cc2] = mpcc.meta.lvar[cc2_orig] - rnlp.δ2[ii]
                rnlp.σ[ii] = relax.mu_factor*ipm.mu
            end
        end
    end

    # update c
    ipm.c[(end-ncc+1):end] .-= get_relaxation(rnlp)
    # Here we assume the barrier update handles whether we throw out the filter.
    return nothing
end

function init_sigma!(
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
    set_relaxation(rnlp, max(sigma_candidate, solver.opts.sigma_min))
    # update c
    ipm.c[(end-ncc+1):end] .-= get_relaxation(rnlp)
    return nothing
end

function init_sigma!(
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

function init_sigma!(
    relax::TwoSidedScholtesUpdate{T},
    rnlp::ScholtesRelaxation{T},
    solver::MadNLPCSolver{T},
) where {T}
    ipm = solver.ipm
    mpcc = solver.mpcc
    ncc = mpcc.meta.ncc
    # update c
    ipm.c[(end-ncc+1):end] .+= get_relaxation(rnlp)
    # calculate new sigma
    sigma_candidate = relax.sigma_mu_ratio*(solver.ipm.mu^relax.sigma_mu_exp)
    set_relaxation(rnlp, max(sigma_candidate, solver.opts.sigma_min))
    # update c
    ipm.c[(end-ncc+1):end] .-= get_relaxation(rnlp)
    return nothing
end

function init_sigma!(
    relax::RelaxLBUpdate{T},
    rnlp::ScholtesRelaxation{T},
    solver::MadNLPCSolver{T},
) where {T}
    ipm = solver.ipm
    mpcc = solver.mpcc
    ncc = mpcc.meta.ncc
    # update c
    ipm.c[(end-ncc+1):end] .+= get_relaxation(rnlp)
    # calculate new sigma
    sigma_candidate = relax.sigma_mu_ratio*(solver.ipm.mu^relax.sigma_mu_exp)
    set_relaxation(rnlp, max(sigma_candidate, solver.opts.sigma_min))
    # update c
    ipm.c[(end-ncc+1):end] .-= get_relaxation(rnlp)
    return nothing
end
