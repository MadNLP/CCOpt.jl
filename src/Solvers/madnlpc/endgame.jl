# Default to do nothing
function do_endgame!(
    solver::MadNLPCSolver,
    rnlp::RNLP,
    endgame::NoEndgameStrategy,
) where {RNLP <: AbstractMPCCRelaxation} end

## Lower bound relaxation endgame
# Do nothing for generic RNLPs
# TODO(@anton) we could do something.
function do_endgame(
    solver::MadNLPCSolver,
    rnlp::RNLP,
    endgame::RelaxLBEndgameStrategy,
) where {RNLP <: AbstractMPCCRelaxation} end

function get_delta_candidate(nu, x, sigma, delta_max)
    if nu + x < 0.0
        return delta_max
    else
        return min(delta_max, sigma/(nu+x))
    end
end

# If using Scholtes do LB relaxation
function do_endgame!(
    solver::MadNLPCSolver{T},
    rnlp::ScholtesRelaxation{T},
    endgame::RelaxLBEndgameStrategy{T},
) where {T}
    ipm = solver.ipm
    cb = ipm.cb
    mpcc = solver.mpcc
    ncc = get_ncc(mpcc)
    ncon = get_ncon(mpcc)
    ind_cc1 = solver.ind_cc1
    ind_cc2 = solver.ind_cc2
    mu = ipm.mu

    updated = false
    nu_bound = ipm.inf_du^endgame.tau
    # Check relaxation for each complementarity.
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

        nu1_inactive = endgame.use_filtered ? nu1_filt <= -(nu_bound) : nu1 <= -(nu_bound)
        if nu1_inactive
            delta_candidate = get_delta_candidate(nu1, x2, rnlp.σ[ii], endgame.delta_max)
            # If not enough change in delta candidate, it isn't worth perturbing newton.
            # In practice this means relaxation happens only once most of the time
            if delta_candidate < endgame.min_delta_inc_factor * rnlp.δ1[ii]
                continue
            end
            # Relax the lower bound, and take a magic step in the multipliers
            rnlp.δ1[ii] = delta_candidate
            updated = true
            MadNLP.variable(ipm.xl)[cc1] = get_lvar(mpcc)[cc1_orig] - rnlp.δ1[ii]

            # Calculate new values
            mu_r = mu + (x1*z1 - mu)
            nu_res = -MadNLP.primal(ipm.f)[cc1] - MadNLP.primal(ipm.zu)[cc1]
            z1_hat = inv(x1+rnlp.δ1[ii])*mu_r # TODO(@anton): maybe do mu+r where r is the old residual
            zs_hat = inv(x2)*(-nu1 + z1_hat) # TODO(@anton): if this doesn't work then calculate ther real residual instead of -nu1.
            z2_hat = x1*zs_hat + nu2 # TODO(@anton) same here
            delta_zs = zs_hat - zs

            # Set new values
            ## TODO(@anton) Set the new slack?
            ## TODO(@anton) also adjust x1?
            #MadNLP.slack(ipm.x)[end-ncc+ii] = -mu*inv(zs_hat)
            ## Set the new duals
            MadNLP.variable(ipm.zl)[cc1] = z1_hat
            MadNLP.variable(ipm.zl)[cc2] = z2_hat
            MadNLP.slack(ipm.zu)[end-ncc+ii] = zs_hat
            ipm.y[end-ncc+ii] = zs_hat

            ## Set the new J'y_c
            # TODO(@anton) is this correct
            ipm.jacl[cc1] += x2*delta_zs*cb.con_scale[end-ncc+ii]
            ipm.jacl[cc2] += x1*delta_zs*cb.con_scale[end-ncc+ii]
            ipm.jacl[end-ncc+ii] -= delta_zs*cb.con_scale[end-ncc+ii]

            ## Set the multiplier contribution in the Hessian of the Lagrangian
            ipm.kkt.hess[end-ncc+ii] = zs_hat*cb.con_scale[end-ncc+ii]
        end

        nu2_inactive = endgame.use_filtered ? nu2_filt <= -(nu_bound) : nu2 <= -(nu_bound)
        if nu2_inactive
            delta_candidate = get_delta_candidate(nu2, x1, rnlp.σ[ii], endgame.delta_max)
            # If not enough change in delta candidate, it isn't worth perturbing newton.
            # In practice this means relaxation happens only once most of the time
            if delta_candidate < endgame.min_delta_inc_factor * rnlp.δ2[ii]
                continue
            end
            rnlp.δ2[ii] = delta_candidate
            updated = true
            MadNLP.variable(ipm.xl)[cc2] = get_lvar(mpcc)[cc2_orig] - rnlp.δ2[ii]

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
            ipm.kkt.hess[end-ncc+ii] = zs_hat*cb.con_scale[end-ncc+ii]
        end
    end

    # Throw out the filter as the barrier problem has changed
    if updated
        empty!(ipm.filter)
        push!(ipm.filter, (ipm.theta_max, -Inf))
    end
    return nothing
end
