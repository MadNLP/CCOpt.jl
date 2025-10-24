function update_sigma!(
    relax::ProportionalRelaxationUpdate{T},
    rnlp::AbstractMPCCRelaxation{T},
    solver::MadNLPCSolver{T},
) where {T}
    ipm = solver.ipm
    mpcc = solver.mpcc
    ncc = mpcc.meta.ncc
    # update c
    sigma_old = rnlp.σ[]
    ipm.c[(end-ncc+1):end] .+= sigma_old
    # calculate new sigma
    sigma_candidate = relax.sigma_mu_ratio*(solver.ipm.mu^relax.sigma_mu_exp)
    if relax.monotone
        solver.rnlp.σ[] = max(min(solver.rnlp.σ[], sigma_candidate), solver.opts.sigma_min)
    else
        solver.rnlp.σ[] = max(sigma_candidate, solver.opts.sigma_min)
    end
    # update c
    ipm.c[(end-ncc+1):end] .-= rnlp.σ[]
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
    sigma_old = rnlp.σ[]
    ipm.c[(end-ncc+1):end] .+= sigma_old
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
    solver.rnlp.σ[] =
        max(gamma_sigma*mean_cc, solver.opts.sigma_min, relax.mu_factor*ipm.mu)
    # update c
    c[(end-ncc+1):end] .-= rnlp.σ[]
    # Throw out the filter as the barrier problem has changed
    empty!(ipm.filter)
    push!(ipm.filter, (ipm.theta_max, -Inf))
    return nothing
end

function kkt_residual_norm(
    rnlp::ScholtesMultiRelaxation{T},
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
    py[(ncon+1):(ncon+ncc)] .-= σ
    pzl .= min.((ipm.x_lr .- ipm.xl_r), ipm.zl_r) # TODO(@anton) probably inefficient replace with map
    pzu .= min.((ipm.xu_r .- ipm.x_ur), ipm.zu_r)
    println("px")
    println.(px)
    println("py")
    println.(py)
    println("pzl")
    println.(pzl)
    println("pzu")
    println.(pzu)
    #println(MadNLP.full(ipm.p))
    return norm(ipm.p, 2)
end

function update_sigma!(
    relax::TwoSidedScholtesUpdate{T},
    rnlp::ScholtesMultiRelaxation{T},
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

    for ii in 1:ncc
        cc1 = ind_cc1[ii]
        cc2 = ind_cc2[ii]
        #x1 = MadNLP.variable(ipm.x)[cc1] - MadNLP.variable(ipm.xl)[cc1]
        x1 = MadNLP.variable(ipm.x)[cc1] - mpcc.meta.lvar[cc1]
        z1 = MadNLP.variable(ipm.zl)[cc1]
        #x2 = MadNLP.variable(ipm.x)[cc2] - MadNLP.variable(ipm.xl)[cc2]
        x2 = MadNLP.variable(ipm.x)[cc2] - mpcc.meta.lvar[cc2]
        z2 = MadNLP.variable(ipm.zl)[cc2]
        zs = MadNLP.slack(ipm.zu)[end-ncc+ii]

        nu1 = z1 - zs*x2
        nu2 = z2 - zs*x1
        println("x1=$(MadNLP.variable(ipm.x)[cc1]), x2=$(MadNLP.variable(ipm.x)[cc2])")
        println("s1=$(x1), s1=$(x2)")
        println("z1=$(z1), z2=$(z2), zs=$(zs)")
        println("nu1=$(nu1), nu2=$(nu2)")
        println("r=$(r), rl=$(rl), ru=$(ru)")
        println("opt = $((rnlp.σopt[ii],rnlp.δ1opt[ii], rnlp.δ2opt[ii]))")
        if nu1 > ru
            rnlp.δ1[ii] = min(relax.kappa*rnlp.δ1[ii], rl)
            MadNLP.variable(ipm.xl)[cc1] = mpcc.meta.lvar[cc1] - rnlp.δ1[ii]
            rnlp.δ1opt[ii] = 0.0
        else
            rnlp.δ1opt[ii] = rnlp.δ1[ii]
        end

        if nu2 > ru
            rnlp.δ2[ii] = min(relax.kappa*rnlp.δ2[ii], rl)
            MadNLP.variable(ipm.xl)[cc2] = mpcc.meta.lvar[cc2] - rnlp.δ2[ii]
            rnlp.δ2opt[ii] = 0.0
        else
            rnlp.δ2opt[ii] = rnlp.δ2[ii]
        end

        if nu1 < ru || nu1 < ru
            rnlp.σ[ii] = min(relax.kappa*rnlp.σ[ii], rl)
            rnlp.σopt[ii] = 0.0
        else
            rnlp.σopt[ii] = rnlp.σ[ii]
        end
    end
end
