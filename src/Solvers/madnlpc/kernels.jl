function MadNLP.get_average_complementarity(solver::MadNLPCSolver{T}) where {T}
    ipm = solver.ipm
    mpcc = solver.mpcc
    ncc = ipm.nlb + ipm.nub + get_ncc(mpcc)
    cc_lb = dot(ipm.x_lr, ipm.zl_r) - dot(ipm.xl_r, ipm.zl_r)
    cc_ub = dot(ipm.xu_r, ipm.zu_r) - dot(ipm.x_ur, ipm.zu_r)
    # We know mpcc is in vertical form, therefore:
    # TODO(@anton) add view for x1, x2
    cc_pr = @views dot(
        MadNLP.variable(ipm.x)[solver.ind_cc1] - MadNLP.variable(ipm.xl)[solver.ind_cc1],
        MadNLP.variable(ipm.x)[solver.ind_cc2] - MadNLP.variable(ipm.xl)[solver.ind_cc2],
    )
    return (cc_lb + cc_ub + cc_pr)/ncc
end

function MadNLP.get_min_complementarity(solver::MadNLPCSolver{T}) where {T}
    ipm = solver.ipm
    mpcc = solver.mpcc
    cc_lb = mapreduce(
        (x_l, xl, zl) -> (x_l-xl)*zl,
        min,
        ipm.x_lr,
        ipm.xl_r,
        ipm.zl_r,
        init=T(Inf),
    )
    cc_ub = mapreduce(
        (x_u, xu, zu) -> (xu-x_u)*zu,
        min,
        ipm.x_ur,
        ipm.xu_r,
        ipm.zu_r,
        init=T(Inf),
    )
    # We know mpcc is in vertical form, therefore:
    cc_pr = @views mapreduce(
        (x1, xl1, x2, x2l) -> (x1-xl1)*(x2-x2l),
        min,
        MadNLP.variable(ipm.x)[solver.ind_cc1],
        MadNLP.variable(ipm.xl)[solver.ind_cc1],
        MadNLP.variable(ipm.x)[solver.ind_cc2],
        MadNLP.variable(ipm.xl)[solver.ind_cc2],
        init=T(Inf),
    )
    return min(cc_lb, cc_ub, cc_pr)
end

@inline function update_c!(c, σ, σ_old, ncc)
    return c[(end-ncc+1):end] .+= σ_old - σ
end

function estimate_mpec_multipliers(solver::MadNLPCSolver{T}) where {T}
    ipm = solver.ipm
    mpcc = solver.mpcc
    ncc = get_ncc(mpcc)
    ncon = get_ncon(mpcc)
    ind_cc1 = solver.ind_cc1
    ind_cc2 = solver.ind_cc2
    N = solver.opts.mpec_multiplier_filter_history

    for ii in 1:ncc
        cc1 = ind_cc1[ii]
        cc2 = ind_cc2[ii]
        x1 = MadNLP.variable(ipm.x)[cc1] - mpcc.meta.lvar[cc1]
        z1 = MadNLP.variable(ipm.zl)[cc1]
        x2 = MadNLP.variable(ipm.x)[cc2] - mpcc.meta.lvar[cc2]
        z2 = MadNLP.variable(ipm.zl)[cc2]
        zs = MadNLP.slack(ipm.zu)[end-ncc+ii]

        f1 = MadNLP.variable(ipm.f)[cc1]
        f2 = MadNLP.variable(ipm.f)[cc2]
        jac1 = ipm.jacl[cc1]
        jac2 = ipm.jacl[cc2]

        # println("multipliers1: $(z1) - $(zs)*$(x2) = $(z1 - zs*x2)")
        # println("grad1: $(f1) + $(jac1 - zs*x2) = $(f1 + jac1 - zs*x2)")
        # println("multipliers2: $(z2) - $(zs)*$(x1) = $(z2 - zs*x1)")
        # println("grad2: $(f2) + $(jac2 - zs*x1) = $(f2 + jac2 - zs*x1)")
        #solver.multipliers_cc1[ii] = f1 + jac1 - zs*x2
        #solver.multipliers_cc2[ii] = f2 + jac2 - zs*x1

        solver.multipliers_cc1[ii] = z1 - zs*x2
        solver.multipliers_cc2[ii] = z2 - zs*x1
    end

    if solver.ipm.cnt.k==0
        solver.multipliers_cc1_filt .= solver.multipliers_cc1
        solver.multipliers_cc2_filt .= solver.multipliers_cc2
    else
        # Do EWMA
        solver.multipliers_cc1_filt .-= solver.multipliers_cc1_filt ./ N
        solver.multipliers_cc2_filt .-= solver.multipliers_cc2_filt ./ N
        solver.multipliers_cc1_filt .+= solver.multipliers_cc1 ./ N
        solver.multipliers_cc2_filt .+= solver.multipliers_cc2 ./ N
    end
end

is_relaxation_acceptable(solver, rnlp, relaxation) = true

function is_relaxation_acceptable(solver, rnlp::ScholtesRelaxation, relax::RelaxLBUpdate)
    # TODO(@anton) this is inefficient but alas
    if !relax.reject_steps
        return true
    end
    ind_cc1 = rnlp.mpcc.meta.ind_cc1
    ind_cc2 = rnlp.mpcc.meta.ind_cc2
    ncc = rnlp.mpcc.meta.ncc
    acceptable = true
    for ii in 1:ncc
        cc1=ind_cc1[ii]
        cc2=ind_cc2[ii]
        x1 = MadNLP.variable(solver.ipm.x_trial)[cc1]
        lbx1 = rnlp.mpcc.meta.lvar[cc1]
        x2 = MadNLP.variable(solver.ipm.x_trial)[cc2]
        lbx2 = rnlp.mpcc.meta.lvar[cc2]
        if x1 < lbx1 - solver.ipm.opt.tol
            rnlp.δ1[ii] = solver.prev_delta1[ii]
            MadNLP.variable(solver.ipm.xl)[cc1] = lbx1 - rnlp.δ1[ii]
            println("δ1[$(ii)] is unacceptable")
            acceptable = false
        end
        if x2 < lbx2 - solver.ipm.opt.tol
            rnlp.δ2[ii] = solver.prev_delta2[ii]
            println("δ2[$(ii)] is unacceptable")
            MadNLP.variable(solver.ipm.xl)[cc1] = lbx2 - rnlp.δ2[ii]
            acceptable = false
        end
    end
    return acceptable
end
