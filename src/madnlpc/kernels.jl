function MadNLP.get_average_complementarity(solver::MadNLPCSolver{T}) where {T}
    ipm = solver.ipm
    mpcc = solver.mpcc
    ncc = ipm.nlb + ipm.nub + mpcc.meta.ncc
    cc_lb = dot(ipm.x_lr, ipm.zl_r) - dot(ipm.xl_r, ipm.zl_r)
    cc_ub = dot(ipm.xu_r, ipm.zu_r) - dot(ipm.x_ur, ipm.zu_r)
    # We know mpcc is in vertical form, therefore:
    # TODO(@anton) add view for x1, x2
    cc_pr = @views dot(
        MadNLP.variable(ipm.x)[mpcc.meta.ind_cc1] -
        MadNLP.variable(ipm.xl)[mpcc.meta.ind_cc1],
        MadNLP.variable(ipm.x)[mpcc.meta.ind_cc2] -
        MadNLP.variable(ipm.xl)[mpcc.meta.ind_cc2],
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
        MadNLP.variable(ipm.x)[mpcc.meta.ind_cc1],
        MadNLP.variable(ipm.xl)[mpcc.meta.ind_cc1],
        MadNLP.variable(ipm.x)[mpcc.meta.ind_cc2],
        MadNLP.variable(ipm.xl)[mpcc.meta.ind_cc2],
        init=T(Inf),
    )
    return min(cc_lb, cc_ub, cc_pr)
end

@inline function update_c!(c, σ, σ_old, ncc)
    return c[(end-ncc+1):end] .+= σ_old - σ
end
