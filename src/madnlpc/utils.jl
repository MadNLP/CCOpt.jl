function log_iter(
    logger::IterateLogger,
    solver::MadNLPCSolver{T, VT};
    magic=false,
) where {T, VT}
    if isnothing(logger.file)
        return nothing
    end
    ncc = solver.mpcc.meta.ncc

    k = solver.ipm.cnt.k
    x0 = MadNLP.variable(solver.ipm.x)[solver.mpcc.meta.ind_x]
    x1 = MadNLP.variable(solver.ipm.x)[solver.mpcc.meta.ind_cc1]
    x2 = MadNLP.variable(solver.ipm.x)[solver.mpcc.meta.ind_cc2]
    s = MadNLP.slack(solver.ipm.x)[(end-ncc+1):end]

    z1 = MadNLP.variable(solver.ipm.zl)[solver.mpcc.meta.ind_cc1]
    z2 = MadNLP.variable(solver.ipm.zl)[solver.mpcc.meta.ind_cc2]
    zs = MadNLP.slack(solver.ipm.zu)[(end-ncc+1):end]

    alpha_pr = solver.ipm.alpha
    alpha_du = solver.ipm.alpha_z

    iter = MadNLPCIterate(k, x0, x1, x2, s, z1, z2, zs, alpha_pr, alpha_du, magic)
    serialize(logger.file, iter)

    return nothing
end

function finalize(logger::IterateLogger)
    if isnothing(logger.file)
        return nothing
    end

    return close(logger.file)
end
