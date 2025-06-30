function log_iter(
    logger::IterateLogger,
    solver::MadNLPCSolver{T, VT};
    magic=false,
) where {T, VT}
    if isnothing(logger.file)
        return nothing
    end
    ipm = solver.ipm
    ncc = solver.mpcc.meta.ncc

    k = ipm.cnt.k
    x0 = MadNLP.variable(ipm.x)[solver.mpcc.meta.ind_x]
    x1 = MadNLP.variable(ipm.x)[solver.mpcc.meta.ind_cc1]
    x2 = MadNLP.variable(ipm.x)[solver.mpcc.meta.ind_cc2]
    s = MadNLP.slack(ipm.x)[(end-ncc+1):end]

    z1 = MadNLP.variable(ipm.zl)[solver.mpcc.meta.ind_cc1]
    z2 = MadNLP.variable(ipm.zl)[solver.mpcc.meta.ind_cc2]
    zs = MadNLP.slack(ipm.zu)[(end-ncc+1):end]

    alpha_pr = ipm.alpha
    alpha_du = ipm.alpha_z

    ls = ipm.cnt.l

    obj = ipm.obj_val
    inf_pr = ipm.inf_pr
    inf_du = ipm.inf_du

    theta = MadNLP.get_theta(ipm.c)
    varphi = MadNLP.get_varphi(ipm.obj_val, ipm.x_lr, ipm.xl_r, ipm.xu_r, ipm.x_ur, ipm.mu)

    mu = ipm.mu
    sigma = solver.scholtes.mu

    W = ipm.kkt.aug_com
    K = Array(Symmetric(W, :L))
    KKT_s = eigvals(K)

    iter = MadNLPCIterate(
        k,
        x0,
        x1,
        x2,
        s,
        z1,
        z2,
        zs,
        alpha_pr,
        alpha_du,
        ls,
        obj,
        inf_pr,
        inf_du,
        theta,
        varphi,
        mu,
        sigma,
        KKT_s,
        magic,
    )
    serialize(logger.file, iter)

    return nothing
end

function finalize(logger::IterateLogger)
    if isnothing(logger.file)
        return nothing
    end

    return close(logger.file)
end
