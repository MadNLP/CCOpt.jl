mutable struct MadNLPCExecutionStats{T, VT} <: AbstractExecutionStats
    options::MadNLP.AbstractOptions
    mpcc_options::MadNLP.AbstractOptions
    status::MadNLP.Status
    objective::T
    solution::VT
    constraints::VT
    multipliers::VT
    multipliers_L::VT
    multipliers_U::VT
    multipliers_x1::VT
    multipliers_x2::VT
    dual_feas::T
    primal_feas::T
    inf_pr_cc::T
    iter::Int
    counters::MadNLPCCounters
end

function MadNLPCExecutionStats(solver::MadNLPCSolver)
    n, m = get_nvar(solver.rnlp), get_ncon(solver.rnlp)
    ncc = solver.mpcc.meta.ncc
    VT = typeof(get_x0(solver.rnlp))
    x = similar(VT, n)
    zl = similar(VT, n)
    zu = similar(VT, n)
    zx1 = similar(VT, ncc)
    zx2 = similar(VT, ncc)
    c = similar(VT, m)
    y = similar(VT, m)
    n = MadNLP.get_nvar(solver.ipm.nlp)
    m = solver.mpcc.meta.ncon
    ind_cc1 = solver.ind_cc1
    ind_cc2 = solver.ind_cc2

    return MadNLPCExecutionStats(
        solver.ipm.opt,
        solver.opts,
        solver.ipm.status,
        MadNLP.unpack_obj(solver.ipm.cb, solver.ipm.obj_val),
        x,
        c,
        y,
        zl,
        zu,
        zx1,
        zx2,
        solver.ipm.inf_du,
        solver.ipm.inf_pr,
        solver.inf_pr_cc,
        0,
        solver.cnt,
    )
end

# TODO(@anton) this is a hack, fix it
function SolverCore.getStatus(result::MadNLPCExecutionStats)
    return MadNLP.get_status_output(result.status, result.options)
end

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
    x1 = MadNLP.variable(ipm.x)[solver.ind_cc1]
    x2 = MadNLP.variable(ipm.x)[solver.ind_cc2]
    s = MadNLP.slack(ipm.x)[(end-ncc+1):end]

    z1 = MadNLP.variable(ipm.zl)[solver.ind_cc1]
    z2 = MadNLP.variable(ipm.zl)[solver.ind_cc2]
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
    sigma = solver.rnlp.mu

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

function get_inf_pr_cc(solver::MadNLPCSolver{T}) where {T}
    return @views(
        mapreduce(
            (a, la, b, lb) -> max((a-la)*(b-lb), la-a, lb-b),
            max,
            MadNLP.variable(solver.ipm.x)[solver.ind_cc1],
            solver.mpcc.meta.lvar[solver.mpcc.meta.ind_cc1],
            MadNLP.variable(solver.ipm.x)[solver.ind_cc2],
            solver.mpcc.meta.lvar[solver.mpcc.meta.ind_cc2];
            init=zero(T),
        )
    )
end

function MadNLP.print_iter(solver::MadNLPCSolver; is_resto=false)
    ipm = solver.ipm
    obj_scale = ipm.cb.obj_scale[]
    mod(ipm.cnt.k, 10)==0 && MadNLP.@info(
        ipm.logger,
        @sprintf(
            "iter    objective    inf_pr   inf_du inf_compl lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls lg(σ)   inf_cc"
        )
    )
    if is_resto
        RR = ipm.RR::RobustRestorer
        inf_du = RR.inf_du_R
        inf_pr = RR.inf_pr_R
        inf_compl = RR.inf_compl_R
        mu = log10(RR.mu_R)
    else
        inf_du = ipm.inf_du
        inf_pr = ipm.inf_pr
        inf_compl = ipm.inf_compl
        mu = log10(ipm.mu)
    end
    MadNLP.@info(
        ipm.logger,
        @sprintf(
            "%4i%s% 10.7e %6.2e %6.2e %7.2e %5.1f %6.2e %s %6.2e %6.2e%s  %i %5.1f  %6.2e",
            ipm.cnt.k,
            is_resto ? "r" : " ",
            ipm.obj_val/obj_scale,
            inf_pr,
            inf_du,
            inf_compl,
            mu,
            ipm.cnt.k == 0 ? 0.0 : norm(MadNLP.primal(ipm.d), Inf),
            ipm.del_w == 0 ? "   - " : @sprintf("%5.1f", log(10, ipm.del_w)),
            ipm.alpha_z,
            ipm.alpha,
            ipm.ftype,
            ipm.cnt.l,
            get_log_relaxation(solver.rnlp),
            solver.inf_pr_cc
        )
    )
    return
end
