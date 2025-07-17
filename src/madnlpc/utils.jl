mutable struct MadNLPCExecutionStats{T, VT} <: AbstractExecutionStats
    options::MadNLP.AbstractOptions
    status::Status
    solution::VT
    counters::MadNLPCCounters
    stats::MadNLP.MadNLPExecutionStats{T, VT}
end

function MadNLPCExecutionStats(solver::MadNLPCSolver)
    return MadNLPCExecutionStats(
        solver.opts,
        solver.status,
        solver.x,
        solver.cnt,
        MadNLP.MadNLPExecutionStats(solver.bnlp_ipm),
    )
end

# TODO(@anton) this is a hack, fix it
function SolverCore.getStatus(result::MadNLPCExecutionStats)
    return get_status_output(result.status, result.options, result.options.bnlp_opts)
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

function linearize_lpec!(solver::MadNLPCSolver, tr::Float64)
    return solver.cnt.lpcc_init_time +=
        @elapsed MadMPEC.linearize!(solver.lpcc, solver.x; tr=tr, presolve_binaries=true)
end

function update_lpec_tr!(solver::MadNLPCSolver, tr::Float64)
    return solver.cnt.lpcc_init_time +=
        @elapsed MadMPEC.tr!(solver.lpcc, solver.x; tr=tr, presolve_binaries=true)
end

function solve_lpec!(solver::MadNLPCSolver{T, VT}; x0=nothing) where {T, VT}
    solver.cnt.lpcc_solves += 1
    solver.cnt.lpcc_init_time += @elapsed begin
        model = MadMPEC.build(solver.lpcc; x0=x0)
    end

    solver.cnt.lpcc_solve_time += @elapsed begin
        optimize!(model)

        optimal = is_solved_and_feasible(model)
        if optimal
            vals = value.(model[:x])
            y = vals[solver.lpcc.integrality .== one(Int32)] .> 0.5
            obj = objective_value(model)
        else
            ncc = solver.mpcc.meta.ncc
            vals = VT(undef, length(model[:x]))
            y = BitVector(undef, ncc)
            obj = typemax(T)
        end
    end
    return optimal, vals, y, obj
end

function build_bnlp_solver!(solver::MadNLPCSolver, b::BitVector)
    return solver.cnt.bnlp_init_time += @elapsed begin
        mpcc = solver.mpcc
        # Create BNLP
        bnlp = BranchNLP(mpcc, convert(Vector{Bool}, b))
        @views begin
            bnlp.meta.x0 .= MadNLP.variable(solver.ipm.x) # Warmstart the BranchNLP
            bnlp.meta.x0[mpcc.meta.ind_cc1[.!b]] .=
                mpcc.meta.lvar[mpcc.meta.ind_cc1[.!b]]
            bnlp.meta.x0[mpcc.meta.ind_cc2[b]] .= mpcc.meta.lvar[mpcc.meta.ind_cc2[b]]
        end
        # Solve the BNLP
        solver.bnlp_ipm = MadNLP.MadNLPSolver(bnlp; solver.opts.bnlp_opts...) # TODO(@anton) again options for BNLP should live somewhere
        #### WARNING: THIS IS A HACK
        # Because there is no way to pass a counters object we have to make sure
        # that all the pointers get updated
        # TODO(@anton) this needs to be done in a smarter way
        solver.bnlp_ipm.cnt = solver.ipm.cnt
        # HACK ENDS HERE
    end
end

function solve_bnlp!(solver::MadNLPCSolver, stats)
    solver.cnt.bnlp_solves += 1
    solver.cnt.bnlp_solve_time += @elapsed begin
        stats = MadNLP.solve!(solver.bnlp_ipm.nlp, solver.bnlp_ipm, stats)
    end
    return stats
end

function solve_bnlp!(solver::MadNLPCSolver)
    solver.cnt.bnlp_solves += 1
    solver.cnt.bnlp_solve_time += @elapsed begin
        stats = MadNLP.solve!(solver.bnlp_ipm)
    end
    return stats
end
