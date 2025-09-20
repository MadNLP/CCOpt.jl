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

@inline function update_c!(c, σ, σ_old, ncc, scale)
    return c[(end-ncc+1):end] .+= scale .* (σ_old - σ)
end

function linearize!(solver::MadNLPCSolver{T}, tr::T) where {T}
    # TODO(@anton) presolving at the LPCC level makes this questionable
    MadMPEC.linearize!(solver.lpcc, solver.mpcc, solver.x; tr=tr)
    return solver.lpcc_solver = build_lpcc_solver(solver.lpcc, solver.opts.lpcc_solver_opts)
end

function tr!(solver::MadNLPCSolver{T}, tr::T) where {T}
    MadMPEC.tr!(solver.lpcc, solver.mpcc, solver.x; tr=tr)
    return solver.lpcc_solver = build_lpcc_solver(solver.lpcc, solver.opts.lpcc_solver_opts)
end

function lpcc_oracle!(lpcc_solver::MadNLPCSolver{T, VT}) where {T, VT}
    lpcc = lpcc_solver.mpcc
    stat = solve_homotopy!(lpcc_solver)
    ncc = length(lpcc.fixed_map)

    optimal = stat.status == NLP_STATIONARY
    if optimal
        vals = stat.solution
        y = BitVector(undef, ncc)
        y[lpcc.fixed_map .== 0] .= comp_res_left(lpcc, vals) .> comp_res_right(lpcc, vals)
        y[lpcc.fixed_map .== 1] .= false
        y[lpcc.fixed_map .== 2] .= true
        obj = stat.stats.objective
    else
        vals = VT(undef, lpcc.meta.nvar)
        y = BitVector(undef, ncc)
        obj = typemax(T)
    end
    println("ncc=$(ncc) fm=$(lpcc.fixed_map), optimal=$(optimal) y=$(y)")
    return optimal, vals, y, obj
end

function lpcc_oracle!(lpcc_solver::MilpSolver{T, VT}) where {T, VT}
    model = lpcc_solver.model
    lpcc = lpcc_solver.lpcc
    ncc = length(lpcc.fixed_map)
    optimize!(model)

    optimal = is_solved_and_feasible(model)
    if optimal
        vals = value.(model[:x])
        y = BitVector(undef, ncc)
        y[lpcc_solver.lpcc.fixed_map .== 0] .= @views(vals[(lpcc.meta.nvar+1):end] .> 0.5)
        y[lpcc_solver.lpcc.fixed_map .== 1] .= false
        y[lpcc_solver.lpcc.fixed_map .== 2] .= true
        obj = objective_value(model)
    else
        vals = VT(undef, length(model[:x]))
        y = BitVector(undef, ncc)
        obj = typemax(T)
    end

    return optimal, vals, y, obj
end

function eval_relaxed_cons_wrapper!(solver::MadNLPCSolver{T}, c::AbstractVector{T}, x::MadNLP.PrimalVector{T}) where {T}
    #TODO(@anton) correct counters here?
    ipm = solver.ipm
    rnlp = solver.rnlp
    cnt = ipm.cnt
    mpcc_ncon = rnlp.mpcc.meta.ncon
    ncc = rnlp.mpcc.meta.ncc
    MadNLP.@trace(solver.logger, "Evaluating relaxed complementarities.")
    @views begin
        cnt.eval_function_time += @elapsed relaxed_cons!(
        rnlp,
        MadNLP.variable(x),
        c[(mpcc_ncon+1):(rnlp.meta.ncon)])
        c[(mpcc_ncon+1):(rnlp.meta.ncon)] .-= MadNLP.slack(x)[end-ncc+1:end]
        c[(mpcc_ncon+1):(rnlp.meta.ncon)] .-= ipm.rhs[end-ncc+1:end]
    end
end
