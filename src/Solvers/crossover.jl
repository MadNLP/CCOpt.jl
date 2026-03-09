# options for LPCC solving with HiGHS
@kwdef struct HiGHSOptions{T}
    M::T = 1000.0
end

# options for LPCC solving with Gurobi
@kwdef struct GurobiOptions{T}
    M::T = 1000.0
end

function solve_lpcc(
    lpcc::LPCC,
    solver_opts::MadNLPCOptions;
    bound_relax_factor=0.0,
    kwargs...,
) where {LPCC <: LPCCModel}
    solver = MadMPEC.MadNLPCSolver(
        lpcc;
        solver_opts=solver_opts,
        bound_relax_factor=0.0,
        kwargs...,
    )
    stats = MadMPEC.solve_homotopy!(solver)

    y = Vector{Bool}(undef, get_ncc(lpcc))
    y .= comp_res_left(lpcc, stats.solution) .> comp_res_right(lpcc, stats.solution)
    return stats.status == MadNLP.SOLVE_SUCCEEDED, stats, y
end

function crossover(
    mpcc::AbstractMPCCModel{T, VT},
    x0::VT,
    lpcc_solver_opts;
    lpcc_solver_kwargs=NamedTuple(),
    bnlp_solver_kwargs=NamedTuple(),
    proj_tr=1.0,
    tr0=1e-4,
    tr_min=1e-6,
    stationarity_tol=1e-8,
    max_iter=10,
) where {T, VT}
    # Do projection
    proj_lpcc = LPCCModel(mpcc, x0; tr=proj_tr)

    success, proj_sol, y0 = solve_lpcc(proj_lpcc, lpcc_solver_opts; lpcc_solver_kwargs...)

    if !success
        return false, x0
    end

    bnlp = BranchNLP(mpcc, y0)
    get_x0(bnlp) .= x0

    bnlp_sol = MadNLP.madnlp(bnlp; bnlp_solver_kwargs...)

    if bnlp_sol.status != MadNLP.SOLVE_SUCCEEDED
        return false, x0
    end

    x_curr = bnlp_sol.solution
    y_curr = y0
    y_next = similar(y0)
    f_curr = bnlp_sol.objective
    tr = tr0
    ii = 1
    while ii <= max_iter
        while true
            lpcc = LPCCModel(mpcc, x_curr; tr=tr)
            success, lpcc_sol, y_next =
                solve_lpcc(lpcc, lpcc_solver_opts; lpcc_solver_kwargs...)
            if norm(lpcc_sol.solution) <= stationarity_tol ||
               f_curr - lpcc_sol.objective <= stationarity_tol
                return :b_stat, x_curr, y_curr
            end
            if any(y_next .!= y_curr)
                y_curr = y_next
                tr = tr0
                break
            else
                tr = 0.1*tr
                if tr <= tr_min
                    return :tr_min, x_curr, y_curr
                end
            end
        end

        bnlp = BranchNLP(mpcc, y_next)
        get_x0(bnlp) .= x_curr

        bnlp_sol = MadNLP.madnlp(bnlp; bound_push=1e-6, bnlp_solver_kwargs...)
        # if sol fails do something else

        x_curr = bnlp_sol.solution
        f_curr = bnlp_sol.objective

        ii += 1
    end
    return :max_iter, x_curr, y_curr
end
