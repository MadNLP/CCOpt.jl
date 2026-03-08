function solve_lpcc(
    lpcc::LPCCModel{T,VT},
    solver_opts::MadNLPCOptions;
    bound_relax_factor=0.0,
    kwargs...
) where {T,VT}
    solver = MadMPEC.MadNLPCSolver(lpcc; solver_opts=solver_opts, bound_relax_factor=0.0, kwargs...)
    stats = MadMPEC.solve_homotopy!(solver)

    y .= comp_res_left(lpcc, vals) .> comp_res_right(lpcc, vals)
    
    return stats, y
end

function crossover(
    mpcc::AbstractMPCCModel{T, VT},
    x0::VT,
    lpcc_solver_opts
    ;
    lpcc_solver_kwargs=NamedTuple(),
    proj_tr=1.0
) where {T, VT}
    # Do projection
    proj_lpcc = LPCCModel(mpcc, x0; tr=proj_tr)
    
    proj_sol, y0 = solve_lpcc(proj_lpcc, lpcc_solver_opts;lpcc_solver_kwargs...)

    return proj_sol, y0
end
