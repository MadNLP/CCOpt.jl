using CSV
using DataFrames
using Plots, BenchmarkProfiles
using AmplNLReader, MadMPEC
using NLPModelsIpopt
using MadNLP, MadNLPHSL
using MadNCL

function mpcc_from_ampl(ampl::AmplNLReader.AmplModel)
    # First we find the nonzero elements in the cvar vector:
    # cvar = 0 → normal nonlinear constraint
    # cvar ≠ 0 → complementarity with the cvar-th variable
    ind_ccc2 = findall(!iszero, ampl.meta.cvar)
    # Then we store the corresponding variables
    ind_vcc1 = ampl.meta.cvar[ind_ccc2]
    return MadMPEC.MPCCModelVarCon(ampl, ind_vcc1, ind_ccc2)
end

function save_madnlp_df(
    names::Vector{<:AbstractString},
    stats_madnlp::Vector{MadMPEC.HomotopySolverStats{T, VT}},
    probs::Any,
    name::AbstractString,
) where {T, VT}
    df_madnlp = DataFrame(
        name=names,
        success=[s.status == MadMPEC.NLP_STATIONARY for s in stats_madnlp],
        status=[s.status for s in stats_madnlp],
        objective=[s.objective for s in stats_madnlp],
        inf_cc=[s.inf_cc for s in stats_madnlp],
        wall_time=[s.wall_time for s in stats_madnlp],
        iter=[s.iter for s in stats_madnlp],
        eval_function_time=[s.eval_function_time for s in stats_madnlp],
        linear_solver_time=[s.linear_solver_time for s in stats_madnlp],
        homotopy_iter=[length(s.nlp_stats) for s in stats_madnlp],
    )
    CSV.write(name, df_madnlp)

    return df_madnlp
end

function save_madnlp_df(
    names::Vector{<:AbstractString},
    stats_madnlp::Vector{MadMPEC.HomotopySolverStats{T, VT}},
    probs::Any,
    name::AbstractString,
) where {T, VT}
    df_madnlp = DataFrame(
        name=names,
        success=[s.status == MadMPEC.NLP_STATIONARY for s in stats_madnlp],
        status=[s.status for s in stats_madnlp],
        objective=[s.objective for s in stats_madnlp],
        inf_cc=[s.inf_cc for s in stats_madnlp],
        wall_time=[s.wall_time for s in stats_madnlp],
        iter=[s.iter for s in stats_madnlp],
        eval_function_time=[s.eval_function_time for s in stats_madnlp],
        linear_solver_time=[s.linear_solver_time for s in stats_madnlp],
        homotopy_iter=[length(s.nlp_stats) for s in stats_madnlp],
    )
    CSV.write(name, df_madnlp)

    return df_madnlp
end

function save_madnlp_c_df(
    names::Vector{<:AbstractString},
    stats_madnlp_c::Vector{MadNLP.MadNLPExecutionStats{T, VT}},
    probs::Vector{MadMPEC.AbstractMPCCModel},
    name::AbstractString,
) where {T, VT}
    inf_cc =
        inf_cc=[
            MadMPEC.comp_residual_product(mpcc, s.solution) for
            (mpcc, s) in zip(probs, stats_madnlp_c)
        ]
    df_madnlp_c = DataFrame(
        name=names,
        success=[
            s.status in [
                MadNLP.SOLVE_SUCCEEDED,
                MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL,
                MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL,
            ] && cc ≤ 1e-8 for (s, cc) in zip(stats_madnlp_c, inf_cc)
        ],
        status=[s.status for s in stats_madnlp_c],
        objective=[s.objective for s in stats_madnlp_c],
        inf_cc=inf_cc,
        wall_time=[s.counters.total_time for s in stats_madnlp_c],
        iter=[s.counters.k for s in stats_madnlp_c],
        eval_function_time=[s.counters.eval_function_time for s in stats_madnlp_c],
        linear_solver_time=[s.counters.linear_solver_time for s in stats_madnlp_c],
    )
    CSV.write(name, df_madnlp_c)

    return df_madnlp_c
end

function save_ipopt_df(
    names::Vector{<:AbstractString},
    stats_ipopt::Vector{MadMPEC.HomotopySolverStats{T, VT}},
    probs::Any,
    name::AbstractString,
) where {T, VT}
    df_ipopt = DataFrame(
        name=names,
        success=[s.status == MadMPEC.NLP_STATIONARY for s in stats_ipopt],
        status=[s.status for s in stats_ipopt],
        objective=[s.objective for s in stats_ipopt],
        inf_cc=[s.inf_cc for s in stats_ipopt],
        wall_time=[s.wall_time for s in stats_ipopt],
        iter=[s.iter for s in stats_ipopt],
        homotopy_iter=[length(s.nlp_stats) for s in stats_ipopt],
    )
    CSV.write(name, df_ipopt)

    return df_ipopt
end

function save_ncl_df(
    names::Vector{<:AbstractString},
    stats_ncl::Vector{Union{MadNCL.NCLStats{T}, Nothing}},
    probs::Vector{<:MadMPEC.AbstractMPCCModel},
    name::AbstractString,
) where {T}
    inf_cc =
        inf_cc=[
            !isnothing(s) ? MadMPEC.comp_residual_product(mpcc, s.solution) : Inf for
            (mpcc, s) in zip(probs, stats_ncl)
        ]
    df_ncl = DataFrame(
        name=names,
        success=[
            (!isnothing(s) ? s.status : MadNLP.INTERNAL_ERROR) in [
                MadNLP.SOLVE_SUCCEEDED,
                MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL,
                MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL,
            ] && cc ≤ 1e-8 for (s, cc) in zip(stats_ncl, inf_cc)
        ],
        status=[!isnothing(s) ? s.status : MadNLP.INTERNAL_ERROR for s in stats_ncl],
        objective=[!isnothing(s) ? s.objective : Inf for s in stats_ncl],
        inf_cc=inf_cc,
        wall_time=[!isnothing(s) ? s.counters.total_time : Inf for s in stats_ncl],
        iter=[!isnothing(s) ? s.counters.k : -1 for s in stats_ncl],
        outer_iter=[!isnothing(s) ? s.iter : -1 for s in stats_ncl],
    )
    CSV.write(name, df_ncl)

    return df_ncl
end

function save_ncl_df(
    names::Vector{<:AbstractString},
    stats_ncl::Vector{Union{MadNCL.NCLStats{T}, Nothing}},
    probs::Any,
    name::AbstractString,
) where {T}
    inf_cc =
        inf_cc=[
            MadMPEC.comp_residual_product(mpcc, s.solution) for
            ((name, mpcc), s) in zip(probs, stats_ncl)
        ]
    df_ncl = DataFrame(
        name=names,
        success=[
            (!isnothing(s) ? s.status : MadNLP.INTERNAL_ERROR) in [
                MadNLP.SOLVE_SUCCEEDED,
                MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL,
                MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL,
            ] && cc ≤ 1e-8 for (s, cc) in zip(stats_ncl, inf_cc)
        ],
        status=[!isnothing(s) ? s.status : MadNLP.INTERNAL_ERROR for s in stats_ncl],
        objective=[!isnothing(s) ? s.objective : Inf for s in stats_ncl],
        inf_cc=inf_cc,
        wall_time=[!isnothing(s) ? s.counters.total_time : Inf for s in stats_ncl],
        iter=[!isnothing(s) ? s.counters.k : -1 for s in stats_ncl],
        outer_iter=[!isnothing(s) ? s.iter : -1 for s in stats_ncl],
    )
    CSV.write(name, df_ncl)

    return df_ncl
end

function perf_plot(
    title::AbstractString,
    names::Vector{<:AbstractString},
    stats;
    cost_col=:wall_time,
)
    costs = foldl(hcat, [stats[name][!, cost_col] for name in names])

    ii=1
    for name in names
        costs[findall(.!(stats[name].success)), ii] .= -1
        ii += 1
    end
    return performance_profile(PlotsBackend(), costs, names, title=title)
end

function solve_benchmark_problem(
    mpcc::MadMPEC.AbstractMPCCModel,
    opts::MadMPEC.HomotopySolverOptions,
    solver::Type,
)
    solver = MadMPEC.HomotopySolver(mpcc, solver, opts)

    return MadMPEC.solve!(solver)
end

function solve_benchmark_problem_madnlp_c(
    mpcc::MadMPEC.AbstractMPCCModel,
    opts::Dict{Symbol, Any},
)
    scholtes = MadMPEC.ScholtesRelaxation(mpcc)
    solver = MadNLP.MadNLPSolver(scholtes; opts...)
    stats = MadMPEC.solve_homotopy!(solver)
    return stats
end

function solve_benchmark_problem(
    mpcc::MadMPEC.AbstractMPCCModel,
    opts::MadNCL.NCLOptions,
    sol_args...,
)
    nlp = MadMPEC.ScholtesRelaxation(mpcc)

    try
        stats = MadNCL.madncl(nlp, ncl_options=opts; sol_args...)
        # print_level=MadNLP.ERROR,
        # linear_solver=Ma27Solver,

        return stats
    catch
        return nothing
    end
end

function run_benchmark(
    probs::Vector{<:MadMPEC.AbstractMPCCModel},
    solfun,
    opts::T,
    solargs...,
) where {T <: Dict}
    stats_vec = Vector{MadNLP.MadNLPExecutionStats{Float64, Vector{Float64}}}()
    sizehint!(stats_vec, length(probs))
    for i in 1:length(probs)
        push!(stats_vec, solfun(probs[i], opts, solargs...))
    end

    return stats_vec
end

function run_benchmark(
    probs::Vector{<:MadMPEC.AbstractMPCCModel},
    solfun,
    opts::T,
    solargs...,
) where {T <: MadMPEC.HomotopySolverOptions}
    stats_vec = Vector{MadMPEC.HomotopySolverStats{Float64, Vector{Float64}}}()
    sizehint!(stats_vec, length(probs))
    for i in 1:length(probs)
        push!(stats_vec, solfun(probs[i], opts, solargs...))
    end

    return stats_vec
end

function run_benchmark(
    probs::Vector{<:MadMPEC.AbstractMPCCModel},
    solfun,
    opts::T,
    solargs...,
) where {T <: MadNCL.NCLOptions}
    stats_vec = Vector{Union{Nothing, MadNCL.NCLStats{Float64}}}()
    sizehint!(stats_vec, length(probs))
    for i in 1:length(probs)
        push!(stats_vec, solfun(probs[i], opts, solargs...))
    end

    return stats_vec
end
