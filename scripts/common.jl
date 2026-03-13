using CSV
using DataFrames
using Plots, BenchmarkProfiles
using AmplNLReader, CCOpt
using NLPModelsIpopt
using MadNLP, MadNLPHSL
using MadNCL
using Distributed

function mpcc_from_ampl(ampl::AmplNLReader.AmplModel)
    # First we find the nonzero elements in the cvar vector:
    # cvar = 0 → normal nonlinear constraint
    # cvar ≠ 0 → complementarity with the cvar-th variable
    ind_ccc2 = findall(!iszero, ampl.meta.cvar)
    # Then we store the corresponding variables
    ind_vcc1 = ampl.meta.cvar[ind_ccc2]
    return CCOpt.MPCCModelVarCon(ampl, ind_vcc1, ind_ccc2)
end

function save_madnlp_df(
    names::Vector{<:AbstractString},
    stats_madnlp::Vector{CCOpt.HomotopySolverStats{T, VT}},
    probs::Any,
    name::AbstractString,
) where {T, VT}
    df_madnlp = DataFrame(
        name=names,
        success=[
            s.status in [MadNLP.SOLVE_SUCCEEDED, MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL] for
            s in stats_madnlp
        ],
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
    stats_madnlp::Vector{CCOpt.HomotopySolverStats{T, VT}},
    probs::Any,
    name::AbstractString,
) where {T, VT}
    df_madnlp = DataFrame(
        name=names,
        success=[
            s.status in [MadNLP.SOLVE_SUCCEEDED, MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL] for
            s in stats_madnlp
        ],
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
    probs::Vector{CCOpt.AbstractMPCCModel},
    name::AbstractString,
) where {T, VT}
    inf_cc =
        inf_cc=[
            min(
                CCOpt.comp_residual(mpcc, s.solution),
                CCOpt.comp_residual_product(mpcc, s.solution),
            ) for (mpcc, s) in zip(probs, stats_madnlp_c)
        ]
    df_madnlp_c = DataFrame(
        name=names,
        success=[
            s.status in [
                MadNLP.SOLVE_SUCCEEDED,
                MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL,
                MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL,
            ] &&
            cc ≤ 5e-8 &&
            s.primal_feas ≤ 5e-8 for (s, cc) in zip(stats_madnlp_c, inf_cc)
        ],
        status=[s.status for s in stats_madnlp_c],
        objective=[s.objective for s in stats_madnlp_c],
        inf_cc=inf_cc,
        inf_pr=[s.primal_feas for s in stats_madnlp_c],
        inf_du=[s.dual_feas for s in stats_madnlp_c],
        wall_time=[s.counters.total_time for s in stats_madnlp_c],
        iter=[s.counters.k for s in stats_madnlp_c],
        eval_function_time=[s.counters.eval_function_time for s in stats_madnlp_c],
        linear_solver_time=[s.counters.linear_solver_time for s in stats_madnlp_c],
    )
    CSV.write(name, df_madnlp_c)

    return df_madnlp_c
end

function save_madnlp_c_df(
    names::Vector{<:AbstractString},
    stats_madnlp_c::Vector{MadNLP.MadNLPExecutionStats{T, VT}},
    probs::Any,
    name::AbstractString,
) where {T, VT}
    inf_cc =
        inf_cc=[
            min(
                CCOpt.comp_residual(mpcc, s.solution),
                CCOpt.comp_residual_product(mpcc, s.solution),
            ) for ((name, mpcc), s) in zip(probs, stats_madnlp_c)
        ]
    df_madnlp_c = DataFrame(
        name=names,
        success=[
            s.status in [
                MadNLP.SOLVE_SUCCEEDED,
                MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL,
                MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL,
            ] &&
            cc ≤ 1e-8 &&
            s.primal_feas ≤ 1e-8 for (s, cc) in zip(stats_madnlp_c, inf_cc)
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

function save_madnlp_c_df(
    names::Vector{<:AbstractString},
    stats_madnlp_c::Vector{CCOpt.RelaxationExecutionStats{T, VT}},
    probs::Vector{CCOpt.AbstractMPCCModel},
    name::AbstractString,
) where {T, VT}
    inf_cc = inf_cc=[s.inf_pr_cc for (mpcc, s) in zip(probs, stats_madnlp_c)]
    df_madnlp_c = DataFrame(
        name=names,
        success=[
            s.status in [
                MadNLP.SOLVE_SUCCEEDED,
                MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL,
                MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL,
            ] &&
            cc ≤ 1e-8 &&
            s.primal_feas ≤ 5e-8 for (s, cc) in zip(stats_madnlp_c, inf_cc)
        ],
        status=[s.status for s in stats_madnlp_c],
        objective=[s.objective for s in stats_madnlp_c],
        inf_cc=inf_cc,
        inf_pr=[s.primal_feas for s in stats_madnlp_c],
        inf_du=[s.dual_feas for s in stats_madnlp_c],
        wall_time=[s.counters.counters.total_time for s in stats_madnlp_c],
        iter=[s.counters.counters.k for s in stats_madnlp_c],
        eval_function_time=[s.counters.counters.eval_function_time for s in stats_madnlp_c],
        linear_solver_time=[s.counters.counters.linear_solver_time for s in stats_madnlp_c],
        solver_time=[s.counters.solver_time for s in stats_madnlp_c],
    )
    CSV.write(name, df_madnlp_c)

    return df_madnlp_c
end

function save_madnlp_c_df(
    names::Vector{<:AbstractString},
    stats_madnlp_c::Vector{CCOpt.RelaxationExecutionStats{T, VT}},
    probs::Any,
    name::AbstractString,
) where {T, VT}
    inf_cc =
        inf_cc=[
            min(
                CCOpt.comp_residual(mpcc, s.solution),
                CCOpt.comp_residual_product(mpcc, s.solution),
            ) for ((name, mpcc), s) in zip(probs, stats_madnlp_c)
        ]
    df_madnlp_c = DataFrame(
        name=names,
        success=[
            s.stats.status in [
                MadNLP.SOLVE_SUCCEEDED,
                MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL,
                MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL,
            ] &&
            cc ≤ 1e-8 &&
            s.stats.primal_feas ≤ 1e-8 for (s, cc) in zip(stats_madnlp_c, inf_cc)
        ],
        status=[s.status for s in stats_madnlp_c],
        objective=[s.objective for s in stats_madnlp_c],
        inf_cc=inf_cc,
        wall_time=[s.counters.total_time for s in stats_madnlp_c],
        iter=[s.counters.counters.k for s in stats_madnlp_c],
        eval_function_time=[
            s.counters..counters.eval_function_time for s in stats_madnlp_c
        ],
        linear_solver_time=[s.counters.counters.linear_solver_time for s in stats_madnlp_c],
    )
    CSV.write(name, df_madnlp_c)

    return df_madnlp_c
end

function save_ipopt_df(
    names::Vector{<:AbstractString},
    stats_ipopt::Vector{CCOpt.HomotopySolverStats{T, VT}},
    probs::Any,
    name::AbstractString,
) where {T, VT}
    df_ipopt = DataFrame(
        name=names,
        success=[
            s.status in [MadNLP.SOLVE_SUCCEEDED, MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL] for
            s in stats_ipopt
        ],
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
    probs::Vector{<:CCOpt.AbstractMPCCModel},
    name::AbstractString,
) where {T}
    inf_cc =
        inf_cc=[
            !isnothing(s) ?
            min(
                CCOpt.comp_residual(mpcc, s.solution),
                CCOpt.comp_residual_product(mpcc, s.solution),
            ) : Inf for (mpcc, s) in zip(probs, stats_ncl)
        ]
    df_ncl = DataFrame(
        name=names,
        success=[
            (!isnothing(s) ? s.status : MadNLP.INTERNAL_ERROR) in [
                MadNLP.SOLVE_SUCCEEDED,
                MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL,
                MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL,
            ] &&
            cc ≤ 1e-8 &&
            s.primal_feas ≤ 1e-8 for (s, cc) in zip(stats_ncl, inf_cc)
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
            !isnothing(s) ?
            min(
                CCOpt.comp_residual(mpcc, s.solution),
                CCOpt.comp_residual_product(mpcc, s.solution),
            ) : Inf for ((name, mpcc), s) in zip(probs, stats_ncl)
        ]
    df_ncl = DataFrame(
        name=names,
        success=[
            (!isnothing(s) ? s.status : MadNLP.INTERNAL_ERROR) in [
                MadNLP.SOLVE_SUCCEEDED,
                MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL,
                MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL,
            ] &&
            cc ≤ 1e-8 &&
            s.primal_feas ≤ 1e-8 for (s, cc) in zip(stats_ncl, inf_cc)
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
    plot_args...,
)
    costs = foldl(hcat, [stats[name][!, cost_col] for name in names])

    ii=1
    for name in names
        costs[findall(.!(stats[name].success)), ii] .= -1
        ii += 1
    end
    return performance_profile(PlotsBackend(), costs, names, title=title; plot_args...)
end

function solve_benchmark_problem(
    mpcc::CCOpt.AbstractMPCCModel,
    opts::CCOpt.HomotopySolverOptions,
    solver::Type,
)
    solver = CCOpt.HomotopySolver(mpcc, solver, opts)

    return CCOpt.solve!(solver)
end

function solve_benchmark_problem(
    mpcc::CCOpt.AbstractMPCCModel,
    opts::CCOpt.RelaxationOptions,
    sol_args...,
)
    solver = CCOpt.RelaxationSolver(mpcc; solver_opts=opts, sol_args...)
    stats = CCOpt.solve_homotopy!(solver)
    return stats
end

function solve_benchmark_problem(
    mpcc::CCOpt.AbstractMPCCModel,
    opts::CCOpt.PenaltyOptions,
    sol_args...,
)
    solver = CCOpt.PenaltySolver(mpcc; solver_opts=opts, sol_args...)
    stats = CCOpt.solve_homotopy!(solver)
    return stats
end

function solve_benchmark_problem(
    mpcc::CCOpt.AbstractMPCCModel,
    opts::MadNCL.NCLOptions,
    sol_args...,
)
    nlp = CCOpt.ScholtesRelaxation(mpcc)

    try
        stats = MadNCL.madncl(nlp, ncl_options=opts; sol_args...)

        return stats
    catch
        return nothing
    end
end

function run_benchmark(
    probs::Vector{<:CCOpt.AbstractMPCCModel},
    solfun,
    opts::CCOpt.RelaxationOptions,
    solargs...,
)
    stats_vec = Vector{CCOpt.RelaxationExecutionStats{Float64, Vector{Float64}}}()
    sizehint!(stats_vec, length(probs))
    for i in 1:length(probs)
        println(probs[i].nlp.nlp.meta.name)
        push!(stats_vec, solfun(probs[i], opts, solargs...))
    end

    return stats_vec
end

function run_benchmark(
    probs::Vector{<:CCOpt.AbstractMPCCModel},
    solfun,
    opts::CCOpt.PenaltyOptions,
    solargs...,
)
    stats_vec = Vector{MadNLP.MadNLPExecutionStats{Float64, Vector{Float64}}}()
    sizehint!(stats_vec, length(probs))
    for i in 1:length(probs)
        println(probs[i].nlp.nlp.meta.name)
        push!(stats_vec, solfun(probs[i], opts, solargs...))
    end

    return stats_vec
end

function run_benchmark(
    probs::Vector{<:CCOpt.AbstractMPCCModel},
    solfun,
    opts::T,
    solargs...,
) where {T <: CCOpt.HomotopySolverOptions}
    stats_vec = Vector{CCOpt.HomotopySolverStats{Float64, Vector{Float64}}}()
    sizehint!(stats_vec, length(probs))
    for i in 1:length(probs)
        push!(stats_vec, solfun(probs[i], opts, solargs...))
    end

    return stats_vec
end

function run_benchmark(
    probs::Vector{<:CCOpt.AbstractMPCCModel},
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

function run_benchmark_procs(probs, solfun, opts::CCOpt.RelaxationOptions, solargs...)
    nprobs = length(probs)
    stats_vec =
        Vector{MadNLP.RelaxationExecutionStats{Float64, Vector{Float64}}}(undef, nprobs)
    names = Vector{String}()
    futures = []

    nw = nworkers()

    if nw != 8
        rmprocs(workers()...)
        addprocs(8)
        @everywhere include(joinpath(@__DIR__, "common.jl"))
        @everywhere include(joinpath(@__DIR__, "random_benchmark/run_random_benchmark.jl"))
    end
    wp = default_worker_pool()
    (pair, state) = iterate(probs)
    while true
        if !isready(wp)
            sleep(0.1)
            continue
        end
        (name, prob) = pair
        f = Future(1)
        errormonitor(@async put!(f, remotecall_fetch(solfun, wp, prob, opts, solargs...)))
        push!(futures, f)
        push!(names, name)
        println(name)
        next = iterate(probs, state)
        if isnothing(next)
            break
        end
        (pair, state) = next
    end

    for ii in 1:length(futures)
        if !isassigned(stats_vec, ii)
            while !isready(futures[ii])
                sleep(0.1)
            end
            stats_vec[ii] = fetch(futures[ii])
        end
    end

    return names, stats_vec
end

function run_benchmark_procs(probs, solfun, opts::CCOpt.PenaltyOptions, solargs...)
    nprobs = length(probs)
    stats_vec = Vector{MadNLP.MadNLPExecutionStats{Float64, Vector{Float64}}}(undef, nprobs)
    names = Vector{String}()
    futures = []

    nw = nworkers()

    if nw != 8
        rmprocs(workers()...)
        addprocs(8)
        @everywhere include(joinpath(@__DIR__, "common.jl"))
        @everywhere include(joinpath(@__DIR__, "random_benchmark/run_random_benchmark.jl"))
    end
    wp = default_worker_pool()
    (pair, state) = iterate(probs)
    while true
        if !isready(wp)
            sleep(0.1)
            continue
        end
        (name, prob) = pair
        f = Future(1)
        errormonitor(@async put!(f, remotecall_fetch(solfun, wp, prob, opts, solargs...)))
        push!(futures, f)
        push!(names, name)
        println(name)
        next = iterate(probs, state)
        if isnothing(next)
            break
        end
        (pair, state) = next
    end

    for ii in 1:length(futures)
        if !isassigned(stats_vec, ii)
            while !isready(futures[ii])
                sleep(0.1)
            end
            stats_vec[ii] = fetch(futures[ii])
        end
    end

    return names, stats_vec
end

function run_benchmark_procs(
    probs,
    solfun,
    opts::T,
    solargs...,
) where {T <: CCOpt.HomotopySolverOptions}
    nprobs = length(probs)
    stats_vec = Vector{CCOpt.HomotopySolverStats{Float64, Vector{Float64}}}(undef, nprobs)
    names = Vector{String}()
    futures = []

    nw = nworkers()

    if nw != 8
        rmprocs(workers()...)
        addprocs(8)
        @everywhere include(joinpath(@__DIR__, "common.jl"))
    end
    wp = default_worker_pool()
    (pair, state) = iterate(probs)
    while true
        if !isready(wp)
            sleep(0.1)
            continue
        end
        (name, prob) = pair
        f = Future(1)
        errormonitor(@async put!(f, remotecall_fetch(solfun, wp, prob, opts, solargs...)))
        push!(futures, f)
        push!(names, name)
        println(name)
        next = iterate(probs, state)
        if isnothing(next)
            break
        end
        (pair, state) = next
    end

    for ii in 1:length(futures)
        if !isassigned(stats_vec, ii)
            while !isready(futures[ii])
                sleep(0.1)
            end
            stats_vec[ii] = fetch(futures[ii])
        end
    end

    return names, stats_vec
end

function run_benchmark_procs(
    probs,
    solfun,
    opts::T,
    solargs...,
) where {T <: MadNCL.NCLOptions}
    nprobs = length(probs)
    stats_vec = Vector{Union{Nothing, MadNCL.NCLStats{Float64}}}(undef, nprobs)
    names = Vector{String}()
    futures = []

    nw = nworkers()

    if nw != 8
        rmprocs(workers()...)
        addprocs(8)
        @everywhere include(joinpath(@__DIR__, "common.jl"))
    end
    wp = default_worker_pool()
    (pair, state) = iterate(probs)
    while true
        if !isready(wp)
            sleep(0.1)
            continue
        end
        (name, prob) = pair
        f = Future(1)
        errormonitor(@async put!(f, remotecall_fetch(solfun, wp, prob, opts, solargs...)))
        push!(futures, f)
        push!(names, name)
        println(name)
        next = iterate(probs, state)
        if isnothing(next)
            break
        end
        (pair, state) = next
    end

    for ii in 1:length(futures)
        if !isassigned(stats_vec, ii)
            while !isready(futures[ii])
                sleep(0.1)
            end
            stats_vec[ii] = fetch(futures[ii])
        end
    end

    return names, stats_vec
end

function run_benchmark_threads(
    probs,
    solfun,
    opts::T,
    solargs...,
) where {T <: Union{CCOpt.RelaxationOptions, CCOpt.PenaltyOptions}}
    nprobs = length(probs)
    stats_vec = Vector{MadNLP.MadNLPExecutionStats{Float64, Vector{Float64}}}(undef, nprobs)
    names = Vector{String}()
    futures = []

    nw = nworkers()

    if nw != 8
        rmprocs(workers()...)
        addprocs(8)
        @everywhere include(joinpath(@__DIR__, "common.jl"))
    end
    wp = default_worker_pool()
    (pair, state) = iterate(probs)
    while true
        if !isready(wp)
            sleep(0.1)
            continue
        end
        (name, prob) = pair
        f = Future(1)
        errormonitor(@async put!(f, remotecall_fetch(solfun, wp, prob, opts, solargs...)))
        push!(futures, f)
        push!(names, name)
        println(name)
        next = iterate(probs, state)
        if isnothing(next)
            break
        end
        (pair, state) = next
    end

    for ii in 1:length(futures)
        if !isassigned(stats_vec, ii)
            while !isready(futures[ii])
                sleep(0.1)
            end
            stats_vec[ii] = fetch(futures[ii])
        end
    end

    return names, stats_vec
end
