include("../common.jl")
include("generate_benchmark.jl")
using Distributed

function run_benchmark(
    probs::RandomMPCCBenchmark,
    solfun,
    opts::T,
    solargs...,
) where {T <: MadMPEC.HomotopySolverOptions}
    stats_vec = Vector{MadMPEC.HomotopySolverStats{Float64, Vector{Float64}}}()
    sizehint!(stats_vec, length(probs))
    names = Vector{String}()
    for (name, prob) in probs
        push!(stats_vec, solfun(prob, opts, solargs...))
        push!(names, name)
        println(name)
    end

    return names, stats_vec
end

function run_benchmark_procs(
    probs::RandomMPCCBenchmark,
    solfun,
    opts::T,
    solargs...,
) where {T <: MadMPEC.HomotopySolverOptions}
    nprobs = length(probs)
    stats_vec = Vector{MadMPEC.HomotopySolverStats{Float64, Vector{Float64}}}(undef, nprobs)
    names = Vector{String}()
    futures = []

    nw = nworkers()

    if nw != 8
        rmprocs(workers()...)
        addprocs(8)
        @everywhere include(joinpath(@__DIR__, "run_random_benchmark.jl"))
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

function run_benchmark(
    probs::RandomMPCCBenchmark,
    solfun,
    opts::T,
    solargs...,
) where {T <: Dict}
    stats_vec = Vector{MadNLP.MadNLPExecutionStats{Float64, Vector{Float64}}}()
    sizehint!(stats_vec, length(probs))
    names = Vector{String}()
    for (name, prob) in probs
        push!(stats_vec, solfun(prob, opts, solargs...))
        push!(names, name)
        println(name)
    end

    return names, stats_vec
end

function run_benchmark_procs(
    probs::RandomMPCCBenchmark,
    solfun,
    opts::T,
    solargs...,
) where {T <: Dict}
    nprobs = length(probs)
    stats_vec = Vector{MadNLP.MadNLPExecutionStats{Float64, Vector{Float64}}}(undef, nprobs)
    names = Vector{String}()
    futures = []

    nw = nworkers()

    if nw != 8
        rmprocs(workers()...)
        addprocs(8)
        @everywhere include(joinpath(@__DIR__, "run_random_benchmark.jl"))
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

function run_benchmark(
    probs::RandomMPCCBenchmark,
    solfun,
    opts::T,
    solargs...,
) where {T <: MadNCL.NCLOptions}
    stats_vec = Vector{Union{Nothing, MadNCL.NCLStats{Float64}}}()
    sizehint!(stats_vec, length(probs))
    names = Vector{String}()
    for (name, prob) in probs
        push!(stats_vec, solfun(prob, opts, solargs...))
        push!(names, name)
        println(name)
    end

    return names, stats_vec
end

function run_benchmark_procs(
    probs::RandomMPCCBenchmark,
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
        @everywhere include(joinpath(@__DIR__, "run_random_benchmark.jl"))
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

function run_random_benchmark(
    args...;
    bench_name="random",
    plot=false,
    n_probs=7,
    max_size=300,
    multiproc=false,
)
    stats = Dict{String, Any}()
    probs = RandomMPCCBenchmark(n_probs, nl_funs, 3; max_size=max_size)
    solnames = Vector{String}()
    for (solname::AbstractString, solfun, dffun, opts, solargs) in args
        if multiproc
            names, stats[solname] = run_benchmark_procs(probs, solfun, opts, solargs...)
        else
            names, stats[solname] = run_benchmark(probs, solfun, opts, solargs...)
        end
        push!(solnames, solname)
        stats[solname] = dffun(
            names,
            stats[solname],
            probs,
            replace(bench_name*"_$(n_probs)_"*"$(max_size)_"*solname, " "=>"_")*".csv",
        )
    end
    if plot
        perf_plot("Performance Plot", solnames, stats)
    end
    return solnames, names, stats
end

function test_random_benchmark(; n_probs=7, max_size=300, multiproc=false)
    opts_ipopt = MadMPEC.HomotopySolverOptions()
    opts_ipopt.print_level = MadNLP.INFO
    opts_ipopt.nlp_solver_options[:print_level] = 0
    opts_ipopt.nlp_solver_options[:max_iter] = 3000
    opts_ipopt.nlp_solver_options[:linear_solver] = "ma27"
    opts_madnlp = MadMPEC.HomotopySolverOptions()
    opts_madnlp.print_level = MadNLP.INFO
    opts_madnlp.nlp_solver_options = Dict(
        :bound_relax_factor=>1e-12,
        :print_level=>MadNLP.ERROR,
        :max_iter=>3000,
        :linear_solver=>Ma27Solver,
    )

    opts_ncl = MadNCL.NCLOptions();

    opts_madnlp_c =
        solver_options = Dict(
            :bound_relax_factor=>1e-12,
            :print_level=>MadNLP.ERROR,
            :max_iter=>3000,
            :linear_solver=>Ma27Solver,
        )

    default_madnlp_c = (
        "random ma27 madNLP-C",
        solve_benchmark_problem_madnlp_c,
        save_madnlp_c_df,
        opts_madnlp_c,
        (),
    )
    default_ipopt = (
        "random ma27 Ipopt",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt,
        (NLPModelsIpopt.IpoptSolver,),
    )
    default_madnlp = (
        "random ma27 madNLP",
        solve_benchmark_problem,
        save_madnlp_df,
        opts_madnlp,
        (MadNLP.MadNLPSolver,),
    )
    default_madncl = (
        "random ma27 madNCL",
        solve_benchmark_problem,
        save_ncl_df,
        opts_ncl,
        ((:print_level, MadNLP.ERROR), (:linear_solver, Ma27Solver)),
    )

    solnames, names, stats = run_random_benchmark(
        default_madnlp_c,
        default_madncl,
        default_madnlp,
        default_ipopt;
        n_probs=n_probs,
        max_size=max_size,
        multiproc=multiproc,
    )

    # solnames, names, stats = run_random_benchmark(
    #     default_madnlp_c;
    #     n_probs=n_probs,
    #     max_size=max_size,
    #     multiproc=multiproc
    # )

    # solnames, names, stats =
    #     run_random_benchmark(default_madnlp, default_ipopt; n_probs=n_probs)

    return solnames, names, stats
end
