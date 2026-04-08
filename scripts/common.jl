using CSV
using DataFrames
using Plots, BenchmarkProfiles
using AmplNLReader, CCOpt
using NLPModelsIpopt
using MadNLP, MadNLPHSL
using MadNCL
using Distributed
using LaTeXStrings, PythonPlot
using NaNMath

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
    stats_madnlp_c::Vector{CCOpt.CCOptExecutionStats{T, VT}},
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
        fact=[s.counters.counters.factorization_cnt for s in stats_madnlp_c],
        eval_function_time=[s.counters.counters.eval_function_time for s in stats_madnlp_c],
        linear_solver_time=[s.counters.counters.linear_solver_time for s in stats_madnlp_c],
        solver_time=[s.counters.solver_time for s in stats_madnlp_c],
    )
    CSV.write(name, df_madnlp_c)

    return df_madnlp_c
end

function save_madnlp_c_df(
    names::Vector{<:AbstractString},
    stats_madnlp_c::Vector{CCOpt.CCOptExecutionStats{T, VT}},
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
    for v in values(stats)
        sort!(v, :name)
    end
    costs = foldl(hcat, [stats[name][!, cost_col] for name in names])

    ii=1
    for name in names
        costs[findall(.!(stats[name].success .== true)), ii] .= -1
        ii += 1
    end
    return performance_profile(PlotsBackend(), costs, names, title=title; plot_args...)
end

function abs_profile_data(
    T::Array{Float64, 2};
    logscale::Bool=true,
    sampletol::Float64=0.0,
    drawtol::Float64=0.0,
)
    vals = copy(T)
    max_val = NaNMath.maximum(vals)
    failures = isnan.(vals)
    vals[failures] .= 2.0*max_val
    sort!(vals, dims=1)
    logscale && vals .= log10.(vals)
    (np, ns) = size(vals)
    max_val = NaNMath.maximum(vals)

    vals = [vals; 2.0 * max_val * ones(1, ns)]
    xs = [1:(np+1);] / np

    x_plot = Vector{Vector{Float64}}(undef, ns)
    y_plot = Vector{Vector{Float64}}(undef, ns)

    for s in 1:ns
        rs = view(vals, :, s)
        xidx = zeros(Int, length(rs) + 1)
        k = 0
        rv = minimum(rs)
        maxval = maximum(rs)
        #println("1 ", xidx)
        while rv < max_val
            k += 1
            xidx[k] = findlast(rs .<= rv)
            rv = max(rs[xidx[k]] + sampletol, rs[xidx[k]+1])
            #println(k)
        end
        xidx[k+1] = length(rs)
        xidx = xidx[xidx .> 0]
        xidx = unique(xidx) # Needed?
        yidx = copy(xidx)
        yidx[end] = yidx[end-1]
        #println(xs)

        x_plot[s] = rs[xidx]
        y_plot[s] = xs[yidx]

        #println(x_plot[s])
        println(y_plot[s])
    end
    return (x_plot, y_plot, max_val)
end

function abs_profile_axis_labels(labels, ns, logscale; kwargs...)
    length(labels) == 0 && (labels = ["column $col" for col in 1:ns])
    kwargs = Dict{Symbol, Any}(kwargs)
    xlabel = pop!(kwargs, :xlabel, "Wall Time (s)")
    ylabel = pop!(kwargs, :ylabel, "Proportion of problems")
    return (xlabel, ylabel, labels)
end

function powertick_10(s::AbstractString)
    codes = Dict(collect(".1234567890") .=> collect("⋅¹²³⁴⁵⁶⁷⁸⁹⁰"))
    hidden_latex = !isnothing(match(r"^\$.*\$$", s))
    ex = r"[0-9.]+"
    for m in eachmatch(ex, s)
        s = if hidden_latex
            replace(s, m.match => "10^{$(m.match)}")
        else
            replace(s, m.match => "10" * map(c -> codes[c], m.match))
        end
    end
    return s
end

function abs_profile_plot(
    ::PlotsBackend,
    x_plot,
    y_plot,
    max_val,
    xlabel,
    ylabel,
    labels,
    title,
    logscale;
    kwargs...,
)
    kwargs = Dict{Symbol, Any}(kwargs)
    linestyles = pop!(kwargs, :linestyles, Symbol[])
    colors = pop!(kwargs, :colors, [])
    profile = Plots.plot(
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        xlims=(1.1*minimum(minimum.(x_plot)), 2.0 * max_val),
        ylims=(0, 1.1),
    )  # initial plot
    for s in 1:length(labels)
        if length(linestyles) > 0
            kwargs[:linestyle] = linestyles[s]
        end
        if length(colors) > 0
            kwargs[:color] = colors[s]
        end
        Plots.plot!(x_plot[s], y_plot[s], t=:steppost, label=labels[s]; kwargs...)  # add to initial plot
    end
    if logscale
        for xt in Plots.xticks(profile)
            Plots.plot!(xticks=(xt[1], map(x -> powertick_10(x), xt[2])))
            Plots.plot!(xtickfontsize=10)
        end
    end
    return profile
end

function abs_profile(
    b::PlotsBackend,
    T::Matrix{Td},
    labels::Vector{S}=String[];
    logscale::Bool=true,
    title::AbstractString="",
    sampletol::Float64=0.0,
    drawtol::Float64=0.0,
    kwargs...,
) where {Td <: Number, S <: AbstractString}
    xlabel, ylabel, labels =
        abs_profile_axis_labels(labels, size(T, 2), logscale; kwargs...)
    (x_plot, y_plot, max_ratio) =
        abs_profile_data(T, logscale=logscale, sampletol=sampletol, drawtol=drawtol)
    return abs_profile_plot(
        b,
        x_plot,
        y_plot,
        max_ratio,
        xlabel,
        ylabel,
        labels,
        title,
        logscale;
        kwargs...,
    )
end

function abs_plot(
    title::AbstractString,
    names::Vector{<:AbstractString},
    stats;
    cost_col=:wall_time,
    plot_args...,
)
    for v in values(stats)
        sort!(v, :name)
    end
    costs = foldl(hcat, [stats[name][!, cost_col] for name in names])

    ii=1
    for name in names
        costs[findall(.!(stats[name].success .== true)), ii] .= NaN
        ii += 1
    end

    return abs_profile(PlotsBackend(), costs, names, title=title; plot_args...)
end

function load_from_matlab_csv(filepath)
    data = DataFrame(CSV.File(filepath))
    rename!(data, Dict(:problem_name => :name, :wall_time_total => :wall_time))
    solnames = unique(data.solver_name)
    names = unique(data.name)

    stats = Dict()
    for solname in solnames
        stats[solname] = sort(data[data.solver_name .== solname, :], :name)
    end
    return solnames, names, stats
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
    stats_vec = Vector{CCOpt.CCOptExecutionStats{Float64, Vector{Float64}}}()
    sizehint!(stats_vec, length(probs))
    for i in 1:length(probs)
        println(probs[i].nlp.nlp.meta.name)
        push!(stats_vec, solfun(probs[i], opts, solargs...))
        GC.gc()
    end

    return stats_vec
end

function run_benchmark(
    probs::Vector{<:CCOpt.AbstractMPCCModel},
    solfun,
    opts::CCOpt.PenaltyOptions,
    solargs...,
)
    stats_vec = Vector{CCOpt.CCOptExecutionStats{Float64, Vector{Float64}}}()
    sizehint!(stats_vec, length(probs))
    for i in 1:length(probs)
        println(probs[i].nlp.nlp.meta.name)
        push!(stats_vec, solfun(probs[i], opts, solargs...))
        GC.gc()
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
        GC.gc()
    end

    return stats_vec
end

function run_benchmark_procs(probs, solfun, opts::CCOpt.RelaxationOptions, solargs...)
    nprobs = length(probs)
    stats_vec = Vector{MadNLP.CCOptExecutionStats{Float64, Vector{Float64}}}(undef, nprobs)
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

function plot_benchmark(solnames, names, stats; cost_col=:wall_time, savepath="test.pdf")
    default()
    default(
        titlefont=(20, "serif"),
        legendfont=(9, "serif"),
        guidefont=(10, "serif"),
        tickfontsize=10,
        linewidth=3,
    )
    pythonplot()

    styles = Dict(
        "CCOpt Relaxation" => (RGBA(0, 0.4470, 0.7410), :solid),
        "CCOpt Penalty" => (RGBA(0, 0.4470, 0.7410), :dash),
        "IPOPT Homotopy" => (RGBA(0.8500, 0.3250, 0.0980), :solid),
        "IPOPT Penalty" => (RGBA(0.8500, 0.3250, 0.0980), :dash),
        "Gurobi SOS1" => (RGBA(0.4940, 0.1840, 0.5560), :solid),
        "Gurobi MIQP" => (RGBA(0.4940, 0.1840, 0.5560), :dash),
        "LCQPow qpOASES" => (RGBA(0.4660, 0.6740, 0.1880), :solid),
        "LCQPow OSQP" => (RGBA(0.4660, 0.6740, 0.1880), :dash),
    )
    colors = map((x) -> styles[x][1], solnames)
    line_styles = map((x) -> styles[x][2], solnames)

    display(
        perf_plot(
            "",
            solnames,
            stats;
            size=(400, 300),
            cost_col=cost_col,
            legend=:bottomright,
            ylabel="fraction solved",
            xlabel=L"$2^{x}$ times best",
            linestyles=line_styles,
            colors=colors,
        ),
    )
    return PythonPlot.savefig(savepath)
end

function plot_benchmark_abs(
    solnames,
    names,
    stats;
    cost_col=:wall_time,
    savepath="test.pdf",
)
    default()
    default(
        titlefont=(20, "serif"),
        legendfont=(9, "serif"),
        guidefont=(10, "serif"),
        tickfontsize=10,
        linewidth=3,
    )
    pythonplot()

    styles = Dict(
        "CCOpt Relaxation" => (RGBA(0, 0.4470, 0.7410), :solid),
        "CCOpt Penalty" => (RGBA(0, 0.4470, 0.7410), :dash),
        "IPOPT Homotopy" => (RGBA(0.8500, 0.3250, 0.0980), :solid),
        "IPOPT Penalty" => (RGBA(0.8500, 0.3250, 0.0980), :dash),
        "Gurobi SOS1" => (RGBA(0.4940, 0.1840, 0.5560), :solid),
        "Gurobi MIQP" => (RGBA(0.4940, 0.1840, 0.5560), :dash),
        "LCQPow qpOASES" => (RGBA(0.4660, 0.6740, 0.1880), :solid),
        "LCQPow OSQP" => (RGBA(0.4660, 0.6740, 0.1880), :dash),
    )
    colors = map((x) -> styles[x][1], solnames)
    line_styles = map((x) -> styles[x][2], solnames)

    display(
        abs_plot(
            "",
            solnames,
            stats;
            size=(400, 300),
            cost_col=cost_col,
            legend=:bottomright,
            ylabel="fraction solved",
            xlabel="Wall Time (s)",
            linestyles=line_styles,
            colors=colors,
        ),
    )
    return PythonPlot.savefig(savepath)
end
