using NLPModelsIpopt
using MadNLP
using MadNCL

include("from_ampl.jl")
include("save_macmpec.jl")

function solve_benchmark_problem(
    mpcc::MadMPEC.AbstractMPCCModel,
    opts::MadMPEC.HomotopySolverOptions,
    solver::Type,
)
    solver = MadMPEC.HomotopySolver(mpcc, solver, opts)

    return MadMPEC.solve!(solver)
end

function solve_benchmark_problem(mpcc::MadMPEC.AbstractMPCCModel, opts::MadNCL.NCLOptions)
    nlp = MadMPEC.ScholtesRelaxation(mpcc)

    try
        return MadNCL.madncl(nlp, ncl_options=opts, print_level=MadNLP.ERROR)
    catch
        return nothing
    end
end

function load_ampl_benchmark(nlpath::AbstractString)
    probs = readdir(abspath(nlpath), join=true)
    mpccs::Vector{MadMPEC.AbstractMPCCModel} = []
    names = Vector{String}()
    for i in 1:length(probs)
        model = AmplNLReader.AmplModel(probs[i])
        mpcc = MadMPEC.vertical_form(mpcc_from_ampl(model))
        push!(mpccs, mpcc)
        push!(names, basename(probs[i]))
    end
    return names, mpccs
end

function run_benchmark(
    probs::Vector{MadMPEC.AbstractMPCCModel},
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
    probs::Vector{MadMPEC.AbstractMPCCModel},
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

function run_macmpec(args...; plot=false, range=:)
    # Take a list of named tuples:
    stats = Dict{String, Any}()
    names, probs = load_ampl_benchmark(joinpath(dirname(@__FILE__), "../data/macMPEC/nls/"))
    solnames = Vector{String}()
    for (solname::AbstractString, solfun, dffun, opts, solargs) in args
        # TODO(@anton) Figure out why this is necessary. Something in mpccmodel is _too_ mutable
        stats[solname] = run_benchmark(probs[range], solfun, opts, solargs...)
        push!(solnames, solname)
        stats[solname] =
            dffun(names[range], stats[solname], replace(solname, " "=>"_")*".csv")
    end
    if plot
        perf_plot("Performance Plot", solnames, stats)
    end
    return solnames, names, stats
end

function test_macmpec(; range=:)
    opts_ipopt = MadMPEC.HomotopySolverOptions()
    opts_ipopt.print_level = 4
    opts_ipopt.nlp_solver_options[:print_level] = 5
    opts_ipopt.nlp_solver_options[:max_iter] = 500
    opts_madnlp = MadMPEC.HomotopySolverOptions()
    opts_madnlp.nlp_solver_options =
        Dict(:bound_relax_factor=>1e-12, :print_level=>MadNLP.DEBUG, :max_iter=>500)

    opts_ncl = MadNCL.NCLOptions();
    #opts_madnlp.nlp_solver_options = Dict(:print_level=>MadNLP.INFO)

    default_ipopt = (
        "nosnoc Ipopt",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt,
        (NLPModelsIpopt.IpoptSolver,),
    )
    default_madnlp = (
        "default madNLP",
        solve_benchmark_problem,
        save_madnlp_df,
        opts_madnlp,
        (MadNLP.MadNLPSolver,),
    )
    default_madncl = ("default madNCL", solve_benchmark_problem, save_ncl_df, opts_ncl, ())

    solnames, names, stats =
        run_macmpec(default_ipopt, default_madnlp, default_madncl; range=range)

    return solnames, names, stats
end

function test_ipopt_mu_update(; range=:)
    opts_ipopt_monotone = MadMPEC.HomotopySolverOptions()
    opts_ipopt_monotone.print_level = 5
    opts_ipopt_monotone.nlp_solver_options[:print_level] = 5
    opts_ipopt_monotone.nlp_solver_options[:max_iter] = 3000
    opts_ipopt_monotone.nlp_solver_options[:mu_strategy] = "monotone"

    opts_ipopt_adaptive_quality = MadMPEC.HomotopySolverOptions()
    opts_ipopt_adaptive_quality.print_level = 5
    opts_ipopt_adaptive_quality.nlp_solver_options[:print_level] = 5
    opts_ipopt_adaptive_quality.nlp_solver_options[:max_iter] = 3000
    opts_ipopt_adaptive_quality.nlp_solver_options[:mu_strategy] = "adaptive"
    opts_ipopt_adaptive_quality.nlp_solver_options[:mu_oracle] = "quality-function"

    opts_ipopt_adaptive_probing = MadMPEC.HomotopySolverOptions()
    opts_ipopt_adaptive_probing.print_level = 5
    opts_ipopt_adaptive_probing.nlp_solver_options[:print_level] = 5
    opts_ipopt_adaptive_probing.nlp_solver_options[:max_iter] = 3000
    opts_ipopt_adaptive_probing.nlp_solver_options[:mu_strategy] = "adaptive"
    opts_ipopt_adaptive_probing.nlp_solver_options[:mu_oracle] = "probing"

    opts_ipopt_adaptive_loqo = MadMPEC.HomotopySolverOptions()
    opts_ipopt_adaptive_loqo.print_level = 5
    opts_ipopt_adaptive_loqo.nlp_solver_options[:print_level] = 5
    opts_ipopt_adaptive_loqo.nlp_solver_options[:max_iter] = 3000
    opts_ipopt_adaptive_loqo.nlp_solver_options[:mu_strategy] = "adaptive"
    opts_ipopt_adaptive_loqo.nlp_solver_options[:mu_oracle] = "loqo"

    ipopt_monotone = (
        "Ipopt monotone",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt_monotone,
        (NLPModelsIpopt.IpoptSolver,),
    )
    ipopt_adaptive_quality = (
        "Ipopt adaptive quality function",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt_adaptive_quality,
        (NLPModelsIpopt.IpoptSolver,),
    )
    ipopt_adaptive_probing = (
        "Ipopt adaptive probing",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt_adaptive_probing,
        (NLPModelsIpopt.IpoptSolver,),
    )
    ipopt_adaptive_loqo = (
        "Ipopt adaptive loqo",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt_adaptive_loqo,
        (NLPModelsIpopt.IpoptSolver,),
    )

    solnames, names, stats = run_macmpec(
        ipopt_monotone,
        ipopt1,
        ipopt_adaptive_probing,
        ipopt_adaptive_loqo;
        range=range,
    )

    return solnames, names, stats
end

function test_homtotopy_bound_push(; range=:)
    opts_ipopt1 = MadMPEC.HomotopySolverOptions()
    opts_ipopt1.print_level = 5
    opts_ipopt1.nlp_solver_options[:print_level] = 5
    opts_ipopt1.nlp_solver_options[:max_iter] = 3000

    ipopt1 = (
        "Ipopt bp default",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt1,
        (NLPModelsIpopt.IpoptSolver,),
    )

    opts_ipopt2 = MadMPEC.HomotopySolverOptions(warm_start_bound_push=1e-5)
    opts_ipopt2.print_level = 5
    opts_ipopt2.nlp_solver_options[:print_level] = 5
    opts_ipopt2.nlp_solver_options[:max_iter] = 3000

    ipopt2 = (
        "Ipopt bp 1e-5",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt2,
        (NLPModelsIpopt.IpoptSolver,),
    )

    opts_ipopt3 = MadMPEC.HomotopySolverOptions(warm_start_bound_push=1e-9)
    opts_ipopt3.print_level = 5
    opts_ipopt3.nlp_solver_options[:print_level] = 5
    opts_ipopt3.nlp_solver_options[:max_iter] = 3000

    ipopt3 = (
        "Ipopt bp 1e-9",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt3,
        (NLPModelsIpopt.IpoptSolver,),
    )

    opts_madnlp1 = MadMPEC.HomotopySolverOptions()
    opts_madnlp1.print_level = 5
    opts_madnlp1.nlp_solver_options =
        Dict(:bound_relax_factor=>1e-12, :print_level=>MadNLP.ERROR, :max_iter=>3000)

    madnlp1 = (
        "madNLP bp default",
        solve_benchmark_problem,
        save_madnlp_df,
        opts_madnlp1,
        (MadNLP.MadNLPSolver,),
    )

    opts_madnlp2 = MadMPEC.HomotopySolverOptions(warm_start_bound_push=1e-5)
    opts_madnlp2.print_level = 5
    opts_madnlp2.nlp_solver_options =
        Dict(:bound_relax_factor=>1e-12, :print_level=>MadNLP.ERROR, :max_iter=>3000)

    madnlp2 = (
        "madNLP bp 1e-5",
        solve_benchmark_problem,
        save_madnlp_df,
        opts_madnlp2,
        (MadNLP.MadNLPSolver,),
    )

    opts_madnlp3 = MadMPEC.HomotopySolverOptions(warm_start_bound_push=1e-9)
    opts_madnlp3.print_level = 5
    opts_madnlp3.nlp_solver_options =
        Dict(:bound_relax_factor=>1e-12, :print_level=>MadNLP.ERROR, :max_iter=>3000)

    madnlp3 = (
        "madNLP bp 1e-9",
        solve_benchmark_problem,
        save_madnlp_df,
        opts_madnlp3,
        (MadNLP.MadNLPSolver,),
    )

    solnames, names, stats =
        run_macmpec(ipopt1, ipopt2, ipopt3, madnlp1, madnlp2, madnlp3; range=range)

    return solnames, names, stats
end

function test_ipopt_norm_type(; range=:)
    opts_ipopt_1_norm = MadMPEC.HomotopySolverOptions()
    opts_ipopt_1_norm.print_level = MadNLP.INFO
    opts_ipopt_1_norm.nlp_solver_options[:print_level] = 0
    opts_ipopt_1_norm.nlp_solver_options[:max_iter] = 3000
    opts_ipopt_1_norm.nlp_solver_options[:mu_strategy] = "adaptive"
    opts_ipopt_1_norm.nlp_solver_options[:mu_oracle] = "quality-function"
    opts_ipopt_1_norm.nlp_solver_options[:quality_function_norm_type] = "1-norm"

    opts_ipopt_2_norm_squared = MadMPEC.HomotopySolverOptions()
    opts_ipopt_2_norm_squared.print_level = MadNLP.INFO
    opts_ipopt_2_norm_squared.nlp_solver_options[:print_level] = 0
    opts_ipopt_2_norm_squared.nlp_solver_options[:max_iter] = 3000
    opts_ipopt_2_norm_squared.nlp_solver_options[:mu_strategy] = "adaptive"
    opts_ipopt_2_norm_squared.nlp_solver_options[:mu_oracle] = "quality-function"
    opts_ipopt_2_norm_squared.nlp_solver_options[:quality_function_norm_type] = "2-norm-squared"

    opts_ipopt_max_norm = MadMPEC.HomotopySolverOptions()
    opts_ipopt_max_norm.print_level = MadNLP.INFO
    opts_ipopt_max_norm.nlp_solver_options[:print_level] = 0
    opts_ipopt_max_norm.nlp_solver_options[:max_iter] = 3000
    opts_ipopt_max_norm.nlp_solver_options[:mu_strategy] = "adaptive"
    opts_ipopt_max_norm.nlp_solver_options[:mu_oracle] = "quality-function"
    opts_ipopt_max_norm.nlp_solver_options[:quality_function_norm_type] = "max-norm"

    opts_ipopt_2_norm = MadMPEC.HomotopySolverOptions()
    opts_ipopt_2_norm.print_level = MadNLP.INFO
    opts_ipopt_2_norm.nlp_solver_options[:print_level] = 0
    opts_ipopt_2_norm.nlp_solver_options[:max_iter] = 3000
    opts_ipopt_2_norm.nlp_solver_options[:mu_strategy] = "adaptive"
    opts_ipopt_2_norm.nlp_solver_options[:mu_oracle] = "quality-function"
    opts_ipopt_2_norm.nlp_solver_options[:quality_function_norm_type] = "2-norm"

    ipopt_1_norm = (
        "Ipopt 1-norm",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt_1_norm,
        (NLPModelsIpopt.IpoptSolver,),
    )
    ipopt_2_norm_squared = (
        "Ipopt 2-norm-squared",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt_2_norm_squared,
        (NLPModelsIpopt.IpoptSolver,),
    )
    ipopt_max_norm = (
        "Ipopt max-norm",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt_max_norm,
        (NLPModelsIpopt.IpoptSolver,),
    )
    ipopt_2_norm = (
        "Ipopt 2-norm",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt_2_norm,
        (NLPModelsIpopt.IpoptSolver,),
    )

    solnames, names, stats = run_macmpec(
        ipopt_1_norm,
        ipopt_2_norm_squared,
        ipopt_max_norm,
        ipopt_2_norm;
        range=range,
    )

    return solnames, names, stats
end
