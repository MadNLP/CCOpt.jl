using NLPModelsIpopt
using MadNLP
using MadNCL

include("from_ampl.jl")
include("save_macmpec.jl")

function solve_benchmark_problem_homotopy(
    mpcc::MadMPEC.AbstractMPCCModel,
    opts::MadMPEC.HomotopySolverOptions,
    solver::Type,
)
    solver = MadMPEC.HomotopySolver(mpcc, solver, opts)

    return MadMPEC.solve!(solver)
end

function solve_benchmark_problem_ncl(
    mpcc::MadMPEC.AbstractMPCCModel,
    opts::MadNCL.NCLOptions,
)
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
        names, probs =
            load_ampl_benchmark(joinpath(dirname(@__FILE__), "../data/macMPEC/nls/"))
        stats[solname] = run_benchmark(probs[range], solfun, opts, solargs...)
        push!(solnames, solname)
        stats[solname] =
            dffun(names[range], stats[solname], replace(solname, " "=>"_")*".csv")
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
        solve_benchmark_problem_homotopy,
        save_ipopt_df,
        opts_ipopt,
        (NLPModelsIpopt.IpoptSolver,),
    )
    default_madnlp = (
        "default madNLP",
        solve_benchmark_problem_homotopy,
        save_madnlp_df,
        opts_madnlp,
        (MadNLP.MadNLPSolver,),
    )
    default_madncl =
        ("default madNCL", solve_benchmark_problem_ncl, save_ncl_df, opts_ncl, ())

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

    opts_ipopt1 = MadMPEC.HomotopySolverOptions()
    opts_ipopt1.print_level = 5
    opts_ipopt1.nlp_solver_options[:print_level] = 5
    opts_ipopt1.nlp_solver_options[:max_iter] = 3000
    opts_ipopt1.nlp_solver_options[:mu_strategy] = "adaptive"
    opts_ipopt1.nlp_solver_options[:mu_oracle] = "quality-function"

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
        solve_benchmark_problem_homotopy,
        save_ipopt_df,
        opts_ipopt_monotone,
        (NLPModelsIpopt.IpoptSolver,),
    )
    ipopt1 = (
        "Ipopt adaptive quality function",
        solve_benchmark_problem_homotopy,
        save_ipopt_df,
        opts_ipopt1,
        (NLPModelsIpopt.IpoptSolver,),
    )
    ipopt_adaptive_probing = (
        "Ipopt adaptive probing",
        solve_benchmark_problem_homotopy,
        save_ipopt_df,
        opts_ipopt_adaptive_probing,
        (NLPModelsIpopt.IpoptSolver,),
    )
    ipopt_adaptive_loqo = (
        "Ipopt adaptive loqo",
        solve_benchmark_problem_homotopy,
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
    opts_ipopt1.nlp_solver_options[:mu_strategy] = "adaptive"
    opts_ipopt1.nlp_solver_options[:mu_oracle] = "quality-function"

    ipopt1 = (
        "Ipopt bp default",
        solve_benchmark_problem_homotopy,
        save_ipopt_df,
        opts_ipopt1,
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
