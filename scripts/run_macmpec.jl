include("from_ampl.jl")
using NLPModelsIpopt
using MadNLP

function solve_benchmark_problem(
    mpcc::MadMPEC.AbstractMPCCModel,
    opts::MadMPEC.HomotopySolverOptions,
    solver::Type,
)
    solver = MadMPEC.HomotopySolver(mpcc, solver, opts)

    return MadMPEC.solve!(solver)
end

function load_ampl_benchmark(nlpath::AbstractString)
    probs = readdir(abspath(nlpath), join=true)
    mpccs::Vector{MadMPEC.AbstractMPCCModel} = []
    names = []
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
    opts::MadMPEC.HomotopySolverOptions,
    solver::Type,
)
    stats_vec = Vector{MadMPEC.HomotopySolverStats{Float64, Vector{Float64}}}()
    sizehint!(stats_vec, length(probs))
    for i in 1:length(probs)
        push!(stats_vec, solve_benchmark_problem(probs[i], opts, solver))
    end

    return stats_vec
end

opts_ipopt = MadMPEC.HomotopySolverOptions()
opts_ipopt.print_level = 4
opts_ipopt.nlp_solver_options[:print_level] = 5
opts_madnlp = MadMPEC.HomotopySolverOptions()
opts_madnlp.nlp_solver_options =
    Dict(:bound_relax_factor=>1e-12, :print_level=>MadNLP.DEBUG)
#opts_madnlp.nlp_solver_options = Dict(:print_level=>MadNLP.INFO)
names, probs = load_ampl_benchmark(joinpath(dirname(@__FILE__), "../data/macMPEC/nls/"))
stats_madnlp = run_benchmark(probs, opts_madnlp, MadNLP.MadNLPSolver)
names, probs = load_ampl_benchmark(joinpath(dirname(@__FILE__), "../data/macMPEC/nls/"))
stats_ipopt = run_benchmark(probs, opts_ipopt, NLPModelsIpopt.IpoptSolver)
