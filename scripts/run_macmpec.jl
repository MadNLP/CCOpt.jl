include("from_ampl.jl")
using NLPModelsIpopt

function solve_macmpec_ampl_problem(nl::AbstractString, opts::MadMPEC.HomotopySolverOptions)
    model = AmplNLReader.AmplModel(nl)
    mpcc = MadMPEC.vertical_form(mpcc_from_ampl(model))

    solver = MadMPEC.HomotopySolver(mpcc, NLPModelsIpopt.IpoptSolver, opts)

    return MadMPEC.solve!(solver)
end

function solve_macmpec(datapath::AbstractString, opts::MadMPEC.HomotopySolverOptions)
    probs = readdir(abspath(datapath), join=true)
    stats_vec = Vector{MadMPEC.HomotopySolverStats}(undef, length(probs))
    for i in 1:length(probs)
        try
            stats_vec[i] = solve_macmpec_ampl_problem(probs[i], opts)
        catch e
            println("Something went wrong with ", probs[i])
        else
            println(stats_vec[i])
        end
    end

    return probs, stats_vec
end
