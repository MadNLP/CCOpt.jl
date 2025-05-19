include("from_ampl.jl")

function solve_macmpec_ampl_problem(nl::AbstractString, opts::MadMPEC.HomotopySolverOptions)
    model = AmplNLReader.AmplModel(nl)
    mpcc = mpcc_from_ampl(model)

    solver = MadMPEC.HomotopySolver(mpcc, opts)

    return solve!(solver)
end
