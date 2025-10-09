@kwdef struct ExactPenaltyOptions{T}
    # complementarity homotopy options
    tau_0::T = 1.0
    tau_growth_rate::T = 10.0
    gamma::T = 0.4

    # Algorithm options
    dynamic_tau_update::Bool = false # Switch between classic and dynamic algorithm from
    # Leyffer2006 paper
    comp_history_length::Int = 3 # Length of history buffer (default from Leyffer2006)
    eta_dynamic_update::T = 0.9 # "sufficient decrease" parameter (default from Leyffer2006)

    # Output options
    output_file::String = ""
    print_level::MadNLP.LogLevels = MadNLP.INFO
    file_print_level::MadNLP.LogLevels = MadNLP.INFO
end

struct ExactPenaltySolver{T, VT}
    mpcc::AbstractMPCCModel{T, VT}
    ell1::Ell1Relaxation{T, VT}
    ipm::MadNLP.MadNLPSolver{T, VT}
    logger::MadNLP.MadNLPLogger
    opts::ExactPenaltyOptions{T}

    pr_comp_hist::CircularBuffer{T} # Complementarity history
end

"""
  This solver implements the two algorithms in the paper:
  Interior Methods for Mathematical Programs with Complementarity Constraints
  Sven Leyffer, Gabriel López-Calva, and Jorge Nocedal
  SIAM Journal on Optimization Vol. 17, Issue 1 (2006)
  epubs.siam.org/doi/10.1137/040621065
"""
function ExactPenaltySolver(
    mpcc::AbstractMPCCModel{T, VT};
    solver_opts=ExactPenaltyOptions{Float64}(),
    ipm_options...,
) where {T, VT}
    ell1 = Ell1Relaxation(mpcc)
    ipm = MadNLP.MadNLPSolver(ell1; ipm_options...)

    logger = MadNLP.MadNLPLogger(
        print_level=solver_opts.print_level,
        file_print_level=solver_opts.file_print_level,
        file=solver_opts.output_file == "" ? nothing : open(solver_opts.output_file, "w+"),
    )

    pr_comp_hist = CircularBuffer{T}(solver_opts.comp_history_length)
    return ExactPenaltySolver(mpcc, ell1, ipm, logger, solver_opts, pr_comp_hist)
end

include("exact_penalty.jl")
