@kwdef struct ExactPenaltyOptions{T}
    penalty::Type = Ell1Relaxation
    # complementarity homotopy options
    tau_0::T = 1.0
    tau_max::T = 1e8
    tau_growth_rate::T = 10.0
    gamma::T = 0.4

    # Algorithm options
    dynamic_tau_update::Bool = false # Switch between classic and dynamic algorithm from
    # Leyffer2006 paper
    comp_history_length::Int = 5 # Length of history buffer (default from Leyffer2006)
    eta_dynamic_update::T = 0.99 # "sufficient decrease" parameter (default from Leyffer2006)

    # regularization
    kkt_regularization::Symbol = :none
    min_eig_value::T = 1e-4
    max_eig_value::T = Inf
    critical_rho_factor::T = 0.9
    min_reg_mu::T = 1e-5

    # Output options
    output_file::String = ""
    print_level::MadNLP.LogLevels = MadNLP.INFO
    file_print_level::MadNLP.LogLevels = MadNLP.INFO
end

mutable struct ExactPenaltySolver{T, VT}
    mpcc::AbstractMPCCModel{T, VT}
    ell1::AbstractMPCCPenaltyModel{T, VT}
    ipm::MadNLP.MadNLPSolver{T, VT}
    logger::MadNLP.MadNLPLogger
    opts::ExactPenaltyOptions{T}

    inf_pr_cc::T

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
    pnlp = solver_opts.penalty(mpcc)
    ipm = MadNLP.MadNLPSolver(pnlp; ipm_options...)

    logger = MadNLP.MadNLPLogger(
        print_level=solver_opts.print_level,
        file_print_level=solver_opts.file_print_level,
        file=solver_opts.output_file == "" ? nothing : open(solver_opts.output_file, "w+"),
    )

    pr_comp_hist = CircularBuffer{T}(solver_opts.comp_history_length)
    return ExactPenaltySolver(mpcc, pnlp, ipm, logger, solver_opts, 0.0, pr_comp_hist)
end
include("utils.jl")
include("exact_penalty.jl")
