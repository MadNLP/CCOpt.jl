@kwdef struct PenaltyOptions{T}
    penalty::Type = Ell1Relaxation
    # complementarity homotopy options
    rho_0::T = 1.0
    rho_max::T = 1e10
    rho_growth_rate::T = 10.0
    gamma::T = 0.4

    # Algorithm options
    dynamic_rho_update::Bool = true # Switch between classic and dynamic algorithm from
    # Leyffer2006 paper
    comp_history_length::Int = 10 # Length of history buffer (default from Leyffer2006)
    eta_dynamic_update::T = 0.99 # "sufficient decrease" parameter (default from Leyffer2006)

    # regularization
    q_regularization::Symbol = :critical_rho
    min_eig_value::T = 1e-8
    max_eig_value::T = Inf
    critical_rho_factor::T = 0.99
    min_reg_mu::T = 5e-6
    max_reg_mu::T = 1e-1

    # Output options
    output_file::String = ""
    print_level::MadNLP.LogLevels = MadNLP.INFO
    file_print_level::MadNLP.LogLevels = MadNLP.INFO
end

mutable struct PenaltySolver{
    T,
    VT,
    MPCC <: AbstractMPCCModel{T, VT},
    PNLP <: AbstractMPCCPenaltyModel{T, VT},
    SOLVER <: MadNLP.MadNLPSolver{T, VT},
}
    const mpcc::MPCC
    const pnlp::PNLP
    const ipm::SOLVER
    const logger::MadNLP.MadNLPLogger
    const opts::PenaltyOptions{T}

    inf_pr_cc::T
    const ind_cc1::Vector{Int} # fixed indices in case of MakeParameter
    const ind_cc2::Vector{Int} # fixed indices in case of MakeParameter

    const pr_comp_hist::CircularBuffer{T} # Complementarity history
end

"""
  This solver implements the two algorithms in the paper:
  Interior Methods for Mathematical Programs with Complementarity Constraints
  Sven Leyffer, Gabriel López-Calva, and Jorge Nocedal
  SIAM Journal on Optimization Vol. 17, Issue 1 (2006)
  epubs.siam.org/doi/10.1137/040621065
"""
function PenaltySolver(
    mpcc::AbstractMPCCModel{T, VT};
    solver_opts=PenaltyOptions(),
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
    ind_cc1 = copy(get_ind_cc1(mpcc))
    ind_cc2 = copy(get_ind_cc2(mpcc))
    _adjust_cc_inds!(ipm.cb, ind_cc1, ind_cc2)
    return PenaltySolver(
        mpcc,
        pnlp,
        ipm,
        logger,
        solver_opts,
        0.0,
        ind_cc1,
        ind_cc2,
        pr_comp_hist,
    )
end
include("utils.jl")
include("exact_penalty.jl")
