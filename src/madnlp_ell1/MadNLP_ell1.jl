@kwdef struct MadNLPEll1Options{T}

    # complementarity homotopy options
    sigma_0::T = 1.0
    sigma_growth_rate::T = 10.0

    # Algorithm options
    dynamic_sigma_update::Bool = false # Switch between classic and dynamic algorithm from
    # Leyffer2005 paper
    comp_history_length::Int = 3 # Length of history buffer (default from Leyffer2005)
    eta_dynamic_update::T = 0.9 # "sufficient decrease" parameter (default from Leyffer2005)

    # Output options
    output_file::String = ""
    print_level::MadNLP.LogLevels = MadNLP.INFO
    file_print_level::MadNLP.LogLevels = MadNLP.INFO

    # Plot Iterations
    plot_iterates::Bool = false
end

struct MadNLPEll1Solver{T, VT}
    mpcc::AbstractMPCCModel{T, VT}
    ell1::Ell1Relaxation{T, VT}
    ipm::MadNLP.MadNLPSolver{T, VT}
    logger::MadNLP.MadNLPLogger
    opts::MadNLPEll1Options{T}

    pr_comp_hist::CircularBuffer{T} # Complementarity history
end

function MadNLPEll1Solver(
    mpcc::AbstractMPCCModel{T, VT};
    madnlpell1_opts=MadNLPEll1Options(),
    ipm_options...,
) where {T, VT}
    ell1 = Ell1Relaxation(mpcc)
    ipm = MadNLP.MadNLPSolver(ell1; ipm_options...)

    logger = MadNLP.MadNLPLogger(
        print_level=madnlpell1_opts.print_level,
        file_print_level=madnlpell1_opts.file_print_level,
        file=madnlpell1_opts.output_file == "" ? nothing :
             open(madnlpc_opts.output_file, "w+"),
    )

    pr_comp_hist = CircularBuffer{T}(madnlpell1_opts.comp_history_length)
    return MadNLPEll1Solver(mpcc, ell1, ipm, logger, madnlpell1_opts, pr_comp_hist)
end

include("madnlp_ell1.jl")
