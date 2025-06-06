@kwdef struct MadNLPCOptions{T}

    # complementarity homotopy options
    sigma_mu_ratio::T = 1.0

    # regularization options
    kkt_regularization::Symbol = :none # Options: :vicente_wright
    mu_thresh::T = 5e-6

    # Magic step options
    use_magic_step::Bool = false
    magic_step_kappa::T = 0.9
    magic_step_projection_heuristic::Symbol = :min_f

    # Output options
    output_file::String = ""
    print_level::MadNLP.LogLevels = MadNLP.INFO
    file_print_level::MadNLP.LogLevels = MadNLP.INFO

    # Plot Iterations
    plot_iterates::Bool = false
end

struct MadNLPCSolver{T, VT}
    mpcc::AbstractMPCCModel{T, VT}
    scholtes::ScholtesRelaxation{T, VT}
    ipm::MadNLP.MadNLPSolver{T, VT}
    logger::MadNLP.MadNLPLogger
    opts::MadNLPCOptions{T}
end

function MadNLPCSolver(
    mpcc::AbstractMPCCModel{T, VT};
    madnlpc_opts=MadNLPCOptions(),
    ipm_options...,
) where {T, VT}
    scholtes = ScholtesRelaxation(mpcc)
    ipm = MadNLP.MadNLPSolver(scholtes; ipm_options...)

    logger = MadNLP.MadNLPLogger(
        print_level=madnlpc_opts.print_level,
        file_print_level=madnlpc_opts.file_print_level,
        file=madnlpc_opts.output_file == "" ? nothing :
             open(madnlpc_opts.output_file, "w+"),
    )

    return MadNLPCSolver(mpcc, scholtes, ipm, logger, madnlpc_opts)
end

include("madnlpc.jl")
