@kwdef mutable struct IterateLogger
    file::Union{IOStream, Nothing} = nothing
end

struct MadNLPCIterate{T, VT}
    k::Int

    x0::VT
    x1::VT
    x2::VT
    s::VT

    z1::VT
    z2::VT
    zs::VT

    alpha_pr::T
    alpha_du::T

    ls::Int

    obj::T
    inf_pr::T
    inf_du::T
    theta::T
    varphi::T
    mu::T

    KKT_s::VT

    magic::Bool
end

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

    # Store Iterations
    iterates_fname::String = ""
end

struct MadNLPCSolver{T, VT}
    mpcc::AbstractMPCCModel{T, VT}
    scholtes::ScholtesRelaxation{T, VT}
    ipm::MadNLP.MadNLPSolver{T, VT}
    logger::MadNLP.MadNLPLogger
    iterate_logger::IterateLogger
    opts::MadNLPCOptions{T}
end

function MadNLPCSolver(
    mpcc::AbstractMPCCModel{T, VT};
    solver_opts=MadNLPCOptions(),
    ipm_options...,
) where {T, VT}
    scholtes = ScholtesRelaxation(mpcc)
    ipm = MadNLP.MadNLPSolver(scholtes; ipm_options...)

    logger = MadNLP.MadNLPLogger(
        print_level=solver_opts.print_level,
        file_print_level=solver_opts.file_print_level,
        file=solver_opts.output_file == "" ? nothing : open(solver_opts.output_file, "w+"),
    )

    iterates_logger = IterateLogger(
        file=solver_opts.iterates_fname == "" ? nothing :
             open(solver_opts.iterates_fname, "w+"),
    )

    return MadNLPCSolver(mpcc, scholtes, ipm, logger, iterates_logger, solver_opts)
end

include("utils.jl")
include("madnlpc.jl")
