@kwdef mutable struct IterateLogger
    file::Union{IOStream, Nothing} = nothing
end

# Relaxation updates
abstract type AbstractRelaxationUpdate{T} end

@kwdef struct ProportionalRelaxationUpdate{T} <: AbstractRelaxationUpdate{T}
    sigma_mu_ratio::T = 1.0
    monotone::Bool = false
end

@kwdef struct LOQORelaxationUpdate{T} <: AbstractRelaxationUpdate{T}
    gamma::T = 0.1 # scale factor
    gamma_min::T = 1e-5 # smallest factor of reduction allowed
    mu_factor::T = 1e-3 # smallest factor of reduction allowed
    r::T = 0.95 # Steplength param
end

# Iterate saving structure
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
    sigma::T

    KKT_s::VT

    magic::Bool
end

# Options struct
@kwdef struct MadNLPCOptions{T}
    # adaptive mu update parameters
    use_specialized_barrier_update::Bool = true

    # complementarity homotopy options
    relaxation_update::AbstractRelaxationUpdate{T} = ProportionalRelaxationUpdate()
    sigma_min::T = 1e-9 # TODO(@anton) I think this should be probably be related to ipm tolerance

    # initialization options
    respect_comp_bounds::Bool = false # Essentially don't relax complementarity variables

    # regularization options
    kkt_regularization::Symbol = :vicente_wright # Options: :vicente_wright
    mu_thresh::T = 5e-6
    eta_factor::T = 0.1

    # Magic step options
    use_magic_step::Bool = false
    magic_step_kappa::T = 0.5
    magic_step_projection_heuristic::Symbol = :min_f
    magic_step_duals::Bool = true
    magic_step_slack::Bool = true
    magic_step_slack_dual::Bool = true

    # mpecopt options
    use_mpecopt::Bool = false
    eps_proj::T = 1e-1
    alpha_eps_proj::T = 1e-1
    M_lpcc::T = 100.0

    # Output options
    output_file::String = ""
    print_level::MadNLP.LogLevels = MadNLP.INFO
    file_print_level::MadNLP.LogLevels = MadNLP.INFO

    # Store Iterations
    iterates_fname::String = ""
end

# MadNLP-C algorithm
mutable struct MadNLPCSolver{T, VT}
    mpcc::AbstractMPCCModel{T, VT}
    scholtes::ScholtesRelaxation{T, VT}
    ipm::MadNLP.MadNLPSolver{T, VT}
    logger::MadNLP.MadNLPLogger
    iterate_logger::IterateLogger
    opts::MadNLPCOptions{T}

    status::Status

    lpcc::LpccMILP{T, VT}
    bnlp_ipm::MadNLP.MadNLPSolver{T, VT}
    eps_proj::T

    x::VT
    b::Vector{Bool} # TODO(@anton) is it actually better to have a Vector{Bool}
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

    lpcc = LpccMILP(mpcc; M=solver_opts.M_lpcc)
    eps_proj = solver_opts.eps_proj
    x = VT(undef, mpcc.meta.nvar)
    b = Vector{Bool}(undef, mpcc.meta.ncc)
    bnlp = BranchNLP(mpcc, b)
    bnlp_ipm = MadNLP.MadNLPSolver(bnlp) # TODO(@anton) also pass the bnlp options somehow

    return MadNLPCSolver(
        mpcc,
        scholtes,
        ipm,
        logger,
        iterates_logger,
        solver_opts,
        INITIAL,
        lpcc,
        bnlp_ipm,
        eps_proj,
        x,
        b,
    )
end

include("utils.jl")
include("kernels.jl")
include("barrier.jl")
include("relaxation.jl")
include("madnlpc.jl")
