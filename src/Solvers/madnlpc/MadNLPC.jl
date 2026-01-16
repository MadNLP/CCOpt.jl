@kwdef mutable struct IterateLogger
    file::Union{IOStream, Nothing} = nothing
end

# Relaxation updates
abstract type AbstractRelaxationUpdate{T} end

"""
  Proportional Relaxation update which updates σ = aμ^b
"""
@kwdef struct ProportionalRelaxationUpdate{T} <: AbstractRelaxationUpdate{T}
    sigma_mu_ratio::T = 1.0 # a
    sigma_mu_exp::T = 1.0 # b
    monotone::Bool = false
end

@kwdef struct LOQORelaxationUpdate{T} <: AbstractRelaxationUpdate{T}
    gamma::T = 0.05 # scale factor
    gamma_min::T = 1e-5 # smallest factor of reduction allowed
    mu_factor::T = 1e-5 # smallest factor of reduction allowed
    r::T = 0.95 # Steplength param
end

"""
  Two-sided Scholtes relaxation attempts to drive either the scholtes bound x1*x2 - σ ≤ 0 to zero
  or the lower bounds of x1 and x2 choosing which to decrease by inspecting the estimated mpcc
  lagrange multipliers.
"""
@kwdef struct TwoSidedScholtesUpdate{T} <: AbstractRelaxationUpdate{T}
    kappa::T = 0.1
    k_ftb::T = 0.9
    tau::T = 0.3
end

"""
  Propotional Relaxation update which updates σ = aμ^b
  Also relaxes the complementarity lower bounds by mu_factor*μ when:
    μ ≤ relax_threshold
    the corresponding estimated mpec multiplier is negative (the scholtes bound is active)
    identified by the multiplier being smaller than μ^tau

  When unrelax is set to true we try to recover erroneously relaxed lower bounds (identified by the lower bound being active)
  this is done by taking steps which push the bound towards zero while making sure to reduce the distance from the iterate
  to the boundary by a factor of k_ftb.
"""
@kwdef mutable struct RelaxLBUpdate{T} <: AbstractRelaxationUpdate{T}
    sigma_mu_ratio::T = 1.0
    sigma_mu_exp::T = 1.0
    monotone::Bool = false
    mu_factor::T = 1.0
    delta_max::T = 1e-3
    tau::T = 0.4
    relax_threshold::T = 1e-6
    k_ftb::T = 0.9
    unrelax::Bool = false
    use_filtered::Bool = false
    reject_steps::Bool = false
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

    y::VT

    p::MadNLP.UnreducedKKTVector{T, VT}

    alpha_pr::T
    alpha_du::T

    ls::Int

    obj::T
    inf_pr::T
    inf_du::T
    theta::T
    varphi::T
    mu::T
    sigma::VT
    nu1::VT
    nu2::VT
    nu1_filt::VT
    nu2_filt::VT
    delta1::VT
    delta2::VT

    KKT_s::VT

    magic::Bool
end

# Options struct
@kwdef struct MadNLPCOptions{T} <: MadNLP.AbstractOptions
    # Relaxation type
    relaxation::Type = ScholtesRelaxation

    # adaptive mu update parameters
    use_specialized_barrier_update::Bool = false

    # complementarity homotopy options
    relaxation_update::AbstractRelaxationUpdate{T} = ProportionalRelaxationUpdate()
    sigma_min::T = 1e-10 # TODO(@anton) I think this should be probably be related to ipm tolerance
    delta_init::T = 0.0

    # initialization options
    respect_comp_bounds::Bool = false # Essentially don't relax complementarity variables
    center_complementarities::Bool = false
    centering_factor::T = 0.5

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

    # Reset slacks
    reset_slacks_on_update::Bool = false

    # MPEC multiplier estimation
    mpec_multiplier_filter_history::T = 1.01

    # Output options
    output_file::String = ""
    print_level::MadNLP.LogLevels = MadNLP.INFO
    file_print_level::MadNLP.LogLevels = MadNLP.INFO

    # Store Iterations
    iterates_fname::String = ""
end

@kwdef mutable struct MadNLPCCounters
    counters::MadNLP.MadNLPCounters
    solver_time::Float64 = 0
end

# MadNLP-C algorithm
mutable struct MadNLPCSolver{
    T,
    VT,
    MPCC <: AbstractMPCCModel{T, VT},
    RNLP <: AbstractMPCCRelaxation{T, VT},
    SOLVER <: MadNLP.MadNLPSolver{T, VT},
}
    const mpcc::MPCC
    const rnlp::RNLP
    const ipm::SOLVER
    const logger::MadNLP.MadNLPLogger
    const iterate_logger::IterateLogger
    const opts::MadNLPCOptions{T}
    const cnt::MadNLPCCounters
    inf_pr_cc::T
    const multipliers_cc1::VT
    const multipliers_cc2::VT
    const multipliers_cc1_filt::VT
    const multipliers_cc2_filt::VT
    const prev_delta1::VT
    const prev_delta2::VT
    const delta_rollback::Bool
    const ind_cc1::Vector{Int} # fixed indices in case of MakeParameter
    const ind_cc2::Vector{Int} # fixed indices in case of MakeParameter
    const ind_cc1_lb::Vector{Int}
    const ind_cc2_lb::Vector{Int}
    const x::VT
    const b::Vector{Bool}
end

# TODO(@anton) fix this to be nonquadratic I guess
# FIXME(@anton) This is broken for fixed complementarities. For now, ignore that.
function _adjust_cc_inds!(cb, ind_cc1, ind_cc2)
    fixed = cb.ind_fixed
    for ii in 1:length(ind_cc1)
        n_less1 = count(<(ind_cc1[ii]), fixed)
        n_less2 = count(<(ind_cc2[ii]), fixed)
        ind_cc1[ii] -= n_less1
        ind_cc2[ii] -= n_less2
    end
end

function MadNLPCSolver(
    mpcc::AbstractMPCCModel{T, VT};
    solver_opts=MadNLPCOptions(),
    ipm_options...,
) where {T, VT}
    rnlp = solver_opts.relaxation(mpcc)
    ipm = MadNLP.MadNLPSolver(rnlp; ipm_options...)
    initialize_relaxation(rnlp, ipm.opt.barrier.mu_init, solver_opts.delta_init)

    logger = MadNLP.MadNLPLogger(
        print_level=solver_opts.print_level,
        file_print_level=solver_opts.file_print_level,
        file=solver_opts.output_file == "" ? nothing : open(solver_opts.output_file, "w+"),
    )

    iterates_logger = IterateLogger(
        file=solver_opts.iterates_fname == "" ? nothing :
             open(solver_opts.iterates_fname, "w+"),
    )

    x = VT(undef, get_nvar(mpcc))
    multipliers_cc1 = VT(undef, get_ncc(mpcc))
    multipliers_cc2 = VT(undef, get_ncc(mpcc))
    multipliers_cc1_filt = VT(undef, get_ncc(mpcc))
    multipliers_cc2_filt = VT(undef, get_ncc(mpcc))
    prev_delta1 = VT(undef, get_ncc(mpcc))
    prev_delta2 = VT(undef, get_ncc(mpcC))
    b = Vector{Bool}(undef, get_ncc(mpcc))
    cnt = MadNLPCCounters(counters=ipm.cnt)
    # TODO(@anton) Can we do this nonquadratically
    ind_cc1 = copy(get_ind_cc1(mpcc))
    ind_cc2 = copy(get_ind_cc2(mpcc))
    println("ind_cc1_orig: $(ind_cc1)")
    println("ind_cc2_orig: $(ind_cc2)")
    _adjust_cc_inds!(ipm.cb, ind_cc1, ind_cc2)
    println("ind_cc1: $(ind_cc1)")
    println("ind_cc2: $(ind_cc2)")
    ind_cc1_lb = map((i)->findfirst((j)->i==j, ipm.kkt.ind_lb), ind_cc1)
    ind_cc2_lb = map((i)->findfirst((j)->i==j, ipm.kkt.ind_lb), ind_cc2)
    return solver = MadNLPCSolver(
        mpcc,
        rnlp,
        ipm,
        logger,
        iterates_logger,
        solver_opts,
        cnt,
        0.0,
        multipliers_cc1,
        multipliers_cc2,
        multipliers_cc1_filt,
        multipliers_cc2_filt,
        prev_delta1,
        prev_delta2,
        false,
        ind_cc1,
        ind_cc2,
        ind_cc1_lb,
        ind_cc2_lb,
        x,
        b,
    )
end

include("utils.jl")
include("kernels.jl")
include("barrier.jl")
include("relaxation.jl")
include("madnlpc.jl")
