@enum(MpecoptStatus::Int,
      B_STATIONARY=1,
      SMALL_STEP=2,
      INFEASIBLE=3,
      MAX_ITER=4
      )
struct MpecoptExecutionStats{T,VT}
    status::MpecoptStatus

    solution::VT
    objective::T
    constraints::VT
    dual_feas::T
    primal_feas::T
    multipliers::VT
    multipliers_L::VT
    multipliers_U::VT
    iter::Int
    counters::MadNLPCounters
end

@kwdef mutable struct MpecoptOptions{T}
    lpcc_solver_opts::Union{MilpSolverOptions{T}, MadNLPCOptions{T}, Nothing}
    bnlp_opts::Dict = Dict(
        :barrier=>MadNLP.MonotoneUpdate(mu_init=1e-3),
        :bound_push=>1e-6,
        :bound_fac=>1e-6,
        :print_level=>print_level,
        :bound_relax_factor=>0.0,
    )
    s_stationarity_tol::T = 1e-8
    b_stationarity_tol::T = 1e-7
    tr0::T = 1e-3
    alpha_tr::T = 1e-1
    tr_min::T = 1e-7
    max_iter::Int = 10
end

struct Mpecopt{T,VT}
    opts::MpecoptOptions{T}
    mpcc::AbstractMPCCModel{T, VT}
    lpcc::AbstractLPCCModel{T, VT}
    lpcc_solver::Union{MadNLPCSolver{T, VT}, MilpSolver{T, VT}}
    bnlp::BranchNLP{T,VT}
    ipm::MadNLP.MadNLPSolver{T, VT}

    status::MpecoptStatus
    x::VT
    b::Vector{Bool}
end
