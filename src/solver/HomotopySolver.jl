# TODO(@anton) remove this
using MadNLP, NLPModelsIpopt
######################### Types #########################
mutable struct HomotopySolverStats{T, VT}
    # TODO(@anton) what needs to live here
    # TODO(@anton) Should subclass AbstractExecutionStats probably
    status::MPCCSolverStatus # Return status from the HomotopySolver
    solution::VT             # solution for primal variables x
    objective::T             # objective achieved
    multipliers::VT          # multipliers for nonlinear constraints (including relaxed complementarities)
    inf_cc::T                # Complementarity infeasibility calculated as max(x₁⊙x₂)

    wall_time::Float64          # Total wall time take by solver
    eval_function_time::Float64 # Total time spent in function eval (only for MadNLP)
    linear_solver_time::Float64 # Total time spent in linear solver (only for MadNLP)
    iter::Int                   # Total number of nlp solver iterations

    nlp_stats::Vector{Any}      # Vector of nlp solver stats.
end

function HomotopySolverStats(mpcc::AbstractMPCCModel{T, VT}) where {T, VT}
    return HomotopySolverStats(
        UNKNOWN,
        VT(undef, mpcc.meta.nvar),
        zero(T),
        VT(undef, mpcc.meta.ncon),
        zero(T),
        0.0,
        0.0,
        0.0,
        0,
        [],
    )
end

@kwdef mutable struct HomotopySolverOptions{T}
    𝜎₀::T = 1.0 # Initial value of complementarity relaxation
    𝛼::T = 0.1  # Linear component of relaxation tightening law
    𝛽::T = 1.0  # Superlinear component of relaxation tightening law

    comp_tol::T = 1e-8 # Target complementarity tolerence in terms of max(x₁⊙x₂)

    N_homotopy::Int = 10 # Max Number of outer iterations

    # TODO(@anton) Do we want an Int here? A symbol (e.g. `:info`, `:warn`, `:debug`, etc.)?
    print_level::Int = 0

    # Decreased bound push for followup iterations
    warm_start_bound_push::Union{Float64, Nothing} = nothing

    nlp_solver_options::Dict{Symbol, Any} = Dict(
        :print_level=>0,
        :sb=>"yes",
        :mu_strategy=>"adaptive",
        :mu_oracle=>"quality-function",
        :bound_relax_factor=>0.0,
    )
end

mutable struct HomotopySolver{M, S, T, VT} <: AbstractMPCCSolver{M, S, T, VT}
    mpcc::M
    nlp::ScholtesRelaxation{T, VT}
    solver::S

    opts::HomotopySolverOptions
    stats::HomotopySolverStats

    start_time::Float64

    x_k::VT
    y_k::VT
    f_k::T
    inf_cc::T

    𝜎::T
end

function HomotopySolver(mpcc::AbstractMPCCModel, S::Type, opts::HomotopySolverOptions)
    nlp = ScholtesRelaxation(mpcc)

    solver = S(nlp)

    stats = HomotopySolverStats(mpcc)

    x_k = nlp.meta.x0
    y_k = nlp.meta.y0

    𝜎 = opts.𝜎₀

    return HomotopySolver(mpcc, nlp, solver, opts, stats, 0.0, x_k, y_k, 0.0, 0.0, 𝜎)
end

######################### Helpers #########################
function solve_rnlp(
    solver::HomotopySolver{M, S, T, VT},
    n::Int,
) where {M, S <: NLPModelsIpopt.IpoptSolver, T, VT}
    if n > 1
        return SolverCore.solve!(
            solver.solver,
            solver.nlp;
            warm_start_init_point="yes",
            warm_start_bound_push=something(
                solver.opts.warm_start_bound_push,
                get(solver.opts.nlp_solver_options, :warm_start_bound_push, nothing),
                0.001,
            ),
            solver.opts.nlp_solver_options...,
        )
        # Additional tuning
        # TODO(@anton) do more testing
        #mu_init=1e-4,
        #warm_start_bound_push=1e-8,
    else
        return SolverCore.solve!(
            solver.solver,
            solver.nlp;
            solver.opts.nlp_solver_options...,
        )
    end
end

function solve_rnlp(
    solver::HomotopySolver{M, S, T, VT},
    n::Int,
) where {M, S <: MadNLP.AbstractMadNLPSolver, T, VT}
    if n > 1
        return SolverCore.solve!(
            solver.solver;
            bound_push=something(
                solver.opts.warm_start_bound_push,
                get(solver.opts.nlp_solver_options, :bound_push, nothing),
                1e-2,
            ),
            solver.opts.nlp_solver_options...,
        )
        # Additional tuning
        # TODO(@anton) do more testing
        # mu_init=1e-4,
        # bound_push=1e-8,
    else
        return SolverCore.solve!(solver.solver; solver.opts.nlp_solver_options...)
    end
end

function set_silent!(
    solver::HomotopySolver{M, S, T, VT},
) where {M, S <: NLPModelsIpopt.IpoptSolver, T, VT}
    return solver.opts.nlp_solver_options[:print_level] = 0
end

function set_silent!(
    solver::HomotopySolver{M, S, T, VT},
) where {M, S <: MadNLP.AbstractMadNLPSolver, T, VT}
    return solver.opts.nlp_solver_options[:print_level] = MadNLP.ERROR
end

function nlp_solve_acceptable(nlp_stats::AbstractExecutionStats)
    return nlp_stats.status ∈ [:first_order, :acceptable, :small_step]
end

function nlp_solve_acceptable(nlp_stats::MadNLP.MadNLPExecutionStats)
    return nlp_stats.status ∈ [
        MadNLP.SOLVE_SUCCEEDED,
        MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL,
        MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL,
    ]
end

function reset_nlp_solver!(
    solver::HomotopySolver{M, S, T, VT},
) where {M, S <: NLPModelsIpopt.IpoptSolver, T, VT}
    # Do nothing
end

function reset_nlp_solver!(
    solver::HomotopySolver{M, S, T, VT},
) where {M, S <: MadNLP.AbstractMadNLPSolver, T, VT}
    solver.solver.cnt.k = 0
    solver.solver.cnt.l = 0
    solver.solver.cnt.t = 0
    solver.solver.cnt.acceptable_cnt = 0
    solver.solver.cnt.unsuccessful_iterate = 0
    return solver.solver.cnt.restoration_fail_count = 0
end

function update_times!(
    solver::HomotopySolver{M, S, T, VT},
    nlp_stats::AbstractExecutionStats,
) where {M, S <: NLPModelsIpopt.IpoptSolver, T, VT}
    # TODO(@anton) ipopt interface doesn't give detailed times
    return solver.stats.iter += nlp_stats.iter
end

function update_times!(
    solver::HomotopySolver{M, S, T, VT},
    nlp_stats::MadNLP.MadNLPExecutionStats,
) where {M, S <: MadNLP.AbstractMadNLPSolver, T, VT}
    solver.stats.eval_function_time = nlp_stats.counters.eval_function_time
    solver.stats.linear_solver_time = nlp_stats.counters.linear_solver_time
    return solver.stats.iter += nlp_stats.iter
end

######################### Main loop #########################
function solve!(
    solver::HomotopySolver{M, T, VT};
    x=nothing,
    y=nothing,
    kwargs...,
) where {M, T, VT}
    return solve!(solver.mpcc, solver, solver.stats; x=x, y=y, kwargs...)
end

function solve!(
    mpcc::AbstractMPCCModel,
    solver::HomotopySolver,
    stats::HomotopySolverStats;
    x=nothing,
    y=nothing,
    kwargs...,
)
    solver.start_time = time()

    if x ≠ nothing
        solver.x_k .= x
    end
    if y ≠ nothing
        solver.y_k .= y
    end

    opts = solver.opts

    # TODO(@anton) Long term we may want a pre-process step for the options
    if iszero(opts.print_level)
        set_silent!(solver)
    end

    converged = false
    timeout = false
    solver.nlp.𝜎[] = solver.𝜎
    ii = 1
    while ii ≤ opts.N_homotopy
        nlp_stats = solve_rnlp(solver, ii)
        push!(stats.nlp_stats, nlp_stats)
        solver.inf_cc = comp_residual_product(mpcc, nlp_stats.solution)
        solver.x_k = nlp_stats.solution
        solver.y_k = nlp_stats.multipliers
        solver.f_k = nlp_stats.objective

        copyto!(NLPModels.get_x0(solver.nlp), solver.x_k)
        copyto!(NLPModels.get_y0(solver.nlp), solver.y_k)
        update_times!(solver, nlp_stats)

        if nlp_solve_acceptable(nlp_stats) && solver.inf_cc ≤ solver.opts.comp_tol
            converged = true
            break
        end

        # 𝛽 > 1 decreases 𝜎 superlinearly when close to convergence
        solver.𝜎 = min(opts.𝛼*solver.𝜎, solver.𝜎^opts.𝛽)
        solver.nlp.𝜎[] = solver.𝜎
        ii += 1
        reset_nlp_solver!(solver)
    end
    if converged
        stats.status = NLP_STATIONARY
        stats.solution = solver.x_k
        stats.multipliers = solver.y_k[1:mpcc.meta.ncon] # Unreliable
        stats.inf_cc = solver.inf_cc
        stats.objective = solver.f_k
    else
        stats.status = MAXIMUM_ITERATIONS_EXCEEDED
        stats.solution = solver.x_k
        stats.multipliers = solver.y_k[1:mpcc.meta.ncon] # Unreliable
        stats.inf_cc = solver.inf_cc
        stats.objective = solver.f_k
    end
    stats.wall_time = time() - solver.start_time;
    return stats
end
