# TODO(@anton) remove this
using MadNLP
######################### Types #########################
mutable struct HomotopySolverStats{T, VT}
    # TODO(@anton) what needs to live here
    # TODO(@anton) Should subclass AbstractExecutionStats probably
    # TODO(@anton) Should have a dictionary of statuses
    status::MPCCSolverStatus
    solution::VT
    objective::T
    multipliers::VT
    inf_cc::T

    wall_time::Float64
    nlp_stats::Vector{Any}
end

function HomotopySolverStats(mpcc::AbstractMPCCModel{T, VT}) where {T, VT}
    return HomotopySolverStats(
        UNKNOWN,
        VT(undef, mpcc.meta.nvar),
        zero(T),
        VT(undef, mpcc.meta.ncon),
        zero(T),
        0.0,
        [],
    )
end

@kwdef mutable struct HomotopySolverOptions{T}
    𝜎₀::T = 1.0
    𝛼::T = 0.1
    𝛽::T = 1.0

    comp_tol::T = 1e-8

    N_homotopy::Int = 9

    # TODO(@anton) Do we want an Int here? A symbol (e.g. `:info`, `:warn`, `:debug`, etc.)?
    print_level::Int = 0

    nlp_solver_options::Dict{Symbol, Any} = Dict(
        :print_level=>0,
        :sb=>"yes",
        :mu_strategy=>"adaptive",
        :mu_oracle=>"quality-function",
        :bound_relax_factor=>0.0,
    )
end

# TODO(@anton) mutable?
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
    if iszero(opts.print_level) && haskey(opts.nlp_solver_options, :print_level)
        opts.nlp_solver_options[:print_level] = 0
    end

    converged = false
    timeout = false
    solver.nlp.𝜎[] = solver.𝜎
    ii = 1
    while ii ≤ opts.N_homotopy
        if typeof(solver.solver) <: MadNLP.AbstractMadNLPSolver
            nlp_stats = SolverCore.solve!(
                solver.solver;
                x=solver.x_k,
                y=solver.y_k,
                opts.nlp_solver_options...,
            )
        else
            nlp_stats = SolverCore.solve!(
                solver.solver,
                solver.nlp;
                x=solver.x_k,
                y=solver.y_k,
                opts.nlp_solver_options...,
            )
        end
        push!(stats.nlp_stats, nlp_stats)
        solver.inf_cc = comp_residual_product(mpcc, nlp_stats.solution)
        solver.x_k = nlp_stats.solution
        solver.y_k = nlp_stats.multipliers
        solver.f_k = nlp_stats.objective

        if nlp_stats.status ∈ [:first_order, :acceptable, :small_step] &&
           solver.inf_cc ≤ solver.opts.comp_tol
            converged = true
            break
        end
        solver.𝜎 = min(opts.𝛼*solver.𝜎, solver.𝜎^opts.𝛽)
        solver.nlp.𝜎[] = solver.𝜎
        reset!(solver.solver, solver.nlp)
        ii += 1
    end
    if converged
        stats.status = NLP_STATIONARY
        stats.solution = solver.x_k
        stats.multipliers = solver.y_k[1:mpcc.meta.ncon] # Unreliable
        stats.objective = solver.f_k
    else
        stats.status = MAXIMUM_ITERATIONS_EXCEEDED
        stats.solution = solver.x_k
        stats.multipliers = solver.y_k[1:mpcc.meta.ncon] # Unreliable
        stats.objective = solver.f_k
    end
    stats.wall_time = time() - solver.start_time;
    return stats
end
