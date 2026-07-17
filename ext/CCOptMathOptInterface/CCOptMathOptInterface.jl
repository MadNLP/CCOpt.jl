################################################################################
# This code is adapted from NLPModelsJuMP.
#
# Original source-code:
# https://github.com/JuliaSmoothOptimizers/NLPModelsJuMP.jl/blob/main/src/MOI_wrapper.jl
#
# Copyright (c) 2018-2019: Abel Soares Siqueira and Dominique Orban
# NLPModelsJuMP.jl is licensed under the [MPL version 2.0](https://www.mozilla.org/MPL/2.0/).
################################################################################
module CCOptMathOptInterface

using MathOptComplements
using CCOpt
using MadNLP
using NLPModelsJuMP

const MOI = MathOptComplements.MOI

function __init__()
    setglobal!(CCOpt, :Optimizer, Optimizer)
    return
end

const _SETS = Union{
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
    MOI.EqualTo{Float64},
    MOI.Interval{Float64},
}

const _CC_SETS = MathOptComplements.ComplementsWithSetType{MOI.Nonnegatives}

mutable struct Optimizer <: MOI.AbstractOptimizer
    options::Dict{String, Any}
    solver
    mpcc::Union{Nothing, CCOpt.MPCCModel}
    stats::Union{
        Nothing,
        CCOpt.CCOptExecutionStats{Float64, Vector{Float64}},
    }
    function Optimizer()
        return new(Dict{String, Any}(), nothing, nothing, nothing)
    end
end

MOI.get(::Optimizer, ::MOI.SolverName) = "CCOpt"

MOI.is_empty(optimizer::Optimizer) = isnothing(optimizer.solver) && isnothing(optimizer.mpcc)

function MOI.empty!(optimizer::Optimizer)
    optimizer.solver = nothing
    optimizer.mpcc = nothing
    optimizer.stats = nothing
    return
end

###
### MOI.RawOptimizerAttribute
###

function MOI.set(optimizer::Optimizer, param::MOI.RawOptimizerAttribute, value)
    return optimizer.options[param.name] = value
end

function MOI.get(optimizer::Optimizer, param::MOI.RawOptimizerAttribute)
    return optimizer.options[param.name]
end

###
### MOI.Silent
###

MOI.supports(::Optimizer, ::MOI.Silent) = true

# TODO
function MOI.set(optimizer::Optimizer, ::MOI.Silent, value::Bool)
    optimizer.silent = value
    return
end

MOI.get(optimizer::Optimizer, ::MOI.Silent) = optimizer.silent

###
### MOI.AbstractModelAttribute
###

function MOI.supports(
    ::Optimizer,
    ::Union{
        MOI.ObjectiveSense,
        MOI.ObjectiveFunction{
            <:Union{
                MOI.VariableIndex,
                MOI.ScalarAffineFunction{Float64},
                MOI.ScalarQuadraticFunction{Float64},
                MOI.ScalarNonlinearFunction,
            }
        },
        MOI.NLPBlock,
        MOI.UserDefinedFunction,
    },
)
    return true
end

###
### MOI.AbstractVariableAttribute
###

function MOI.supports(::Optimizer, ::MOI.VariablePrimalStart, ::Type{MOI.VariableIndex})
    return true
end

###
### `supports_constraint`
###

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{
        <:Union{
            MOI.VariableIndex,
            MOI.ScalarAffineFunction{Float64},
            MOI.ScalarQuadraticFunction{Float64},
            MOI.ScalarNonlinearFunction,
        },
    },
    ::Type{<:_SETS},
)
    return true
end

# Support for complementarity constraint
# The problem should be reformulated before using the MathOptComplements's bridges
function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VectorOfVariables},
    ::Type{_CC_SETS},
)
    return true
end

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    MOI.empty!(dest)
    options = Dict{Symbol, Any}(
        Symbol(key) => dest.options[key] for key in keys(dest.options)
    )

    contypes = MOI.get(src, MOI.ListOfConstraintTypesPresent())
    cc_cons = MOI.get(src, MOI.ListOfConstraintIndices{MOI.VectorOfVariables, _CC_SETS}())
    if length(cc_cons) == 0
        error("The model does not have any complementarity constraints. Please switch to an appropriate solver.")
    end
    ind_cc1, ind_cc2 = MOI.VariableIndex[], MOI.VariableIndex[]

    for cidx in cc_cons
        fun = MOI.get(src, MOI.ConstraintFunction(), cidx)
        set = MOI.get(src, MOI.ConstraintSet(), cidx)
        n_comp = div(set.dimension, 2)
        for cc in 1:n_comp
            push!(ind_cc1, fun.variables[cc])
            push!(ind_cc2, fun.variables[cc + n_comp])
        end

        MOI.delete(src, cidx)
    end

    nlp, index_map = NLPModelsJuMP.nlp_model(src)

    # TODO: add complementarity back

    ind_x1 = getfield.(ind_cc1, :value)
    ind_x2 = getfield.(ind_cc2, :value)

    dest.mpcc = CCOpt.MPCCModelVarVar(nlp, ind_x1, ind_x2)
    dest.solver = CCOpt.RelaxationSolver(dest.mpcc; options...)

    return index_map
end

function MOI.optimize!(model::Optimizer)
    model.stats = CCOpt.solve_homotopy!(model.solver)
    return
end

function MOI.get(optimizer::Optimizer, ::MOI.SolveTimeSec)
    cnt = optimizer.stats.counters
    return cnt.total_time
end

function MOI.get(optimizer::Optimizer, ::MOI.RawStatusString)
    return optimizer.stats.status
end

struct RawStatus <: MOI.AbstractModelAttribute
    name::Symbol
end

MOI.is_set_by_optimize(::RawStatus) = true

function MOI.get(optimizer::Optimizer, attr::RawStatus)
    return getfield(optimizer.stats, attr.name)
end

const TERMINATION_STATUS = Dict{MadNLP.Status,MOI.TerminationStatusCode}(
    MadNLP.SOLVE_SUCCEEDED => MOI.LOCALLY_SOLVED,
    MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL => MOI.ALMOST_LOCALLY_SOLVED,
    MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL => MOI.SLOW_PROGRESS,
    MadNLP.DIVERGING_ITERATES => MOI.INFEASIBLE_OR_UNBOUNDED,
    MadNLP.INFEASIBLE_PROBLEM_DETECTED => MOI.LOCALLY_INFEASIBLE,
    MadNLP.MAXIMUM_ITERATIONS_EXCEEDED => MOI.ITERATION_LIMIT,
    MadNLP.MAXIMUM_WALLTIME_EXCEEDED => MOI.TIME_LIMIT,
    MadNLP.INITIAL => MOI.OPTIMIZE_NOT_CALLED,
    MadNLP.RESTORATION_FAILED => MOI.NUMERICAL_ERROR,
    MadNLP.INVALID_NUMBER_DETECTED => MOI.INVALID_MODEL,
    MadNLP.ERROR_IN_STEP_COMPUTATION => MOI.NUMERICAL_ERROR,
    MadNLP.NOT_ENOUGH_DEGREES_OF_FREEDOM => MOI.INVALID_MODEL,
    MadNLP.USER_REQUESTED_STOP => MOI.INTERRUPTED,
    MadNLP.INTERNAL_ERROR => MOI.OTHER_ERROR,
    MadNLP.INVALID_NUMBER_OBJECTIVE => MOI.INVALID_MODEL,
    MadNLP.INVALID_NUMBER_GRADIENT => MOI.INVALID_MODEL,
    MadNLP.INVALID_NUMBER_CONSTRAINTS => MOI.INVALID_MODEL,
    MadNLP.INVALID_NUMBER_JACOBIAN => MOI.INVALID_MODEL,
    MadNLP.INVALID_NUMBER_HESSIAN_LAGRANGIAN => MOI.INVALID_MODEL,
)

function MOI.get(optimizer::Optimizer, ::MOI.TerminationStatus)
    if isnothing(optimizer.stats)
        return MOI.OPTIMIZE_NOT_CALLED
    end
    return TERMINATION_STATUS[optimizer.stats.status]
end

function MOI.get(optimizer::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.stats.objective
end

function MOI.get(optimizer::Optimizer, attr::MOI.PrimalStatus)
    if attr.result_index > MOI.get(optimizer, MOI.ResultCount())
        return MOI.NO_SOLUTION
    elseif MOI.get(optimizer, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
        return MOI.FEASIBLE_POINT
    else
        # TODO
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

function MOI.get(::Optimizer, ::MOI.DualStatus)
    # TODO
    return MOI.NO_SOLUTION
end

function MOI.get(optimizer::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.stats.solution[vi.value]
end

MOI.get(optimizer::Optimizer, ::MOI.ResultCount) = 1

end
