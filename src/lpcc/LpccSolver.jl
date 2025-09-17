abstract type AbstractLpccSolver{T} end
abstract type AbstractLpccSolverOptions{T} end

# TODO(@anton) Assumes vertical form
struct SparseLpcc{T, VT}
    arows::Vector{Int32}
    acols::Vector{Int32}
    avals::VT
    lba::VT
    uba::VT

    lbx::VT
    ubx::VT

    ind_cc1::Vector{Int32}
    ind_cc2::Vector{Int32}
end

# define milp solver for mpccs
@kwdef mutable struct MilpSolverOptions{T} <: AbstractLpccSolverOptions{T}
    verbose::Bool = true
    abs_gap::T = 1e-9
    rel_gap::T = 1e-4
    opt_tol::T = 1e-8
    feas_tol::T = 1e-8
    int_feas_tol::T = 1e-8
    M::T = 1000.0
    solver_type::Type = Gurobi.Optimizer
end

mutable struct MilpSolver{T, VT}
    bigm::BigMModel{T, VT}
    lpcc::AbstractMPCCModel{T, VT}
    opts::MilpSolverOptions{T}
    model::Model

    function MilpSolver(
        lpcc::AbstractMPCCModel{T, VT},
        opts::MilpSolverOptions{T},
    ) where {T, VT}
        model = Model(opts.solver_type)
        nvar = lpcc.meta.nvar
        ncon = lpcc.meta.ncon
        ncc = lpcc.meta.ncc

        bigm = BigMModel(lpcc, opts.M)
        # TODO(@anton) Improvements to QuadraticModels, QuadraticModelsGurobi, and the creation of
        #              QuadraticModelsHighs, would make this indirection through JuMP unnecessary.
        #              This is a decent amount of work though :(
        @variable(
            model,
            bigm.meta.lvar[i] <= x[i=1:length(bigm.meta.lvar)] <= bigm.meta.uvar[i]
        )
        @objective(
            model,
            lpcc.meta.minimize ? MIN_SENSE : MAX_SENSE,
            sum(NLPModels.grad(bigm, bigm.meta.x0) .* x)
        )
        @constraint(
            model,
            bigm.meta.lcon .<= NLPModels.jac(bigm, bigm.meta.x0) * x .<= bigm.meta.ucon
        )
        for ii in (nvar+1):(nvar+ncc)
            JuMP.set_binary(x[ii])
        end

        MOI.set.(model, MOI.VariablePrimalStart(), x[1:nvar], lpcc.meta.x0)
        solver = new{T, VT}(bigm, lpcc, opts, model)
        set_opts!(solver)

        return solver
    end
end

function set_opts!(solver::MilpSolver{T, VT}) where {T, VT}
    model = solver.model
    opts = solver.opts
    if opts.solver_type == Gurobi.Optimizer
        MOI.set(model, MOI.Silent(), !opts.verbose)
        JuMP.set_optimizer_attribute(model, "FeasibilityTol", opts.feas_tol)
        JuMP.set_optimizer_attribute(model, "IntFeasTol", opts.int_feas_tol)
        JuMP.set_optimizer_attribute(model, "OptimalityTol", opts.opt_tol)
        JuMP.set_optimizer_attribute(model, "MIPGap", opts.rel_gap)
        JuMP.set_optimizer_attribute(model, "MIPFocus", 1)
        JuMP.set_optimizer_attribute(model, "Presolve", 1)
        JuMP.set_optimizer_attribute(model, "MIPGapAbs", opts.abs_gap)
    elseif opts.solver_type == HiGHS.Optimizer
        MOI.set(model, MOI.Silent(), !opts.verbose)
        JuMP.set_optimizer_attribute(model, "kkt_tolerance", opts.feas_tol)
        JuMP.set_optimizer_attribute(model, "mip_feasiblility_tolerance", opts.int_feas_tol)
        JuMP.set_optimizer_attribute(model, "optimality_tolerance", opts.opt_tol)
        JuMP.set_optimizer_attribute(model, "mip_rel_gap", opts.rel_gap)
        JuMP.set_optimizer_attribute(model, "mip_abs_gap", opts.abs_gap)
    end
    return nothing
end

function build(solver::MilpSolver{T, VT}; y0=nothing) where {T, VT}
    lpcc = solver.lpcc
    opts = solver.opts

    model = Model(opts.solver_type)
    nvar = lpcc.meta.nvar
    ncon = lpcc.meta.ncon
    ncc = lpcc.meta.ncc

    # TODO(@anton) Improvements to QuadraticModels, QuadraticModelsGurobi, and the creation of
    #              QuadraticModelsHighs, would make this indirection through JuMP unnecessary.
    #              This is a decent amount of work though :(
    @variable(model, lpcc.meta.lbx[i] <= x[i=1:length(lpcc.meta.lbx)] <= lpcc.meta.ubx[i])
    @objective(model, lpcc.meta.minimize ? MIN_SENSE : MAX_SENSE, sum(lpcc.nlp.data.c .* x))
    @constraint(model, lpcc.lba .<= lpcc.A * x .<= lpcc.uba)

    for ii in (nvar+1):(nvar+ncc)
        JuMP.set_binary(x[ii])
    end

    MOI.set.(model, MOI.VariablePrimalStart(), x[1:nvar], lpcc.meta.x0)
    set_opts!(solver)
    solver.model = Model(ST)
    set_opts!(model, lpcc, opts)

    @variable(model, lpcc.lbx[i] <= x[i=1:length(lpcc.lbx)] <= lpcc.ubx[i])
    @objective(model, lpcc.mpcc.meta.minimize ? MIN_SENSE : MAX_SENSE, sum(lpcc.c .* x))
    @constraint(model, lpcc.lba .<= lpcc.A * x .<= lpcc.uba)
    for ii in 1:length(lpcc.integrality)
        lpcc.integrality[ii] == one(Int32) && JuMP.set_binary(x[ii])
    end

    if !isnothing(y0)
        MOI.set.(model, MOI.VariablePrimalStart(), x[], x0)
    end
    return model
end
