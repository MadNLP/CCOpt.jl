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

struct LpccMILP{T, VT, MT, ST}
    mpcc::AbstractMPCCModel{T, VT}

    # Buffers for COO representation
    arows::Vector{Int32}
    acols::Vector{Int32}
    avals::VT

    # Internal vectors for sparse!
    klasttouch::Vector{Int32}
    csrrowptr::Vector{Int32}
    csrcolval::Vector{Int32}
    csrnzval::VT

    # Problem data
    A::MT
    c::VT
    lba::VT
    uba::VT
    lbx::VT
    ubx::VT
    integrality::Vector{Int32} # kContinuous = 0, kInteger = 1

    M::T

    function LpccMILP(
        mpcc::AbstractMPCCModel{T, VT};
        M=100.0,
        solver=Gurobi.Optimizer,
    ) where {T, VT}
        # TODO(@anton) we assume vertical form
        if !is_vertical(mpcc)
            error("Linearization to LpccMILP currently expects a vertical form MPCC")
        end
        arows::Vector{Int32}, acols::Vector{Int32} = jac_structure(mpcc)
        avals = jac_coord(mpcc, mpcc.meta.x0)
        nvar = mpcc.meta.nvar
        ncon = mpcc.meta.ncon
        ncc = mpcc.meta.ncc
        # Build remaining COO representation
        for ii in 1:ncc
            # Ms-x_1 > -lbx_2
            push!(arows, ncon + ii)
            push!(arows, ncon + ii)
            push!(acols, mpcc.meta.ind_cc1[ii])
            push!(acols, nvar + ii)
            push!(avals, -1.0)
            push!(avals, M) # M
            # -Ms-x_2 > -M - lbx_2
            push!(arows, ncon + ncc + ii)
            push!(arows, ncon + ncc + ii)
            push!(acols, mpcc.meta.ind_cc2[ii])
            push!(acols, nvar + ii)
            push!(avals, -1.0)
            push!(avals, -M) # M
        end

        # Populate sparse matrix and intermediate vectors
        m::Int32 = ncon+2*ncc
        n::Int32 = nvar+ncc
        A = sparse(arows, acols, avals, m, n)
        klasttouch = Vector{Int32}(undef, n)
        csrrowptr = Vector{Int32}(undef, m + 1)
        csrcolptr = Vector{Int32}(undef, length(arows))
        csrnzval = VT(undef, length(arows))

        c = VT(undef, n)
        @views grad!(mpcc, mpcc.meta.x0, c[1:nvar])
        c[(nvar+1):end] .= 0.0
        lba = VT(undef, m)
        uba = VT(undef, m)
        lbx = VT(undef, n)
        ubx = VT(undef, n)
        integrality = Vector{Int32}(undef, nvar+ncc)

        integrality[1:nvar] .= 0
        integrality[(nvar+1):(nvar+ncc)] .= 1

        return new{T, VT, typeof(A), solver}(
            mpcc,
            arows,
            acols,
            avals,
            klasttouch,
            csrrowptr,
            csrcolptr,
            csrnzval,
            A,
            c,
            lba,
            uba,
            lbx,
            ubx,
            integrality,
            M,
        )
    end
end

@kwdef struct LpccMILPOptions{T} <: AbstractLpccSolverOptions{T}
    verbose::Bool = true
    abs_gap::T = 1e-9
    rel_gap::T = 1e-4
    opt_tol::T = 1e-8
    feas_tol::T = 1e-8
    int_feas_tol::T = 1e-8
end

function linearize!(lpcc::LpccMILP, x::AbstractVector; tr=1e-1, presolve_binaries=true)
    mpcc = lpcc.mpcc
    nvar = mpcc.meta.nvar
    ncon = mpcc.meta.ncon
    ncc = mpcc.meta.ncc
    @views begin
        # Update gradient
        grad!(mpcc, x, lpcc.c[1:nvar])
        # Update COO:
        jac_coord!(mpcc, x, lpcc.avals[1:mpcc.meta.nnzj])
    end
    # Rebuild Sparse Matrix:
    # TODO(@anton) is there a better way of doing this through the HIGHS interface itself?
    # TODO(@anton) We actually only need the csr in our case.
    m::Int32 = ncon+2*ncc
    n::Int32 = nvar+ncc
    SparseArrays.sparse!(
        lpcc.arows,
        lpcc.acols,
        lpcc.avals,
        m,
        n,
        +,
        lpcc.klasttouch,
        lpcc.csrrowptr,
        lpcc.csrcolval,
        lpcc.csrnzval,
        lpcc.A.colptr,
        lpcc.A.rowval,
        lpcc.A.nzval,
    )

    # Apply trust region
    ind_x = mpcc.meta.ind_x
    ind_cc1 = mpcc.meta.ind_cc1
    ind_cc2 = mpcc.meta.ind_cc2
    @views begin
        # Lower bound on x0
        lpcc.lbx[ind_x] .= max.(mpcc.meta.lvar[ind_x] .- x[ind_x], -tr)
        # Lower bound on x1 and x2
        lpcc.lbx[ind_cc1] .= max.(mpcc.meta.lvar[ind_cc1] .- x[ind_cc1], -tr)
        lpcc.lbx[ind_cc2] .= max.(mpcc.meta.lvar[ind_cc2] .- x[ind_cc2], -tr)
        # upper bound on all
        lpcc.ubx[1:mpcc.meta.nvar] .= min.(mpcc.meta.uvar .- x, tr)
    end
    # Preprocess the binaries based on the trust region
    if presolve_binaries
        a = 0
        for ii in 1:ncc
            if x[ind_cc1[ii]] - tr > mpcc.meta.lvar[ind_cc1[ii]]
                lpcc.lbx[mpcc.meta.nvar+ii] = 1.0
                lpcc.ubx[mpcc.meta.nvar+ii] = 1.0
            elseif x[ind_cc2[ii]] - tr > mpcc.meta.lvar[ind_cc2[ii]]
                lpcc.lbx[mpcc.meta.nvar+ii] = 0.0
                lpcc.ubx[mpcc.meta.nvar+ii] = 0.0
            else
                lpcc.lbx[mpcc.meta.nvar+ii] = 0.0
                lpcc.ubx[mpcc.meta.nvar+ii] = 1.0
                a += 1
            end
        end
    else
        lpcc.lbx[(mpcc.meta.nvar+1):end] .= 0.0
        lpcc.ubx[(mpcc.meta.nvar+1):end] .= 1.0
    end
    # Calculate linearization bounds
    # TODO(@anton) we don't need to re-eval this actually. Fix this
    @views begin
        cons!(lpcc.mpcc, x, lpcc.lba[1:ncon])
        lpcc.uba[1:ncon] .= lpcc.mpcc.meta.ucon .- lpcc.lba[1:ncon]
        lpcc.lba[1:ncon] .= lpcc.mpcc.meta.lcon .- lpcc.lba[1:ncon]
    end
    lpcc.uba[(ncon+1):end] .= Inf
    lpcc.lba[(ncon+1):(ncon+ncc)] = x[mpcc.meta.ind_cc1] .- mpcc.meta.lvar[ind_cc1]
    lpcc.lba[(ncon+ncc+1):end] = x[mpcc.meta.ind_cc2] .- lpcc.M .- mpcc.meta.lvar[ind_cc2]

    return lpcc
end

function tr!(
    lpcc::LpccMILP{T, VT},
    x::AbstractVector{T},
    tr::T;
    presolve_binaries=true,
) where {T, VT}
    mpcc = lpcc.mpcc
    ind_x = mpcc.meta.ind_x
    ind_cc1 = mpcc.meta.ind_cc1
    ind_cc2 = mpcc.meta.ind_cc2
    nvar = mpcc.meta.nvar
    ncon = mpcc.meta.ncon
    ncc = mpcc.meta.ncc
    @views begin
        # Lower bound on x0
        lpcc.lbx[ind_x] .= max.(mpcc.meta.lvar[ind_x] .- x[ind_x], -tr)
        # Lower bound on x1 and x2
        lpcc.lbx[ind_cc1] .= max.(mpcc.meta.lvar[ind_cc1] .- x[ind_cc1], -tr)
        lpcc.lbx[ind_cc2] .= max.(mpcc.meta.lvar[ind_cc2] .- x[ind_cc2], -tr)
        # upper bound on all
        lpcc.ubx[1:mpcc.meta.nvar] .= min.(mpcc.meta.uvar .- x, tr)
    end
    # Preprocess the binaries based on the trust region
    if presolve_binaries
        for ii in 1:ncc
            if x[ind_cc1[ii]] - tr > mpcc.meta.lvar[ind_cc1[ii]]
                lpcc.lbx[mpcc.meta.nvar+ii] = 1.0
                lpcc.ubx[mpcc.meta.nvar+ii] = 1.0
            elseif x[ind_cc2[ii]] - tr > mpcc.meta.lvar[ind_cc2[ii]]
                lpcc.lbx[mpcc.meta.nvar+ii] = 0.0
                lpcc.ubx[mpcc.meta.nvar+ii] = 0.0
            else
                lpcc.lbx[mpcc.meta.nvar+ii] = 0.0
                lpcc.ubx[mpcc.meta.nvar+ii] = 1.0
            end
        end
    else
        lpcc.lbx[(mpcc.meta.nvar+1):end] .= 0.0
        lpcc.ubx[(mpcc.meta.nvar+1):end] .= 1.0
    end
end

function set_opts!(
    model::Model,
    lpcc::LpccMILP{T, VT, MT, Gurobi.Optimizer},
    opts::LpccMILPOptions{T},
) where {T, VT, MT}
    MOI.set(model, MOI.Silent(), !opts.verbose)
    JuMP.set_optimizer_attribute(model, "FeasibilityTol", opts.feas_tol)
    JuMP.set_optimizer_attribute(model, "IntFeasTol", opts.int_feas_tol)
    JuMP.set_optimizer_attribute(model, "OptimalityTol", opts.opt_tol)
    JuMP.set_optimizer_attribute(model, "MIPGap", opts.rel_gap)
    JuMP.set_optimizer_attribute(model, "MIPFocus", 1)
    JuMP.set_optimizer_attribute(model, "Presolve", 1)
    return JuMP.set_optimizer_attribute(model, "MIPGapAbs", opts.abs_gap)
end

function set_opts!(
    model::Model,
    lpcc::LpccMILP{T, VT, MT, HiGHS.Optimizer},
    opts::LpccMILPOptions{T},
) where {T, VT, MT}
    MOI.set(model, MOI.Silent(), !opts.verbose)
    JuMP.set_optimizer_attribute(model, "kkt_tolerance", opts.feas_tol)
    JuMP.set_optimizer_attribute(model, "mip_feasiblility_tolerance", opts.int_feas_tol)
    JuMP.set_optimizer_attribute(model, "optimality_tolerance", opts.opt_tol)
    JuMP.set_optimizer_attribute(model, "mip_rel_gap", opts.rel_gap)
    return JuMP.set_optimizer_attribute(model, "mip_abs_gap", opts.abs_gap)
end

function build(
    lpcc::LpccMILP{T, VT, MT, ST},
    opts::LpccMILPOptions{T};
    x0=nothing,
) where {T, VT, MT, ST}
    model = Model(ST)
    set_opts!(model, lpcc, opts)

    @variable(model, lpcc.lbx[i] <= x[i=1:length(lpcc.lbx)] <= lpcc.ubx[i])
    @objective(model, lpcc.mpcc.meta.minimize ? MIN_SENSE : MAX_SENSE, sum(lpcc.c .* x))
    @constraint(model, lpcc.lba .<= lpcc.A * x .<= lpcc.uba)
    for ii in 1:length(lpcc.integrality)
        lpcc.integrality[ii] == one(Int32) && JuMP.set_binary(x[ii])
    end

    if !isnothing(x0)
        MOI.set.(model, MOI.VariablePrimalStart(), x, x0)
    end
    return model
end

function solve_highs(lpcc::LpccMILP)
    # TODO(@anton) this is inefficient as we build the HiGHS solver each time,
    # However, the HiGHs interface doesn't allow for efficiently updating the A matrix
    # via just passing e.g. a new values array.

    highs = Highs_create()
    # Set options
    Highs_setBoolOptionValue(highs, "log_to_console", true)
    Highs_setDoubleOptionValue(highs, "kkt_tolerance", 1e-7)
    Highs_setDoubleOptionValue(highs, "mip_feasibility_tolerance", 1e-7)
    Highs_setDoubleOptionValue(highs, "mip_rel_gap", 1e-6)
    # Add variables
    Highs_addVars(highs, length(lpcc.lbx), lpcc.lbx, lpcc.ubx)
    # TODO(@anton) no need for comprehensions here, (or actually storing A.
    #              we could just store colptr, rowval, and nzval, and modify them in linearze!
    # Add constraints
    Highs_addRows(
        highs,
        lpcc.A.m,
        lpcc.lba,
        lpcc.uba,
        length(lpcc.A.nzval),
        [i-one(i) for i in lpcc.csrrowptr[1:lpcc.A.m]],
        [i-one(i) for i in lpcc.csrcolval],
        lpcc.csrnzval,
    )
    # Add objective and set sense
    Highs_changeColsCostByRange(highs, zero(Int32), lpcc.A.n-1, lpcc.c)
    Highs_changeObjectiveSense(highs, lpcc.mpcc.meta.minimize ? one(Int32) : -one(Int32))
    # Set integrality
    Highs_changeColsIntegralityByRange(highs, zero(Int32), lpcc.A.n-1, lpcc.integrality)

    # Solve the LPCC
    Highs_run(highs)

    # create return values
    optimal = Highs_getModelStatus(highs) == HiGHS.kHighsModelStatusOptimal

    vals = Vector{Cdouble}(undef, lpcc.A.n)
    Highs_getSolution(highs, vals, C_NULL, C_NULL, C_NULL)

    y = vals[lpcc.integrality .== one(Int32)] .> 0.5

    obj = Highs_getObjectiveValue(highs)

    Highs_destroy(highs)

    return optimal, vals, y, obj
end
