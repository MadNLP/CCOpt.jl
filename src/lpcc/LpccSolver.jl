abstract type AbstractLpccSolver{T} end

# TODO(@anton) Assumes Vertical
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

struct LpccMILP{T, VT, MT}
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

    function LpccMILP(mpcc::AbstractMPCCModel{T, VT}; M=100.0) where {T, VT}
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

        return new{T, VT, typeof(A)}(
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

function linearize!(lpcc::LpccMILP, x::AbstractVector; tr=1e-1)
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
    lpcc.lbx[(mpcc.meta.nvar+1):end] .= 0.0
    lpcc.ubx[(mpcc.meta.nvar+1):end] .= 1.0

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

function solve_with_highs(lpcc::LpccMILP)
    # TODO(@anton) this is inefficient as we build the HiGHS solver each time,
    # However, the HiGHs interface doesn't allow for efficiently updating the A matrix
    # via just passing e.g. a new values array.

    highs = Highs_create()
    # Set options
    #Highs_setStringOptionValue(highs, "presolve", "off")
    # Add variables
    Highs_addVars(highs, length(lpcc.lbx), lpcc.lbx, lpcc.ubx)
    # TODO(@anton) no need for comprehensions here, (or actually storing A.
    #              we could just store colptr, rowval, and nzval, and modify them in linearze!
    # Add constraints
    println([i-1 for i in lpcc.A.colptr[1:lpcc.A.n]])
    println([i-1 for i in lpcc.A.rowval])

    Highs_addRows(
        highs,
        lpcc.A.m,
        lpcc.lba,
        lpcc.uba,
        length(lpcc.A.nzval),
        [i-one(i) for i in lpcc.csrrowptr[1:lpcc.A.n]],
        [i-one(i) for i in lpcc.csrcolval],
        lpcc.csrnzval,
    )
    # Add objective
    Highs_changeColsCostByRange(highs, 0, lpcc.A.n-1, lpcc.c)
    # Set integrality
    Highs_changeColsIntegralityByRange(highs, 0, lpcc.A.n-1, lpcc.integrality)

    Highs_run(highs)
    Highs_destroy(highs)

    return highs
end
