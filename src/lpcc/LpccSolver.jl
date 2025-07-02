abstract type AbstractLpccSolver{T} end

# TODO(@anton) Assumes Vertical
struct SparseLpcc{T, VT}
    arows::IndexSet
    acols::IndexSet
    avals::VT
    lba::VT
    uba::VT

    lbx::VT
    ubx::VT

    ind_cc1::IndexSet
    ind_cc2::IndexSet
end

struct LpccMILP{T, VT, MT}
    mpcc::AbstractMPCCModel{T, VT}

    arows::IndexSet
    acols::IndexSet
    avals::VT

    A::MT
    lba::VT
    uba::VT
    lbx::VT
    ubx::VT
    integrality::Vector{Bool}

    function LpccMILP(mpcc::AbstractMPCCModel{T, VT}) where {T, VT}
        # TODO(@anton) we assume vertical form
        if !is_vertical(mpcc)
            error("Linearization to LpccMILP currently expects a vertical form MPCC")
        end
        arows, acols = jac_structure(mpcc)
        avals = jac_coord(mpcc, mpcc.meta.x0)
        nvar = mpcc.meta.nvar
        ncon = mpcc.meta.ncon
        ncc = mpcc.meta.ncc
        # Build remaining
        for ii in 1:ncc
            # Ms-x_1 > -lbx_2
            push!(arows, ncon + ii)
            push!(arows, ncon + ii)
            push!(acols, mpcc.meta.ind_cc1[ii])
            push!(acols, nvar + ii)
            push!(avals, -1.0)
            push!(avals, 100.0) # M
            # -Ms-x_2 > -M - lbx_2
            push!(arows, ncon + ncc + ii)
            push!(arows, ncon + ncc + ii)
            push!(acols, mpcc.meta.ind_cc2[ii])
            push!(acols, nvar + ii)
            push!(avals, -1.0)
            push!(avals, -100.0) # M
        end

        A = sparse(arows, acols, avals)
        lba = VT(undef, ncon+2*ncc)
        uba = VT(undef, ncon+2*ncc)
        lbx = VT(undef, nvar+ncc)
        ubx = VT(undef, nvar+ncc)
        integrality = Vector{Bool}(undef, nvar+ncc)

        integrality[1:nvar] .= false
        integrality[(nvar+1):(nvar+ncc)] .= true

        return new{T, VT, typeof(A)}(
            mpcc,
            arows,
            acols,
            avals,
            A,
            lba,
            uba,
            lbx,
            ubx,
            integrality,
        )
    end
end
