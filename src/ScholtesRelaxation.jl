######################### Scholtes Relaxation #########################
struct ScholtesRelaxation{T, VT} <: NLPModels.AbstractNLPModel{T, VT}
    mpcc::AbstractMPCCModel{T, VT}
    meta::NLPModels.NLPModelMeta{T, VT}
    𝜎::Base.RefValue{T}
end

function ScholtesRelaxation(mpcc::AbstractMPCCModel{T, VT}) where {T, VT}
    if !is_vertical(mpcc)
        # TODO(@anton) Perhaps we should do this automatically in the future or we can support non-vertical form scholtes
        #              though this makes the callbacks a bit more complicated
        error(
            "Scholtes Relaxation currently expects a vertical form MPCC, use vertical_form(mpcc) to convert it.",
        )
    end

    # Update only what needs to be updated
    ncon = mpcc.meta.ncon + mpcc.meta.ncc
    lcon = append!(mpcc.meta.lcon, -Inf*ones(mpcc.meta.ncc))
    ucon = append!(mpcc.meta.ucon, zeros(mpcc.meta.ncc))
    y0 = append!(mpcc.meta.y0, zeros(mpcc.meta.ncc))
    # TODO(@anton) this is a lower bound only accurate for vertical form: we can calculate this exactly if we store
    #              the nnz for the jacobians of G, and H, which we now do.
    nnzj = mpcc.meta.nnzj + 2*mpcc.meta.ncc
    nln_nnzj = mpcc.meta.nln_nnzj + 2*mpcc.meta.ncc # All the nln values are

    # TODO(@anton) this is a bug actually. we need to check the structure of the mpcc (and the underlying nlp) to
    #              figure out if the nnzh is correct as if the off diagonals are not already in the nonzeros.
    #
    # TODO(@anton) This may or may not break the assumptions made by show(::NLPModelMeta)
    nnzh = mpcc.meta.nnzh + mpcc.meta.ncc
    # TODO(@anton) We may need to change how nlv(b,o,c) are handled because we actually cannot
    #              backcalculate how these need to change necessarily.
    #              However these seem to not be used anywhere in the NLPModels API so I am ignoring them.

    meta = NLPModels.NLPModelMeta(
        mpcc.nlp.meta,
        ncon=ncon,
        lcon=lcon,
        y0=y0,
        nnzj=nnzj,
        nln_nnzj=nln_nnzj,
        nnzh=nnzh,
    )
    𝜎 = zero(T)
    return ScholtesRelaxation(mpcc, meta, Ref(𝜎))
end

# Counters should be forwarded
function Base.getproperty(rnlp::ScholtesRelaxation, sym::Symbol)
    if sym ∈ [:counters]
        getproperty(rnlp.mpcc.nlp, sym)
    else
        getfield(rnlp, sym)
    end
end

######################### NLPModels Callbacks #########################
NLPModels.obj(rnlp::ScholtesRelaxation, x::AbstractVector) = NLPModels.obj(rnlp.mpcc, x)
function NLPModels.grad!(rnlp::ScholtesRelaxation, x::AbstractVector, gx::AbstractVector)
    return NLPModels.grad!(rnlp.mpcc, x, gx)
end
function NLPModels.objgrad!(rnlp::ScholtesRelaxation, x::AbstractVector, g::AbstractVector)
    return NLPModels.objgrad!(rnlp.mpcc, x, g)
end

function NLPModels.cons_lin!(
    rnlp::ScholtesRelaxation,
    x::AbstractVector,
    cx::AbstractVector,
)
    if get_ncon(rnlp.mpcc.nlp) > 0
        return cons_lin!(rnlp.mpcc, x, cx)
    else
        return cx
    end
end

function NLPModels.cons_nln!(
    rnlp::ScholtesRelaxation,
    x::AbstractVector,
    cx::AbstractVector,
)
    mpcc_nnln = rnlp.mpcc.meta.nnln
    # This if statement is necessary as it seems that without it c!(cx,x) does not exist in a possible underlying ADNLPModel
    if get_ncon(rnlp.mpcc.nlp) > 0
        cons_nln!(rnlp.mpcc, x, view(cx, 1:mpcc_nnln))
    end
    # TODO(@anton) figure out if the intermediate outputs cause allocations
    cx[(mpcc_nnln+1):(rnlp.meta.nnln)] =
        (comp_left(rnlp.mpcc, x) .- lcomp_left(rnlp.mpcc)) .*
        (comp_right(rnlp.mpcc, x) .- lcomp_right(rnlp.mpcc)) .- rnlp.𝜎[]
    return cx
end

function NLPModels.jac_lin_structure!(
    rnlp::ScholtesRelaxation,
    rows::AbstractVector{Int},
    cols::AbstractVector{Int},
)
    jac_lin_structure!(rnlp.mpcc, rows, cols) # get including complementarities
    return rows, cols
end

function NLPModels.jac_nln_structure!(
    rnlp::ScholtesRelaxation,
    rows::AbstractVector{Int},
    cols::AbstractVector{Int},
)
    @views jac_nln_structure!(
        rnlp.mpcc,
        rows[1:rnlp.mpcc.meta.nln_nnzj],
        cols[1:rnlp.mpcc.meta.nln_nnzj],
    ) # get including complementarities

    for i in 1:rnlp.mpcc.meta.ncc
        rows[i+rnlp.mpcc.meta.nln_nnzj] = i + rnlp.mpcc.meta.ncon
        cols[i+rnlp.mpcc.meta.nln_nnzj] = rnlp.mpcc.meta.ind_cc1[i]
    end
    for i in 1:rnlp.mpcc.meta.ncc
        rows[i+rnlp.mpcc.meta.nln_nnzj+rnlp.mpcc.meta.ncc] = i + rnlp.mpcc.meta.ncon
        cols[i+rnlp.mpcc.meta.nln_nnzj+rnlp.mpcc.meta.ncc] = rnlp.mpcc.meta.ind_cc2[i]
    end

    return rows, cols
end

function NLPModels.jac_lin_coord!(
    rnlp::ScholtesRelaxation,
    x::AbstractVector,
    j::AbstractVector,
)
    return jac_lin_coord!(rnlp.mpcc, x, j)
end
function NLPModels.jac_nln_coord!(
    rnlp::ScholtesRelaxation,
    x::AbstractVector,
    j::AbstractVector,
)
    jac_nln_coord!(rnlp.mpcc, x, @view(j[1:rnlp.mpcc.meta.nln_nnzj]))

    comp_res_right!(
        rnlp.mpcc,
        x,
        @view(j[(rnlp.mpcc.meta.nln_nnzj+1):(rnlp.mpcc.meta.nln_nnzj+rnlp.mpcc.meta.ncc)])
    )
    comp_res_left!(
        rnlp.mpcc,
        x,
        @view(
            j[(rnlp.mpcc.meta.nln_nnzj+rnlp.mpcc.meta.ncc+1):(rnlp.mpcc.meta.nln_nnzj+2*rnlp.mpcc.meta.ncc)]
        )
    )
    return j
end

function NLPModels.jprod_lin!(
    rnlp::ScholtesRelaxation,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jv[1:rnlp.meta.nlin] = jac_lin(rnlp, x) * v
    return Jv
end

function NLPModels.jprod_nln!(
    rnlp::ScholtesRelaxation,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jv[1:rnlp.meta.nnln] = jac_nln(rnlp, x) * v
    return Jv
end

function NLPModels.jtprod_lin!(
    rnlp::ScholtesRelaxation,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jtv[1:rnlp.meta.nvar] = jac_lin(rnlp, x)' * v
    return Jtv
end

function NLPModels.jtprod_nln!(
    rnlp::ScholtesRelaxation,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jtv[1:rnlp.meta.nvar] = jac_nln(rnlp, x)' * v
    return Jtv
end

function NLPModels.hess_structure!(
    rnlp::ScholtesRelaxation,
    rows::Vector{Int},
    cols::Vector{Int},
)
    @views hess_structure!(
        rnlp.mpcc,
        rows[1:rnlp.mpcc.meta.nnzh],
        cols[1:rnlp.mpcc.meta.nnzh],
    )
    # TODO(@anton) it seems hard to vectorize in one operation this because there is no efficient unzip in Base:
    #              See https://github.com/JuliaLang/julia/issues/13942 for details
    for i in 1:rnlp.mpcc.meta.ncc
        cols[i+rnlp.mpcc.meta.nnzh], rows[i+rnlp.mpcc.meta.nnzh] =
            minmax(rnlp.mpcc.meta.ind_cc1[i], rnlp.mpcc.meta.ind_cc2[i])
    end
    return rows, cols
end
function NLPModels.hess_coord!(
    rnlp::ScholtesRelaxation{T, VT},
    x::AbstractVector{T},
    y::AbstractVector{T},
    H::AbstractVector{T};
    obj_weight::Real=one(T),
) where {T, VT}
    @views hess_coord!(
        rnlp.mpcc,
        x,
        y[1:rnlp.mpcc.meta.ncon],
        H[1:rnlp.mpcc.meta.nnzh];
        obj_weight=obj_weight,
    )
    for i in 1:rnlp.mpcc.meta.ncc
        H[i+rnlp.mpcc.meta.nnzh] = y[i+rnlp.mpcc.meta.ncon]
    end
    return H
end

function NLPModels.hprod!(
    rnlp::ScholtesRelaxation{T, VT},
    x::AbstractVector{T},
    y::AbstractVector{T},
    v::AbstractVector{T},
    Hv::AbstractVector;
    obj_weight::Real=one(T),
) where {T, VT}
    @views hprod!(rnlp.mpcc, x, y[1:rnlp.mpcc.meta.ncon], v, Hv; obj_weight=obj_weight)
    for i in 1:rnlp.mpcc.meta.ncc
        Hv[rnlp.mpcc.meta.ind_cc1[i]] +=
            v[rnlp.mpcc.meta.ind_cc2[i]]*y[i+rnlp.mpcc.meta.ncon]
        Hv[rnlp.mpcc.meta.ind_cc2[i]] +=
            v[rnlp.mpcc.meta.ind_cc1[i]]*y[i+rnlp.mpcc.meta.ncon]
    end
    return Hv
end
