######################### Ell1 Relaxation #########################
struct Ell1Relaxation{T, VT} <: NLPModels.AbstractNLPModel{T, VT}
    mpcc::AbstractMPCCModel{T, VT}
    meta::NLPModels.NLPModelMeta{T, VT}
    𝜎::Base.RefValue{T}
end

function Ell1Relaxation(mpcc::AbstractMPCCModel{T, VT}) where {T, VT}
    if !is_vertical(mpcc)
        # TODO(@anton) Perhaps we should do this automatically in the future or we can support non-vertical form scholtes
        #              though this makes the callbacks a bit more complicated
        error(
            "Ell1 Relaxation currently expects a vertical form MPCC, use vertical_form(mpcc) to convert it.",
        )
    end

    # Update only what needs to be updated
    ncon = mpcc.meta.ncon
    lcon = mpcc.meta.lcon
    ucon = mpcc.meta.ucon
    y0 = mpcc.meta.y0
    nnzj = mpcc.meta.nnzj
    nln_nnzj = mpcc.meta.nln_nnzj

    nnzh = mpcc.meta.nnzh + mpcc.meta.ncc
    # TODO(@anton) We may need to change how nlv(b,o,c) are handled because we actually cannot
    #              backcalculate how these need to change necessarily.
    #              However these seem to not be used anywhere in the NLPModels API so I am ignoring them.

    meta = NLPModels.NLPModelMeta(
        mpcc.nlp.meta,
        ncon=ncon,
        lcon=lcon,
        ucon=ucon,
        y0=y0,
        nnzj=nnzj,
        nln_nnzj=nln_nnzj,
        nnzh=nnzh,
    )
    𝜎 = zero(T)
    return Ell1Relaxation(mpcc, meta, Ref(𝜎))
end

# Counters should be forwarded
function Base.getproperty(rnlp::Ell1Relaxation, sym::Symbol)
    if sym ∈ [:counters]
        getproperty(rnlp.mpcc.nlp, sym)
    else
        getfield(rnlp, sym)
    end
end

######################### NLPModels Callbacks #########################
function NLPModels.obj(rnlp::Ell1Relaxation, x::AbstractVector)
    obj = NLPModels.obj(rnlp.mpcc, x)
    for i in 1:rnlp.mpcc.meta.ncc
        icc1 = rnlp.mpcc.meta.ind_cc1[i]
        icc2 = rnlp.mpcc.meta.ind_cc2[i]
        obj +=
            (1/rnlp.𝜎[]) *
            (x[icc1] - rnlp.meta.lvar[icc1]) *
            (x[icc2] - rnlp.meta.lvar[icc2])
    end
    return obj
end

# TODO update grad
function NLPModels.grad!(rnlp::Ell1Relaxation, x::AbstractVector, gx::AbstractVector)
    NLPModels.grad!(rnlp.mpcc, x, gx)
    for i in 1:rnlp.mpcc.meta.ncc
        icc1 = rnlp.mpcc.meta.ind_cc1[i]
        icc2 = rnlp.mpcc.meta.ind_cc2[i]
        gx[icc1] += (1/rnlp.𝜎[]) * (x[icc2] - rnlp.meta.lvar[icc2])
        gx[icc2] += (1/rnlp.𝜎[]) * (x[icc1] - rnlp.meta.lvar[icc1])
    end
    return gx
end

function NLPModels.objgrad!(rnlp::Ell1Relaxation, x::AbstractVector, gx::AbstractVector)
    obj, gx = NLPModels.objgrad!(rnlp.mpcc, x, gx)
    for i in 1:rnlp.mpcc.meta.ncc
        icc1 = rnlp.mpcc.meta.ind_cc1[i]
        icc2 = rnlp.mpcc.meta.ind_cc2[i]
        obj +=
            (1/rnlp.𝜎[]) *
            (x[icc1] - rnlp.meta.lvar[icc1]) *
            (x[icc2] - rnlp.meta.lvar[icc2])
        gx[icc1] += (1/rnlp.𝜎[]) * (x[icc2] - rnlp.meta.lvar[icc2])
        gx[icc2] += (1/rnlp.𝜎[]) * (x[icc1] - rnlp.meta.lvar[icc1])
    end
    return obj, gx
end

function NLPModels.cons!(rnlp::Ell1Relaxation, x::AbstractVector, cx::AbstractVector)
    mpcc_ncon = rnlp.mpcc.meta.ncon
    if get_ncon(rnlp.mpcc.nlp) > 0
        cons!(rnlp.mpcc, x, view(cx, 1:mpcc_ncon))
    end
    return cx
end

function NLPModels.cons_lin!(rnlp::Ell1Relaxation, x::AbstractVector, cx::AbstractVector)
    if get_ncon(rnlp.mpcc.nlp) > 0
        return cons_lin!(rnlp.mpcc, x, cx)
    else
        return cx
    end
end

function NLPModels.cons_nln!(rnlp::Ell1Relaxation, x::AbstractVector, cx::AbstractVector)
    mpcc_nnln = rnlp.mpcc.meta.nnln
    # This if statement is necessary as it seems that without it c!(cx,x) does not exist in a possible underlying ADNLPModel
    if get_ncon(rnlp.mpcc.nlp) > 0
        cons_nln!(rnlp.mpcc, x, view(cx, 1:mpcc_nnln))
    end
    return cx
end

function NLPModels.jac_structure!(
    rnlp::Ell1Relaxation,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    @views jac_structure!(
        rnlp.mpcc,
        rows[1:rnlp.mpcc.meta.nnzj],
        cols[1:rnlp.mpcc.meta.nnzj],
    ) # get including complementarities

    return rows, cols
end

function NLPModels.jac_lin_structure!(
    rnlp::Ell1Relaxation,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    jac_lin_structure!(rnlp.mpcc, rows, cols) # get including complementarities
    return rows, cols
end

function NLPModels.jac_nln_structure!(
    rnlp::Ell1Relaxation,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    @views jac_nln_structure!(
        rnlp.mpcc,
        rows[1:rnlp.mpcc.meta.nln_nnzj],
        cols[1:rnlp.mpcc.meta.nln_nnzj],
    ) # get including complementarities

    return rows, cols
end

function NLPModels.jac_coord!(rnlp::Ell1Relaxation, x::AbstractVector, j::AbstractVector)
    jac_coord!(rnlp.mpcc, x, @view(j[1:rnlp.mpcc.meta.nnzj]))

    return j
end

function NLPModels.jac_lin_coord!(
    rnlp::Ell1Relaxation,
    x::AbstractVector,
    jac::AbstractVector,
)
    return jac_lin_coord!(rnlp.mpcc, x, jac)
end

function NLPModels.jac_nln_coord!(
    rnlp::Ell1Relaxation,
    x::AbstractVector,
    jac::AbstractVector,
)
    jac_nln_coord!(rnlp.mpcc, x, @view(jac[1:rnlp.mpcc.meta.nln_nnzj]))

    return jac
end

function NLPModels.jprod_lin!(
    rnlp::Ell1Relaxation,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jv[1:rnlp.meta.nlin] = jac_lin(rnlp, x) * v
    return Jv
end

function NLPModels.jprod_nln!(
    rnlp::Ell1Relaxation,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jv[1:rnlp.meta.nnln] = jac_nln(rnlp, x) * v
    return Jv
end

function NLPModels.jtprod_lin!(
    rnlp::Ell1Relaxation,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jtv[1:rnlp.meta.nvar] = jac_lin(rnlp, x)' * v
    return Jtv
end

function NLPModels.jtprod_nln!(
    rnlp::Ell1Relaxation,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jtv[1:rnlp.meta.nvar] = jac_nln(rnlp, x)' * v
    return Jtv
end

function NLPModels.hess_structure!(
    rnlp::Ell1Relaxation,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
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
    rnlp::Ell1Relaxation{T, VT},
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
        H[i+rnlp.mpcc.meta.nnzh] = (1/rnlp.𝜎[])
    end
    return H
end

function NLPModels.hprod!(
    rnlp::Ell1Relaxation{T, VT},
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
