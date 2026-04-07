######################### Ell1 Relaxation #########################
struct Ell1Relaxation{T, VT, MT} <: AbstractMPCCPenaltyModel{T, VT}
    mpcc::MT
    meta::NLPModels.NLPModelMeta{T, VT}
    ρ::Base.RefValue{T}
    counters::NLPModels.Counters
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
    ncon = get_ncon(mpcc)
    lcon = get_lcon(mpcc)
    ucon = get_ucon(mpcc)
    y0 = get_y0(mpcc)
    nnzj = get_nnzj(mpcc)
    nln_nnzj = get_nln_nnzj(mpcc)

    nnzh = get_nnzh(mpcc) + get_ncc(mpcc)
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
        grad_available=true,
        jac_available=true,
        hess_available=true,
        jprod_available=true,
        jtprod_available=true,
        hprod_available=true,
    )
    ρ = zero(T)
    return Ell1Relaxation(mpcc, meta, Ref(ρ), mpcc.counters)
end

######################### NLPModels Callbacks #########################
function NLPModels.obj(rnlp::Ell1Relaxation{T, VT}, x::AbstractVector) where {T, VT}
    obj = NLPModels.obj(rnlp.mpcc, x)
    sense = rnlp.meta.minimize ? one(T) : -one(T)
    for i in 1:get_ncc(rnlp.mpcc)
        icc1 = get_ind_cc1(rnlp.mpcc)[i]
        icc2 = get_ind_cc2(rnlp.mpcc)[i]
        obj +=
            sense *
            (rnlp.ρ[]) *
            (x[icc1] - rnlp.meta.lvar[icc1]) *
            (x[icc2] - rnlp.meta.lvar[icc2])
    end
    return obj
end

function NLPModels.grad!(
    rnlp::Ell1Relaxation{T, VT},
    x::AbstractVector,
    gx::AbstractVector,
) where {T, VT}
    NLPModels.grad!(rnlp.mpcc, x, gx)
    sense = rnlp.meta.minimize ? one(T) : -one(T)
    for i in 1:get_ncc(rnlp.mpcc)
        icc1 = get_ind_cc1(rnlp.mpcc)[i]
        icc2 = get_ind_cc2(rnlp.mpcc)[i]
        gx[icc1] += sense * (rnlp.ρ[]) * (x[icc2] - rnlp.meta.lvar[icc2])
        gx[icc2] += sense * (rnlp.ρ[]) * (x[icc1] - rnlp.meta.lvar[icc1])
    end
    return gx
end

function NLPModels.objgrad!(
    rnlp::Ell1Relaxation{T, VT},
    x::AbstractVector,
    gx::AbstractVector,
) where {T, VT}
    obj, gx = NLPModels.objgrad!(rnlp.mpcc, x, gx)
    sense = rnlp.meta.minimize ? one(T) : -one(T)
    for i in 1:get_ncc(rnlp.mpcc)
        icc1 = get_ind_cc1(rnlp.mpcc)[i]
        icc2 = get_ind_cc2(rnlp.mpcc)[i]
        obj +=
            sense *
            (rnlp.ρ[]) *
            (x[icc1] - rnlp.meta.lvar[icc1]) *
            (x[icc2] - rnlp.meta.lvar[icc2])
        gx[icc1] += sense * (rnlp.ρ[]) * (x[icc2] - rnlp.meta.lvar[icc2])
        gx[icc2] += sense * (rnlp.ρ[]) * (x[icc1] - rnlp.meta.lvar[icc1])
    end
    return obj, gx
end

function NLPModels.cons!(rnlp::Ell1Relaxation, x::AbstractVector, cx::AbstractVector)
    mpcc_ncon = get_ncon(rnlp.mpcc)
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
    mpcc_nnln = get_nnln(rnlp.mpcc)
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
        rows[1:get_nnzj(rnlp.mpcc)],
        cols[1:get_nnzj(rnlp.mpcc)],
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
        rows[1:get_nln_nnzj(rnlp.mpcc)],
        cols[1:get_nln_nnzj(rnlp.mpcc)],
    ) # get including complementarities

    return rows, cols
end

function NLPModels.jac_coord!(rnlp::Ell1Relaxation, x::AbstractVector, j::AbstractVector)
    jac_coord!(rnlp.mpcc, x, @view(j[1:get_nnzj(rnlp.mpcc)]))

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
    jac_nln_coord!(rnlp.mpcc, x, @view(jac[1:get_nln_nnzj(rnlp.mpcc)]))

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
        rows[1:get_nnzh(rnlp.mpcc)],
        cols[1:get_nnzh(rnlp.mpcc)],
    )
    # TODO(@anton) it seems hard to vectorize in one operation this because there is no efficient unzip in Base:
    #              See https://github.com/JuliaLang/julia/issues/13942 for details
    for i in 1:get_ncc(rnlp.mpcc)
        cols[i+get_nnzh(rnlp.mpcc)], rows[i+get_nnzh(rnlp.mpcc)] =
            minmax(get_ind_cc1(rnlp.mpcc)[i], get_ind_cc2(rnlp.mpcc)[i])
    end
    return rows, cols
end
function NLPModels.hess_coord!(
    rnlp::Ell1Relaxation{T, VT},
    x::AbstractVector{T},
    y::AbstractVector{T},
    H::AbstractVector{T};
    obj_weight::T=one(T),
) where {T, VT}
    @views hess_coord!(
        rnlp.mpcc,
        x,
        y[1:get_ncon(rnlp.mpcc)],
        H[1:get_nnzh(rnlp.mpcc)];
        obj_weight=obj_weight,
    )
    sense = rnlp.meta.minimize ? one(T) : -one(T)
    for i in 1:get_ncc(rnlp.mpcc)
        H[i+get_nnzh(rnlp.mpcc)] = obj_weight*sense*(rnlp.ρ[])
    end
    return H
end

function NLPModels.hprod!(
    rnlp::Ell1Relaxation{T, VT},
    x::AbstractVector{T},
    y::AbstractVector{T},
    v::AbstractVector{T},
    Hv::AbstractVector;
    obj_weight::T=one(T),
) where {T, VT}
    @views hprod!(rnlp.mpcc, x, y[1:get_ncon(rnlp.mpcc)], v, Hv; obj_weight=obj_weight)
    sense = rnlp.meta.minimize ? one(T) : -one(T)
    for i in 1:get_ncc(rnlp.mpcc)
        Hv[get_ind_cc1(rnlp.mpcc)[i]] +=
            obj_weight*sense*v[get_ind_cc2(rnlp.mpcc)[i]]*(rnlp.ρ[])
        Hv[get_ind_cc2(rnlp.mpcc)[i]] +=
            obj_weight*sense*v[get_ind_cc1(rnlp.mpcc)[i]]*(rnlp.ρ[])
    end
    return Hv
end
