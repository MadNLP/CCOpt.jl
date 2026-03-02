######################### Lasry-Lions Penalty Relaxation #########################
struct LasryLionsPenalty{T, VT} <: AbstractMPCCPenaltyModel{T, VT}
    mpcc::AbstractMPCCModel{T, VT}
    meta::NLPModels.NLPModelMeta{T, VT}
    β::Base.RefValue{T}
    λ::Base.RefValue{T}
end

function lasrylions(x1::T, x2::T, λ::T, β::T) where {T}
    if x1 < 0 && x2 < 0
        return inv(λ)*inv(T(2)*(T(1)-β))*(x1^2 + x2^2)
    elseif (1-β)*x1 <= x2 && x2 <= inv(1-β)*x1
        return inv(λ)*(inv(T(2)*β*(T(2)-β))*(x1+x2)^2 - inv(T(2)*β)*(x1^2 + x2^2))
    else
        return inv(λ)*inv(T(2)*(T(1)-β))*min(x1, x2)^2
    end
end

function dlasrylions(x1::T, x2::T, λ::T, β::T) where {T}
    if x1 < 0 && x2 < 0
        return (inv(λ)*inv(1-β)*x1, inv(λ)*inv(1-β)*x2)
    elseif (T(1)-β)*x1 <= x2 && x2 <= inv(T(1)-β)*x1
        return (
            inv(λ)*(inv(β*(T(2)-β))*(x1+x2) - inv(β)*x1),
            inv(λ)*(inv(β*(T(2)-β))*(x1+x2) - inv(β)*x2),
        )
    elseif x2 >= inv(T(1)-β)*x1
        return (inv(λ)*inv((T(1)-β))*x1, T(0))
    else
        return (T(0), inv(λ)*inv((T(1)-β))*x2)
    end
end

function ddlasrylions(x1::T, x2::T, λ::T, β::T) where {T}
    if x1 < 0 && x2 < 0
        return (inv(λ)*inv(1-β), inv(λ)*inv(1-β), 0)
    elseif (T(1)-β)*x1 <= x2 && x2 <= inv(T(1)-β)*x1
        return (
            inv(λ)*(inv(β*(T(2)-β)) - inv(β)),
            inv(λ)*(inv(β*(T(2)-β)) - inv(β)),
            inv(λ)*(inv(β*(T(2)-β))),
        )
    elseif x2 >= inv(T(1)-β)*x1
        return (inv(λ)*inv((T(1)-β)), T(0), T(0))
    else
        return (T(0), inv(λ)*inv((T(1)-β)), T(0))
    end
end

function LasryLionsPenalty(
    mpcc::AbstractMPCCModel{T, VT};
    β::T=T(0.9999),
    λ::T=T(0.0),
) where {T, VT}
    if !is_vertical(mpcc)
        # TODO(@anton) Perhaps we should do this automatically in the future or we can support non-vertical form scholtes
        #              though this makes the callbacks a bit more complicated
        error(
            "Lasry-Lions Relaxation currently expects a vertical form MPCC, use vertical_form(mpcc) to convert it.",
        )
    end

    # Update only what needs to be updated
    ncon = get_ncon(mpcc)
    lcon = get_lcon(mpcc)
    ucon = get_ucon(mpcc)
    y0 = get_y0(mpcc)
    nnzj = get_nnzj(mpcc)
    nln_nnzj = get_nln_nnzj(mpcc)

    nnzh = get_nnzh(mpcc) + 3*get_ncc(mpcc)
    # TODO(@anton) We may need to change how nlv(b,o,c) are handled because we actually cannot
    #              backcalculate how these need to change necessarily.
    #              However these seem to not be used anywhere in the NLPModels API so I am ignoring them.

    lvar = copy(get_lvar(mpcc))
    lvar[get_ind_cc1(mpcc)] .= T(-Inf)
    lvar[get_ind_cc2(mpcc)] .= T(-Inf)

    meta = NLPModels.NLPModelMeta(
        mpcc.nlp.meta,
        ncon=ncon,
        lcon=lcon,
        ucon=ucon,
        y0=y0,
        nnzj=nnzj,
        nln_nnzj=nln_nnzj,
        nnzh=nnzh,
        lvar=lvar,
        grad_available=true,
        jac_available=true,
        hess_available=true,
        jprod_available=true,
        jtprod_available=true,
        hprod_available=true,
    )
    return LasryLionsPenalty(mpcc, meta, Ref(β), Ref(λ))
end

# Counters should be forwarded
function Base.getproperty(rnlp::LasryLionsPenalty, sym::Symbol)
    if sym ∈ [:counters]
        getproperty(rnlp.mpcc.nlp, sym)
    else
        getfield(rnlp, sym)
    end
end

######################### NLPModels Callbacks #########################
function NLPModels.obj(rnlp::LasryLionsPenalty{T, VT}, x::AbstractVector) where {T, VT}
    obj = NLPModels.obj(rnlp.mpcc, x)
    sense = rnlp.meta.minimize ? one(T) : -one(T)
    @views begin
        obj +=
            sense * mapreduce(
                (x1, lx1, x2, lx2)->lasrylions(x1-lx1, x2-lx2, rnlp.λ[], rnlp.β[]),
                +,
                x[get_ind_cc1(rnlp.mpcc)],
                get_lvar(rnlp.mpcc)[get_ind_cc1(rnlp.mpcc)],
                x[get_ind_cc2(rnlp.mpcc)],
                get_lvar(rnlp.mpcc)[get_ind_cc2(rnlp.mpcc)],
            )
    end
    return obj
end

function NLPModels.grad!(
    rnlp::LasryLionsPenalty{T, VT},
    x::AbstractVector,
    gx::AbstractVector,
) where {T, VT}
    NLPModels.grad!(rnlp.mpcc, x, gx)
    sense = rnlp.meta.minimize ? one(T) : -one(T)
    for i in 1:get_ncc(rnlp.mpcc)
        icc1 = get_ind_cc1(rnlp.mpcc)[i]
        icc2 = get_ind_cc2(rnlp.mpcc)[i]
        (d1, d2) = dlasrylions(
            (x[icc1] - get_lvar(rnlp.mpcc)[icc1]),
            (x[icc2] - get_lvar(rnlp.mpcc)[icc2]),
            rnlp.λ[],
            rnlp.β[],
        )
        gx[icc1] += sense * d1
        gx[icc2] += sense * d2
    end
    return gx
end

function NLPModels.objgrad!(
    rnlp::LasryLionsPenalty{T, VT},
    x::AbstractVector,
    gx::AbstractVector,
) where {T, VT}
    obj, gx = NLPModels.objgrad!(rnlp.mpcc, x, gx)
    sense = rnlp.meta.minimize ? one(T) : -one(T)
    @views begin
        obj +=
            sense * mapreduce(
                (x1, lx1, x2, lx2)->lasrylions(x1-lx1, x2-lx2, rnlp.λ[], rnlp.β[]),
                +,
                x[get_ind_cc1(rnlp.mpcc)],
                get_lvar(rnlp.mpcc)[get_ind_cc1(rnlp.mpcc)],
                x[get_ind_cc2(rnlp.mpcc)],
                get_lvar(rnlp.mpcc)[get_ind_cc2(rnlp.mpcc)],
            )
    end
    for i in 1:get_ncc(rnlp.mpcc)
        icc1 = get_ind_cc1(rnlp.mpcc)[i]
        icc2 = get_ind_cc2(rnlp.mpcc)[i]
        (d1, d2) = dlasrylions(
            (x[icc1] - get_lvar(rnlp.mpcc)[icc1]),
            (x[icc2] - get_lvar(rnlp.mpcc)[icc2]),
            rnlp.λ[],
            rnlp.β[],
        )
        gx[icc1] += sense * d1
        gx[icc2] += sense * d2
    end
    return obj, gx
end

function NLPModels.cons!(rnlp::LasryLionsPenalty, x::AbstractVector, cx::AbstractVector)
    mpcc_ncon = get_ncon(rnlp.mpcc)
    if get_ncon(rnlp.mpcc.nlp) > 0
        cons!(rnlp.mpcc, x, view(cx, 1:mpcc_ncon))
    end
    return cx
end

function NLPModels.cons_lin!(rnlp::LasryLionsPenalty, x::AbstractVector, cx::AbstractVector)
    if get_ncon(rnlp.mpcc.nlp) > 0
        return cons_lin!(rnlp.mpcc, x, cx)
    else
        return cx
    end
end

function NLPModels.cons_nln!(rnlp::LasryLionsPenalty, x::AbstractVector, cx::AbstractVector)
    mpcc_nnln = get_nnln(rnlp.mpcc)
    # This if statement is necessary as it seems that without it c!(cx,x) does not exist in a possible underlying ADNLPModel
    if get_ncon(rnlp.mpcc.nlp) > 0
        cons_nln!(rnlp.mpcc, x, view(cx, 1:mpcc_nnln))
    end
    return cx
end

function NLPModels.jac_structure!(
    rnlp::LasryLionsPenalty,
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
    rnlp::LasryLionsPenalty,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    jac_lin_structure!(rnlp.mpcc, rows, cols) # get including complementarities
    return rows, cols
end

function NLPModels.jac_nln_structure!(
    rnlp::LasryLionsPenalty,
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

function NLPModels.jac_coord!(rnlp::LasryLionsPenalty, x::AbstractVector, j::AbstractVector)
    jac_coord!(rnlp.mpcc, x, @view(j[1:get_nnzj(rnlp.mpcc)]))

    return j
end

function NLPModels.jac_lin_coord!(
    rnlp::LasryLionsPenalty,
    x::AbstractVector,
    jac::AbstractVector,
)
    return jac_lin_coord!(rnlp.mpcc, x, jac)
end

function NLPModels.jac_nln_coord!(
    rnlp::LasryLionsPenalty,
    x::AbstractVector,
    jac::AbstractVector,
)
    jac_nln_coord!(rnlp.mpcc, x, @view(jac[1:get_nln_nnzj(rnlp.mpcc)]))

    return jac
end

function NLPModels.jprod_lin!(
    rnlp::LasryLionsPenalty,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jv[1:rnlp.meta.nlin] = jac_lin(rnlp, x) * v
    return Jv
end

function NLPModels.jprod_nln!(
    rnlp::LasryLionsPenalty,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jv[1:rnlp.meta.nnln] = jac_nln(rnlp, x) * v
    return Jv
end

function NLPModels.jtprod_lin!(
    rnlp::LasryLionsPenalty,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jtv[1:rnlp.meta.nvar] = jac_lin(rnlp, x)' * v
    return Jtv
end

function NLPModels.jtprod_nln!(
    rnlp::LasryLionsPenalty,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jtv[1:rnlp.meta.nvar] = jac_nln(rnlp, x)' * v
    return Jtv
end

function NLPModels.hess_structure!(
    rnlp::LasryLionsPenalty,
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
    nnzh = get_nnzh(rnlp.mpcc)
    ncc = get_ncc(rnlp.mpcc)
    # Off diagonal terms
    for i in 1:ncc
        cols[i+nnzh], rows[i+nnzh] =
            minmax(get_ind_cc1(rnlp.mpcc)[i], get_ind_cc2(rnlp.mpcc)[i])
    end
    # Diagonal terms
    for i in 1:ncc
        cols[i+nnzh+ncc], rows[i+nnzh+ncc] =
            get_ind_cc1(rnlp.mpcc)[i], get_ind_cc1(rnlp.mpcc)[i]
    end
    for i in 1:ncc
        cols[i+nnzh+2*ncc], rows[i+nnzh+2*ncc] =
            get_ind_cc2(rnlp.mpcc)[i], get_ind_cc2(rnlp.mpcc)[i]
    end
    return rows, cols
end
function NLPModels.hess_coord!(
    rnlp::LasryLionsPenalty{T, VT},
    x::AbstractVector{T},
    y::AbstractVector{T},
    H::AbstractVector{T};
    obj_weight::Real=one(T),
) where {T, VT}
    @views hess_coord!(
        rnlp.mpcc,
        x,
        y[1:get_ncon(rnlp.mpcc)],
        H[1:get_nnzh(rnlp.mpcc)];
        obj_weight=obj_weight,
    )
    sense = rnlp.meta.minimize ? one(T) : -one(T)
    # TODO(@anton) deduplicate by maybe having one more buffer for a^2 + b^2
    nnzh = get_nnzh(rnlp.mpcc)
    ncc = get_ncc(rnlp.mpcc)
    ncon = get_ncon(rnlp.mpcc)
    for i in 1:ncc
        icc1 = get_ind_cc1(rnlp.mpcc)[i]
        icc2 = get_ind_cc2(rnlp.mpcc)[i]
        (q11, q22, q12) = ddlasrylions(
            (x[icc1] - get_lvar(rnlp.mpcc)[icc1]),
            (x[icc2] - get_lvar(rnlp.mpcc)[icc2]),
            rnlp.λ[],
            rnlp.β[],
        )
        H[i+nnzh] = q12
        H[i+nnzh+ncc] = q11
        H[i+nnzh+2*ncc] = q22
    end
    return H
end

function NLPModels.hprod!(
    rnlp::LasryLionsPenalty{T, VT},
    x::AbstractVector{T},
    y::AbstractVector{T},
    v::AbstractVector{T},
    Hv::AbstractVector;
    obj_weight::Real=one(T),
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

function get_penalty(rnlp::LasryLionsPenalty)
    return inv(rnlp.λ[])
end

function set_penalty(rnlp::LasryLionsPenalty{T}, ρ::T) where {T}
    rnlp.λ[] = inv(ρ)
    return nothing
end
