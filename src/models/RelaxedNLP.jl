struct RelaxedNLP{T, VT} <: NLPModels.AbstractNLPModel{T, VT}
    mpcc::AbstractMPCCModel{T, VT}
    meta::NLPModels.NLPModelMeta{T, VT}
end

function RelaxedNLP(mpcc::AbstractMPCCModel{T, VT}) where {T, VT}
    if !is_vertical(mpcc)
        # TODO(@anton) Perhaps we should do this automatically in the future or we can support non-vertical form scholtes
        #              though this makes the callbacks a bit more complicated
        error(
            "Relaxed NLP creation currently expects a vertical form MPCC, use vertical_form(mpcc) to convert it.",
        )
    end

    # Update only the variable bounds
    uvar = copy(mpcc.meta.uvar)

    x0=copy(mpcc.meta.x0)
    # Copy x0 so changing RNLP x0 does not change mpcc x0
    meta = NLPModels.NLPModelMeta(mpcc.nlp.meta, uvar=uvar, x0=x0)
    return RelaxedNLP(mpcc, meta)
end

# Counters should be forwarded
function Base.getproperty(rnlp::RelaxedNLP, sym::Symbol)
    if sym ∈ [:counters]
        getproperty(rnlp.mpcc.nlp, sym)
    else
        getfield(rnlp, sym)
    end
end

######################### NLPModels Callbacks #########################
NLPModels.obj(rnlp::RelaxedNLP, x::AbstractVector) = NLPModels.obj(rnlp.mpcc, x)

function NLPModels.grad!(rnlp::RelaxedNLP, x::AbstractVector, gx::AbstractVector)
    return NLPModels.grad!(rnlp.mpcc, x, gx)
end

function NLPModels.objgrad!(rnlp::RelaxedNLP, x::AbstractVector, g::AbstractVector)
    return NLPModels.objgrad!(rnlp.mpcc, x, g)
end

function NLPModels.cons!(rnlp::RelaxedNLP, x::AbstractVector, cx::AbstractVector)
    mpcc_ncon = rnlp.mpcc.meta.ncon
    if get_ncon(rnlp.mpcc.nlp) > 0
        cons!(rnlp.mpcc, x, view(cx, 1:mpcc_ncon))
    end
    return cx
end

function NLPModels.cons_lin!(rnlp::RelaxedNLP, x::AbstractVector, cx::AbstractVector)
    if get_ncon(rnlp.mpcc.nlp) > 0
        return cons_lin!(rnlp.mpcc, x, cx)
    else
        return cx
    end
end

function NLPModels.cons_nln!(rnlp::RelaxedNLP, x::AbstractVector, cx::AbstractVector)
    # This if statement is necessary as it seems that without it c!(cx,x) does not exist in a possible underlying ADNLPModel
    if get_ncon(rnlp.mpcc.nlp) > 0
        cons_nln!(rnlp.mpcc, x, cx)
    end
    return cx
end

function NLPModels.jac_structure!(
    rnlp::RelaxedNLP,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    @views jac_structure!(rnlp.mpcc, rows, cols) # get including complementarities

    return rows, cols
end

function NLPModels.jac_lin_structure!(
    rnlp::RelaxedNLP,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    jac_lin_structure!(rnlp.mpcc, rows, cols) # get including complementarities
    return rows, cols
end

function NLPModels.jac_nln_structure!(
    rnlp::RelaxedNLP,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    @views jac_nln_structure!(rnlp.mpcc, rows, cols) # get including complementarities

    return rows, cols
end

function NLPModels.jac_coord!(rnlp::RelaxedNLP, x::AbstractVector, j::AbstractVector)
    jac_coord!(rnlp.mpcc, x, j)

    return j
end

function NLPModels.jac_lin_coord!(rnlp::RelaxedNLP, x::AbstractVector, jac::AbstractVector)
    return jac_lin_coord!(rnlp.mpcc, x, jac)
end

function NLPModels.jac_nln_coord!(rnlp::RelaxedNLP, x::AbstractVector, jac::AbstractVector)
    return jac_nln_coord!(rnlp.mpcc, x, jac)
end

function NLPModels.jprod_lin!(
    rnlp::RelaxedNLP,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jv[1:rnlp.meta.nlin] = jac_lin(rnlp, x) * v
    return Jv
end

function NLPModels.jprod_nln!(
    rnlp::RelaxedNLP,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jv[1:rnlp.meta.nnln] = jac_nln(rnlp, x) * v
    return Jv
end

function NLPModels.jtprod_lin!(
    rnlp::RelaxedNLP,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jtv[1:rnlp.meta.nvar] = jac_lin(rnlp, x)' * v
    return Jtv
end

function NLPModels.jtprod_nln!(
    rnlp::RelaxedNLP,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jtv[1:rnlp.meta.nvar] = jac_nln(rnlp, x)' * v
    return Jtv
end

function NLPModels.hess_structure!(
    rnlp::RelaxedNLP,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    return hess_structure!(rnlp.mpcc, rows, cols)
end
function NLPModels.hess_coord!(
    rnlp::RelaxedNLP{T, VT},
    x::AbstractVector{T},
    y::AbstractVector{T},
    H::AbstractVector{T};
    obj_weight::Real=one(T),
) where {T, VT}
    return hess_coord!(rnlp.mpcc, x, y, H; obj_weight=obj_weight)
end

function NLPModels.hprod!(
    rnlp::RelaxedNLP{T, VT},
    x::AbstractVector{T},
    y::AbstractVector{T},
    v::AbstractVector{T},
    Hv::AbstractVector;
    obj_weight::Real=one(T),
) where {T, VT}
    return hess_coord!(rnlp.mpcc, x, y, v, Hv; obj_weight=obj_weight)
end
