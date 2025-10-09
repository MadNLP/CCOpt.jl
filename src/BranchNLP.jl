struct BranchNLP{T, VT} <: NLPModels.AbstractNLPModel{T, VT}
    mpcc::AbstractMPCCModel{T, VT}
    meta::NLPModels.NLPModelMeta{T, VT}
    b::Vector{Bool}
end

function BranchNLP(mpcc::AbstractMPCCModel{T, VT}, b::Vector{Bool}) where {T, VT}
    if !is_vertical(mpcc)
        # TODO(@anton) Perhaps we should do this automatically in the future or we can support non-vertical form scholtes
        #              though this makes the callbacks a bit more complicated
        error(
            "Branch NLP creation currently expects a vertical form MPCC, use vertical_form(mpcc) to convert it.",
        )
    end

    # Update only the variable bounds
    uvar = copy(mpcc.meta.uvar)

    uvar[mpcc.meta.ind_cc1[.!b]] .= mpcc.meta.lvar[mpcc.meta.ind_cc1[.!b]]
    uvar[mpcc.meta.ind_cc2[b]] .= mpcc.meta.lvar[mpcc.meta.ind_cc2[b]]
    x0=copy(mpcc.meta.x0)
    # Copy x0 so changing BNLP x0 does not change mpcc x0
    meta = NLPModels.NLPModelMeta(mpcc.nlp.meta, uvar=uvar, x0=x0)
    return BranchNLP(mpcc, meta, b)
end

# Counters should be forwarded
function Base.getproperty(rnlp::BranchNLP, sym::Symbol)
    if sym ∈ [:counters]
        getproperty(rnlp.mpcc.nlp, sym)
    else
        getfield(rnlp, sym)
    end
end

######################### NLPModels Callbacks #########################
NLPModels.obj(bnlp::BranchNLP, x::AbstractVector) = NLPModels.obj(bnlp.mpcc, x)

function NLPModels.grad!(bnlp::BranchNLP, x::AbstractVector, gx::AbstractVector)
    return NLPModels.grad!(bnlp.mpcc, x, gx)
end

function NLPModels.objgrad!(bnlp::BranchNLP, x::AbstractVector, g::AbstractVector)
    return NLPModels.objgrad!(bnlp.mpcc, x, g)
end

function NLPModels.cons!(bnlp::BranchNLP, x::AbstractVector, cx::AbstractVector)
    mpcc_ncon = bnlp.mpcc.meta.ncon
    if get_ncon(bnlp.mpcc.nlp) > 0
        cons!(bnlp.mpcc, x, view(cx, 1:mpcc_ncon))
    end
    return cx
end

function NLPModels.cons_lin!(bnlp::BranchNLP, x::AbstractVector, cx::AbstractVector)
    if get_ncon(bnlp.mpcc.nlp) > 0
        return cons_lin!(bnlp.mpcc, x, cx)
    else
        return cx
    end
end

function NLPModels.cons_nln!(bnlp::BranchNLP, x::AbstractVector, cx::AbstractVector)
    # This if statement is necessary as it seems that without it c!(cx,x) does not exist in a possible underlying ADNLPModel
    if get_ncon(bnlp.mpcc.nlp) > 0
        cons_nln!(bnlp.mpcc, x, cx)
    end
    return cx
end

function NLPModels.jac_structure!(
    bnlp::BranchNLP,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    @views jac_structure!(bnlp.mpcc, rows, cols) # get including complementarities

    return rows, cols
end

function NLPModels.jac_lin_structure!(
    bnlp::BranchNLP,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    jac_lin_structure!(bnlp.mpcc, rows, cols) # get including complementarities
    return rows, cols
end

function NLPModels.jac_nln_structure!(
    bnlp::BranchNLP,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    @views jac_nln_structure!(bnlp.mpcc, rows, cols) # get including complementarities

    return rows, cols
end

function NLPModels.jac_coord!(bnlp::BranchNLP, x::AbstractVector, j::AbstractVector)
    jac_coord!(bnlp.mpcc, x, j)

    return j
end

function NLPModels.jac_lin_coord!(bnlp::BranchNLP, x::AbstractVector, jac::AbstractVector)
    return jac_lin_coord!(bnlp.mpcc, x, jac)
end

function NLPModels.jac_nln_coord!(bnlp::BranchNLP, x::AbstractVector, jac::AbstractVector)
    return jac_nln_coord!(bnlp.mpcc, x, jac)
end

function NLPModels.jprod_lin!(
    bnlp::BranchNLP,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jv[1:bnlp.meta.nlin] = jac_lin(bnlp, x) * v
    return Jv
end

function NLPModels.jprod_nln!(
    bnlp::BranchNLP,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jv[1:bnlp.meta.nnln] = jac_nln(bnlp, x) * v
    return Jv
end

function NLPModels.jtprod_lin!(
    bnlp::BranchNLP,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jtv[1:bnlp.meta.nvar] = jac_lin(bnlp, x)' * v
    return Jtv
end

function NLPModels.jtprod_nln!(
    bnlp::BranchNLP,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jtv[1:bnlp.meta.nvar] = jac_nln(bnlp, x)' * v
    return Jtv
end

function NLPModels.hess_structure!(
    bnlp::BranchNLP,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    return hess_structure!(bnlp.mpcc, rows, cols)
end
function NLPModels.hess_coord!(
    bnlp::BranchNLP{T, VT},
    x::AbstractVector{T},
    y::AbstractVector{T},
    H::AbstractVector{T};
    obj_weight::Real=one(T),
) where {T, VT}
    return hess_coord!(bnlp.mpcc, x, y, H; obj_weight=obj_weight)
end

function NLPModels.hprod!(
    bnlp::BranchNLP{T, VT},
    x::AbstractVector{T},
    y::AbstractVector{T},
    v::AbstractVector{T},
    Hv::AbstractVector;
    obj_weight::Real=one(T),
) where {T, VT}
    return hess_coord!(bnlp.mpcc, x, y, v, Hv; obj_weight=obj_weight)
end
