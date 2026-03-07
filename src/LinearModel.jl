struct LinearModel{T,VT,MAT<:AbstractMatrix{T},NLP_META} <: NLPModels.AbstractNLPModel{T, VT}
  c::VT
  A::MAT
  x0::VT

  meta::NLPModelMeta{T,VT,NLP_META}
end

function LinearModel(nlp::AbstractNLPModel{T,VT}, x0::VT; tr::T=Inf) where {T,VT}
    lvar = copy!(get_lvar(nlp))
    uvar = copy!(get_uvar(nlp))
    lvar .-= x0
    uvar .-= x0
    
    A = jac(nlp, x0)
    lcon = mul(A, x0)
    ucon = copy!(lcon)
    lcon .= get_lcon(nlp) - lcon
    ucon .= get_ucon(nlp) - ucon
    x0 = similar(get_x0(nlp))
    fill!(x0, 0.0)
    y0 = similar(get_y0(nlp))
    fill!(y0, 0.0)
    
    meta = NLPModels.NLPModelMeta(
        nlp.meta,
        lcon=lcon,
        ucon=ucon,
        lvar=lvar,
        uvar=uvar,
        x0=x0,
        nnzh=get_nnzj(nlp),
        nln_nnzj=0,
        lin_nnzj=get_nnzj(nlp),
        nnzh=0,
        nln_nnzh=0,
        lin_nnzh=0,
        lin=get_ncon(nlp),
        islp=true,
        y0=y0,
    )
end

function NLPModels.obj(lp::LinearModel, x::AbstractVector)
    return NLPModels.obj(lp.nlp, view(x, 1:get_nvar(lp.nlp)))
end
function NLPModels.grad!(lp::LinearModel, x::AbstractVector, gx::AbstractVector)
    @views NLPModels.grad!(lp.nlp, x[1:get_nvar(lp.nlp)], gx[1:get_nvar(lp.nlp)])
    gx[(get_nvar(lp.nlp)+1):get_nvar(lp)] .= 0
    return gx
end
function NLPModels.objgrad!(lp::LinearModel, x::AbstractVector, g::AbstractVector)
    return NLPModels.objgrad!(lp.nlp, view(x, 1:get_nvar(lp.nlp)), g)
end

function NLPModels.cons!(lp::LinearModel, x::AbstractVector, cx::AbstractVector)
    return mul!(cx,lp.A, x)
end

function NLPModels.cons_lin!(lp::LinearModel, x::AbstractVector, cx::AbstractVector)
    return mul!(cx,lp.A, x)
end

function NLPModels.cons_nln!(lp::LinearModel, x::AbstractVector, cx::AbstractVector)
    return cx
end

function fill_structure!(S::SparseMatrixCSC, rows, cols)
    count = 1
    @inbounds for col = 1:size(S, 2), k = S.colptr[col]:(S.colptr[col + 1] - 1)
        rows[count] = S.rowval[k]
        cols[count] = col
        count += 1
    end
end

function fill_coord!(S::SparseMatrixCSC, vals)
    count = 1
    @inbounds for col = 1:size(S, 2), k = S.colptr[col]:(S.colptr[col + 1] - 1)
        vals[count] = S.nzval[k]
        count += 1
    end
end

function NLPModels.jac_structure!(
    lp::LinearModel{T,VT,SparseMatrixCSC},
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
) where {T,VT}
    fill_structure!(lp.A, rows, cols)
    return rows, cols
end

function NLPModels.jac_lin_structure!(
    lp::LinearModel{T,VT,SparseMatrixCSC},
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
) where {T,VT}
    fill_structure!(lp.A, rows, cols)
    return rows, cols
end

function NLPModels.jac_nln_structure!(
    lp::LinearModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    return rows, cols
end

function NLPModels.jac_coord!(lp::LinearModel, x::AbstractVector, j::AbstractVector)
    fill_coord!(lp.A, j)
    return j
end

function NLPModels.jac_lin_coord!(
    lp::LinearModel,
    x::AbstractVector,
    j::AbstractVector,
)
    fill_coord!(lp.A, j)
    return j
end

function NLPModels.jac_nln_coord!(
    lp::LinearModel,
    x::AbstractVector,
    j::AbstractVector,
)
    return j
end

function NLPModels.jprod!(
    lp::LinearModel,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    return mul!(Jv, lp.A, v)
end

function NLPModels.jprod_lin!(
    lp::LinearModel,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    return mul!(Jv, lp.A, v)
end

function NLPModels.jprod_nln!(
    lp::LinearModel,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    return Jv
end

function NLPModels.jtprod!(
    lp::LinearModel,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    return mul!(Jtv, transpose(lp.A), v)
end

function NLPModels.jtprod_lin!(
    lp::LinearModel,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    return mul!(Jtv, transpose(lp.A), v)
end

function NLPModels.jtprod_nln!(
    lp::LinearModel,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    return Jtv
end

function NLPModels.hess_structure!(
    lp::LinearModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    return rows,cols
end

function NLPModels.hess_coord!(
    lp::LinearModel{T, VT},
    x::AbstractVector{T},
    y::AbstractVector{T},
    H::AbstractVector{T};
    obj_weight::Real=one(T),
) where {T, VT}
    return H

function NLPModels.hprod!(
    lp::LinearModel{T, VT},
    x::AbstractVector{T},
    y::AbstractVector{T},
    v::AbstractVector{T},
    Hv::AbstractVector;
    obj_weight::Real=one(T),
) where {T, VT}
    return Hv
end
