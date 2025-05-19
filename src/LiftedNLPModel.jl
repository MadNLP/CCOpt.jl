"""
  Lifted NLP model which applies lifting to given indices of the nonlinear constraints.
  Currently takes a parent model and a list of indicies of constraints that should be Lifted.
  The lifting variables then take on the bounds of the original
"""
######################### Metadata Definition #########################
struct LiftedNLPModelMeta{T, VT} <: AbstractNLPModelMeta{T, VT}
    parent::AbstractNLPModelMeta{T, VT}

    ind_lift::IndexSet
    ind_lin_lift::IndexSet
    ind_nln_lift::IndexSet
    ind_lift_var
    ind_lin_lift_var::IndexSet
    ind_nln_lift_var::IndexSet
    nlift::Int
    lin_nlift::Int
    nln_nlift::Int
end

function Base.getproperty(meta::LiftedNLPModelMeta, sym::Symbol)
    if sym ∈ fieldnames(LiftedNLPModelMeta) # NOTE: This is either elegant or EXTREMELY not, depending on how "static" the field names are
        getfield(meta, sym)
    else
        getproperty(meta.parent, sym)
    end
end

######################### LiftedNLPModel Definition #########################
# TODO(@anton) should this be a mutable struct
struct LiftedNLPModel{T, VT} <: AbstractNLPModel{T, VT}
    nlp::AbstractNLPModel{T, VT}

    meta::LiftedNLPModelMeta{T, VT}
end

# Counters should be forwarded
function Base.getproperty(lnlp::LiftedNLPModel, sym::Symbol)
    if sym ∈ [:counters]
        getproperty(lnlp.nlp, sym)
    else
        getfield(lnlp, sym)
    end
end

function LiftedNLPModel(nlp::AbstractNLPModel, ind_lift::IndexSet)
    # Get indicies for lin/nln
    # TODO(@anton) Perhaps warn if lifting linear constraints
    ind_lin_lift = [i for i in 1:nlp.meta.nlin if nlp.meta.lin[i] ∈ ind_lift]
    ind_nln_lift = [i for i in 1:nlp.meta.nnln if nlp.meta.nln[i] ∈ ind_lift]

    # number of lifting variables
    nlift = length(ind_lift)
    lin_nlift = length(ind_lin_lift)
    nln_nlift = length(ind_nln_lift)
    nvar = nlp.meta.nvar + nlift

    ind_lift_var = (nlp.meta.nvar+1):(nlp.meta.nvar+nlift)
    ind_lin_lift_var = [nlp.meta.nvar+i for i in 1:nlift if ind_lift[i] ∈ nlp.meta.lin]
    ind_nln_lift_var = [nlp.meta.nvar+i for i in 1:nlift if ind_lift[1] ∈ nlp.meta.nln]

    # add variable bounds for slacks and set initial value to the residual
    lvar = vcat(nlp.meta.lvar, nlp.meta.lcon[ind_lift])
    uvar = vcat(nlp.meta.uvar, nlp.meta.ucon[ind_lift])
    x0 = vcat(nlp.meta.x0, .-cons(nlp, nlp.meta.x0)[ind_lift])

    # Update the constraints to equality constraints.
    # TODO(@anton) also check if lifting equality constraints for some reason
    lcon = copy(nlp.meta.lcon)
    lcon[ind_lift] .= 0
    ucon = copy(nlp.meta.ucon)
    ucon[ind_lift] .= 0

    # Nonzeros for lifting variables in the jacobian
    nnzj = nlp.meta.nnzj + nlift
    lin_nnzj = nlp.meta.lin_nnzj + lin_nlift
    nln_nnzj = nlp.meta.nln_nnzj + nln_nlift

    parent_meta = NLPModels.NLPModelMeta(
        nlp.meta,
        nvar=nvar,
        lcon=lcon,
        ucon=ucon,
        lvar=lvar,
        uvar=uvar,
        x0=x0,
        nnzj=nnzj,
        nln_nnzj=nln_nnzj,
        lin_nnzj=lin_nnzj,
    )

    meta = LiftedNLPModelMeta(
        parent_meta,
        ind_lift,
        ind_lin_lift,
        ind_nln_lift,
        ind_lift_var,
        ind_lin_lift_var,
        ind_nln_lift_var,
        nlift,
        lin_nlift,
        nln_nlift,
    )

    return LiftedNLPModel(nlp, meta)
end

######################### NLPModels API Implementation #########################
function NLPModels.obj(lnlp::LiftedNLPModel, x::AbstractVector)
    return NLPModels.obj(lnlp.nlp, view(x, 1:lnlp.nlp.meta.nvar))
end
function NLPModels.grad!(lnlp::LiftedNLPModel, x::AbstractVector, gx::AbstractVector)
    @views NLPModels.grad!(lnlp.nlp, x[1:lnlp.nlp.meta.nvar], gx[1:lnlp.nlp.meta.nvar])
    gx[(lnlp.nlp.meta.nvar+1):lnlp.meta.nvar] .= 0
    return gx
end
function NLPModels.objgrad!(lnlp::LiftedNLPModel, x::AbstractVector, g::AbstractVector)
    return NLPModels.objgrad!(lnlp.nlp, view(x, 1:lnlp.nlp.meta.nvar), g)
end

function NLPModels.cons!(lnlp::LiftedNLPModel, x::AbstractVector, cx::AbstractVector)
    cons!(lnlp.nlp, view(x, 1:lnlp.nlp.meta.nvar), cx)
    cx[lnlp.meta.ind_lift] .-= x[lnlp.meta.ind_lift_var]
    return cx
end

function NLPModels.cons_lin!(lnlp::LiftedNLPModel, x::AbstractVector, cx::AbstractVector)
    cons_lin!(lnlp.nlp, view(x, 1:lnlp.nlp.meta.nvar), cx)
    cx[lnlp.meta.ind_lin_lift] .-= x[lnlp.meta.ind_lin_lift_var]
    return cx
end

function NLPModels.cons_nln!(lnlp::LiftedNLPModel, x::AbstractVector, cx::AbstractVector)
    cons_nln!(lnlp.nlp, view(x, 1:lnlp.nlp.meta.nvar), cx)
    cx[lnlp.meta.ind_nln_lift] .-= x[lnlp.meta.ind_nln_lift_var]
    return cx
end

function NLPModels.jac_structure!(
    lnlp::LiftedNLPModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    @views jac_structure!(lnlp.nlp, rows[1:lnlp.nlp.meta.nnzj], cols[1:lnlp.nlp.meta.nnzj]) # get including complementarities

    for i in 1:lnlp.meta.nlift
        rows[i+lnlp.nlp.meta.nnzj] = lnlp.meta.ind_lift[i]
        cols[i+lnlp.nlp.meta.nnzj] = lnlp.meta.ind_lift_var[i]
    end
    return rows, cols
end

function NLPModels.jac_lin_structure!(
    lnlp::LiftedNLPModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    @views jac_lin_structure!(
        lnlp.nlp,
        rows[1:lnlp.nlp.meta.lin_nnzj],
        cols[1:lnlp.nlp.meta.lin_nnzj],
    ) # get including complementarities

    for i in 1:lnlp.meta.lin_nlift
        rows[i+lnlp.nlp.meta.lin_nnzj] = lnlp.meta.ind_lin_lift[i]
        cols[i+lnlp.nlp.meta.lin_nnzj] = lnlp.meta.ind_lin_lift_var[i]
    end
    return rows, cols
end

function NLPModels.jac_nln_structure!(
    lnlp::LiftedNLPModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    @views jac_nln_structure!(
        lnlp.nlp,
        rows[1:lnlp.nlp.meta.nln_nnzj],
        cols[1:lnlp.nlp.meta.nln_nnzj],
    )

    for i in 1:lnlp.meta.nln_nlift
        rows[i+lnlp.nlp.meta.nln_nnzj] = lnlp.meta.ind_nln_lift[i]
        cols[i+lnlp.nlp.meta.nln_nnzj] = lnlp.meta.ind_nln_lift_var[i]
    end
    return rows, cols
end

function NLPModels.jac_coord!(lnlp::LiftedNLPModel, x::AbstractVector, j::AbstractVector)
    @views jac_coord!(lnlp.nlp, x[1:lnlp.nlp.meta.nvar], j[1:lnlp.nlp.meta.nnzj])
    for i in 1:lnlp.meta.nlift
        j[i+lnlp.nlp.meta.nnzj] = -1
    end
    return j
end

function NLPModels.jac_lin_coord!(
    lnlp::LiftedNLPModel,
    x::AbstractVector,
    j::AbstractVector,
)
    @views jac_lin_coord!(lnlp.nlp, x[1:lnlp.nlp.meta.nvar], j[1:lnlp.nlp.meta.lin_nnzj])
    for i in 1:lnlp.meta.lin_nlift
        j[i+lnlp.nlp.meta.lin_nnzj] = -1
    end
    return j
end

function NLPModels.jac_nln_coord!(
    lnlp::LiftedNLPModel,
    x::AbstractVector,
    j::AbstractVector,
)
    @views jac_nln_coord!(lnlp.nlp, x[1:lnlp.nlp.meta.nvar], j[1:lnlp.nlp.meta.nln_nnzj])
    for i in 1:lnlp.meta.nln_nlift
        j[i+lnlp.nlp.meta.nln_nnzj] = -1
    end
    return j
end

function NLPModels.jprod!(
    lnlp::LiftedNLPModel,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jv[1:lnlp.meta.ncon] .= jac(lnlp, x) * v
    return Jv
end

function NLPModels.jprod_lin!(
    lnlp::LiftedNLPModel,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jv[1:lnlp.meta.nlin] .= jac_lin(lnlp, x) * v
    return Jv
end

function NLPModels.jprod_nln!(
    lnlp::LiftedNLPModel,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jv[1:lnlp.meta.nnln] .= jac_nln(lnlp, x) * v
    return Jv
end

function NLPModels.jtprod!(
    lnlp::LiftedNLPModel,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jtv[1:lnlp.meta.nvar] = jac(lnlp, x)' * v
    return Jtv
end

function NLPModels.jtprod_lin!(
    lnlp::LiftedNLPModel,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jtv[1:lnlp.meta.nvar] = jac_lin(lnlp, x)' * v
    return Jtv
end

function NLPModels.jtprod_nln!(
    lnlp::LiftedNLPModel,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jtv[1:lnlp.meta.nvar] = jac_nln(lnlp, x)' * v
    return Jtv
end

function NLPModels.hess_structure!(
    lnlp::LiftedNLPModel,
    rows::Vector{Int},
    cols::Vector{Int},
)
    return hess_structure!(lnlp.nlp, rows, cols)
end

function NLPModels.hess_coord!(
    lnlp::LiftedNLPModel{T, VT},
    x::AbstractVector{T},
    y::AbstractVector{T},
    H::AbstractVector{T};
    obj_weight::Real=one(T),
) where {T, VT}
    return @views hess_coord!(
        lnlp.nlp,
        x[1:lnlp.nlp.meta.nvar],
        y,
        H;
        obj_weight=obj_weight,
    )
end

function NLPModels.hprod!(
    lnlp::LiftedNLPModel{T, VT},
    x::AbstractVector{T},
    y::AbstractVector{T},
    v::AbstractVector{T},
    Hv::AbstractVector;
    obj_weight::Real=one(T),
) where {T, VT}
    @views hprod!(
        lnlp.nlp,
        x[1:lnlp.nlp.meta.nvar],
        y,
        v[1:lnlp.nlp.meta.nvar],
        Hv[1:lnlp.nlp.meta.nvar];
        obj_weight=obj_weight,
    )
    Hv[(lnlp.nlp.meta.nvar+1):lnlp.meta.nvar] .= 0
    return Hv
end
