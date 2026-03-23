"""
  Lifted NLP model which applies lifting to given indices of the nonlinear constraints.
  Currently takes a parent model and a list of indicies of constraints that should be Lifted.
  The lifting variables then take on the bounds of the original
"""
######################### Metadata Definition #########################
struct LiftedNLPModelMeta{T, VT, MT <: AbstractNLPModelMeta{T, VT}} <:
       AbstractNLPModelMeta{T, VT}
    parent::MT

    ind_lift::IndexSet
    ind_lin_lift::IndexSet
    ind_nln_lift::IndexSet
    ind_lift_var::IndexSet
    ind_lin_lift_var::IndexSet
    ind_nln_lift_var::IndexSet
    nlift::Int
    lin_nlift::Int
    nln_nlift::Int
end

######################### LiftedNLPModel Definition #########################
# TODO(@anton) should this be a mutable struct
struct LiftedNLPModel{T, VT, NLP <: AbstractNLPModel{T, VT}, NMT} <: AbstractNLPModel{T, VT}
    nlp::NLP

    meta::LiftedNLPModelMeta{T, VT, NMT}
    counters::NLPModels.Counters
end

function LiftedNLPModel(nlp::AbstractNLPModel, ind_lift::IndexSet)
    # Get indicies for lin/nln
    # TODO(@anton) Perhaps warn if lifting linear constraints
    ind_lin_lift::IndexSet = [i for i in 1:get_nlin(nlp) if get_lin(nlp)[i] ∈ ind_lift]
    ind_nln_lift::IndexSet = [i for i in 1:get_nnln(nlp) if get_nln(nlp)[i] ∈ ind_lift]

    # number of lifting variables
    nlift = length(ind_lift)
    lin_nlift = length(ind_lin_lift)
    nln_nlift = length(ind_nln_lift)
    nvar = get_nvar(nlp) + nlift

    ind_lift_var = collect((get_nvar(nlp)+1):(get_nvar(nlp)+nlift))
    ind_lin_lift_var::IndexSet =
        [get_nvar(nlp)+i for i in 1:nlift if ind_lift[i] ∈ get_lin(nlp)]
    ind_nln_lift_var::IndexSet =
        [get_nvar(nlp)+i for i in 1:nlift if ind_lift[i] ∈ get_nln(nlp)]

    # add variable bounds for slacks and set initial value to the residual
    lvar = vcat(get_lvar(nlp), get_lcon(nlp)[ind_lift])
    uvar = vcat(get_uvar(nlp), get_ucon(nlp)[ind_lift])
    x0 = vcat(get_x0(nlp), NLPModels.cons(nlp, get_x0(nlp))[ind_lift])

    # Update the constraints to equality constraints.
    # TODO(@anton) also check if lifting equality constraints for some reason
    lcon = copy(get_lcon(nlp))
    lcon[ind_lift] .= 0
    ucon = copy(get_ucon(nlp))
    ucon[ind_lift] .= 0

    # Nonzeros for lifting variables in the jacobian
    nnzj = get_nnzj(nlp) + nlift
    lin_nnzj = get_lin_nnzj(nlp) + lin_nlift
    nln_nnzj = get_nln_nnzj(nlp) + nln_nlift

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

    return LiftedNLPModel(nlp, meta, nlp.counters)
end

######################### NLPModels API Implementation #########################
function NLPModels.obj(lnlp::LiftedNLPModel, x::AbstractVector)
    return NLPModels.obj(lnlp.nlp, view(x, 1:get_nvar(lnlp.nlp)))
end
function NLPModels.grad!(lnlp::LiftedNLPModel, x::AbstractVector, gx::AbstractVector)
    @views NLPModels.grad!(lnlp.nlp, x[1:get_nvar(lnlp.nlp)], gx[1:get_nvar(lnlp.nlp)])
    gx[(get_nvar(lnlp.nlp)+1):get_nvar(lnlp)] .= 0
    return gx
end
function NLPModels.objgrad!(lnlp::LiftedNLPModel, x::AbstractVector, g::AbstractVector)
    return NLPModels.objgrad!(lnlp.nlp, view(x, 1:get_nvar(lnlp.nlp)), g)
end

function NLPModels.cons!(lnlp::LiftedNLPModel, x::AbstractVector, cx::AbstractVector)
    cons!(lnlp.nlp, view(x, 1:get_nvar(lnlp.nlp)), cx)
    @views cx[get_ind_lift(lnlp)] .-= x[get_ind_lift_var(lnlp)]
    return cx
end

function NLPModels.cons_lin!(lnlp::LiftedNLPModel, x::AbstractVector, cx::AbstractVector)
    cons_lin!(lnlp.nlp, view(x, 1:get_nvar(lnlp.nlp)), cx)
    @views cx[get_ind_lin_lift(lnlp)] .-= x[get_ind_lin_lift_var(lnlp)]
    return cx
end

function NLPModels.cons_nln!(lnlp::LiftedNLPModel, x::AbstractVector, cx::AbstractVector)
    cons_nln!(lnlp.nlp, view(x, 1:get_nvar(lnlp.nlp)), cx)
    @views cx[get_ind_nln_lift(lnlp)] .-= x[get_ind_nln_lift_var(lnlp)]
    return cx
end

function NLPModels.jac_structure!(
    lnlp::LiftedNLPModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    @views jac_structure!(lnlp.nlp, rows[1:get_nnzj(lnlp.nlp)], cols[1:get_nnzj(lnlp.nlp)]) # get including complementarities

    for i in 1:get_nlift(lnlp)
        rows[i+get_nnzj(lnlp.nlp)] = get_ind_lift(lnlp)[i]
        cols[i+get_nnzj(lnlp.nlp)] = get_ind_lift_var(lnlp)[i]
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
        rows[1:get_lin_nnzj(lnlp.nlp)],
        cols[1:get_lin_nnzj(lnlp.nlp)],
    ) # get including complementarities

    for i in 1:get_lin_nlift(lnlp)
        rows[i+get_lin_nnzj(lnlp.nlp)] = get_ind_lin_lift(lnlp)[i]
        cols[i+get_lin_nnzj(lnlp.nlp)] = get_ind_lin_lift_var(lnlp)[i]
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
        rows[1:get_nln_nnzj(lnlp.nlp)],
        cols[1:get_nln_nnzj(lnlp.nlp)],
    )

    for i in 1:get_nln_nlift(lnlp)
        rows[i+get_nln_nnzj(lnlp.nlp)] = get_ind_nln_lift(lnlp)[i]
        cols[i+get_nln_nnzj(lnlp.nlp)] = get_ind_nln_lift_var(lnlp)[i]
    end
    return rows, cols
end

function NLPModels.jac_coord!(lnlp::LiftedNLPModel, x::AbstractVector, j::AbstractVector)
    @views jac_coord!(lnlp.nlp, x[1:get_nvar(lnlp.nlp)], j[1:get_nnzj(lnlp.nlp)])
    j[(get_nnzj(lnlp.nlp)+1):(get_nnzj(lnlp.nlp)+get_nlift(lnlp))] .= -1
    return j
end

function NLPModels.jac_lin_coord!(
    lnlp::LiftedNLPModel,
    x::AbstractVector,
    j::AbstractVector,
)
    @views jac_lin_coord!(lnlp.nlp, x[1:get_nvar(lnlp.nlp)], j[1:get_lin_nnzj(lnlp.nlp)])
    j[(get_lin_nnzj(lnlp.nlp)+1):(get_lin_nnzj(lnlp.nlp)+get_lin_nlift(lnlp))] .= -1
    return j
end

function NLPModels.jac_nln_coord!(
    lnlp::LiftedNLPModel,
    x::AbstractVector,
    j::AbstractVector,
)
    @views jac_nln_coord!(lnlp.nlp, x[1:get_nvar(lnlp.nlp)], j[1:get_nln_nnzj(lnlp.nlp)])
    j[(get_nln_nnzj(lnlp.nlp)+1):(get_nln_nnzj(lnlp.nlp)+get_nln_nlift(lnlp))] .= -1
    return j
end

function NLPModels.jprod!(
    lnlp::LiftedNLPModel,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jv[1:get_ncon(lnlp)] .= jac(lnlp, x) * v
    return Jv
end

function NLPModels.jprod_lin!(
    lnlp::LiftedNLPModel,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jv[1:get_nlin(lnlp)] .= jac_lin(lnlp, x) * v
    return Jv
end

function NLPModels.jprod_nln!(
    lnlp::LiftedNLPModel,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jv[1:get_nnln(lnlp)] .= jac_nln(lnlp, x) * v
    return Jv
end

function NLPModels.jtprod!(
    lnlp::LiftedNLPModel,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jtv[1:get_nvar(lnlp)] = jac(lnlp, x)' * v
    return Jtv
end

function NLPModels.jtprod_lin!(
    lnlp::LiftedNLPModel,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jtv[1:get_nvar(lnlp)] = jac_lin(lnlp, x)' * v
    return Jtv
end

function NLPModels.jtprod_nln!(
    lnlp::LiftedNLPModel,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jtv[1:get_nvar(lnlp)] = jac_nln(lnlp, x)' * v
    return Jtv
end

function NLPModels.hess_structure!(
    lnlp::LiftedNLPModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
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
        x[1:get_nvar(lnlp.nlp)],
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
        x[1:get_nvar(lnlp.nlp)],
        y,
        v[1:get_nvar(lnlp.nlp)],
        Hv[1:get_nvar(lnlp.nlp)];
        obj_weight=obj_weight,
    )
    Hv[(get_nvar(lnlp.nlp)+1):get_nvar(lnlp)] .= 0
    return Hv
end

######################## Typed gets #######################
for field in fieldnames(NLPModelMeta) ∪ fieldnames(LiftedNLPModelMeta)
    meth = Symbol("get_", field)
    if field ∈ fieldnames(NLPModelMeta)
        if field ∈ fieldnames(LiftedNLPModelMeta)
            @eval NLPModels.$meth(meta::LiftedNLPModelMeta) =
                getproperty(meta, $(QuoteNode(field)))
        else
            @eval NLPModels.$meth(meta::LiftedNLPModelMeta) = $meth(meta.parent)
        end
        @eval NLPModels.$meth(lnlp::LiftedNLPModel) = $meth(lnlp.meta)
    else
        @eval begin
            @doc """
                  $($meth)(nlp)
                  $($meth)(meta)
                  Return the value $($(QuoteNode(field))) from meta or nlp.meta.
      """
            $meth(meta::LiftedNLPModelMeta) = getproperty(meta, $(QuoteNode(field)))
        end
        @eval $meth(lnlp::LiftedNLPModel) = $meth(lnlp.meta)
        @eval export $meth
    end
end
