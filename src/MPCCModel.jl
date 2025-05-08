"""
  Abstract type for problems in the form:
  min f(w)
  s.t. lbc ≤ c(w) ≤ ubc
       0 ≤ G(w) ⟂ H(w) ≥ 0
"""
abstract type AbstractMPCCModel{T, VT} <: NLPModels.AbstractNLPModel{T, VT} end

# TODO(@anton) For a prototype this is a bit of a hack to account for not having meta and counters
#              In principle we want to have a MPCCModelMeta and MPCCModelCounters?
function Base.getproperty(mpcc::AbstractMPCCModel, sym::Symbol)
    if sym ∈ [:meta, :counters]
        getfield(mpcc.nlp, sym)
    else
        getfield(mpcc, sym)
    end
end

function Base.show(io::IO, mpcc::AbstractMPCCModel)
    show_header(io, mpcc)
    show(io, mpcc.nlp.meta)
    show(io, mpcc.nlp.counters)
end

######################### Type Definitions #########################
# Typealias for index set
# TODO(@anton) Perhaps this should be Set{Int}
IndexSet = Vector{Int}

abstract type CCType end
struct VarCon <: CCType end
struct VarVar <: CCType end
struct ConCon <: CCType end


######################### MPCC Types #########################
struct MPCCModelVarVar{T,VT} <: AbstractMPCCModel{T,VT}
    nlp::NLPModels.AbstractNLPModel{T, VT}

    meta::MPCCModelMeta{T, VT}
end

# Constructor
function MPCCModelVarVar(nlp::AbstractNLPModel, ind_vcc1::IndexSet, ind_vcc2::IndexSet)
    # compute non-complementarity variables/constraints
    ncc = length(ind_vcc1)
    ind_x = setminus(1:nlp.meta.nvar, union(ind_vcc1, ind_vcc2))
    ind_c = collect(1:nlp.meta.ncon)

    # compute jacobian structure indexset
    ind_j_lin_triplets = collect(1:nlp.meta.lin_nnzj)
    ind_j_nln_triplets = collect(1:nlp.meta.nln_nnzj)

    # UNUSED
    ind_ccc1 = IndexSet();
    ind_ccc2 = IndexSet();
    ind_cc1 = ind_vcc1;
    ind_cc2 = ind_vcc2;
    cc_types = fill!(Vector{CCType}(undef,ncc), VarVar)


    meta = MPCCModelMeta(Ref(nlp.meta), ncc,
                         ind_vcc1, ind_vcc2,
                         ind_ccc, ind_ccc2,
                         ind_cc1, ind_cc2,
                         cc_types,
                         ind_x, ind_c,
                         ind_j_lin_triplets, ind_j_nln_triplets)

    MPCCModelVarVar(nlp, meta)
end


struct MPCCModelConCon{T,VT} <: AbstractMPCCModel{T,VT}
    nlp::NLPModels.AbstractNLPModel{T, VT}

    meta::MPCCModelMeta{T, VT}
end

# Constructor
function MPCCModelConCon(nlp::AbstractNLPModel, ind_ccc1::IndexSet, ind_ccc2::IndexSet)
    # compute non-complementarity variables/constraints
    ncc = length(ind_ccc1)
    ind_x = collect(1:nlp.meta.nvar)
    ind_c = setminus(1:nlp.meta.ncon, union(ind_ccc1, ind_ccc2))

    # compute jacobian structure indexset
    lin_rows, lin_cols = NLPModels.jac_lin_structure(nlp)
    nln_rows, nln_cols = NLPModels.jac_nln_structure(nlp)
    ind_j_lin_triplets = findall(x->!((x∈ind_ccc1) || (x∈ind_ccc2)),lin_rows)
    ind_j_nln_triplets = findall(x->!((x∈ind_ccc1) || (x∈ind_ccc2)),nln_rows)

    # UNUSED
    ind_vcc1 = IndexSet();
    ind_vcc2 = IndexSet();
    ind_cc1 = ind_ccc1;
    ind_cc2 = ind_ccc2;
    cc_types = fill!(Vector{CCType}(undef,ncc), ConCon)


    meta = MPCCModelMeta(Ref(nlp.meta), ncc,
                         ind_vcc1, ind_vcc2,
                         ind_ccc, ind_ccc2,
                         ind_cc1, ind_cc2,
                         cc_types,
                         ind_x, ind_c,
                         ind_j_lin_triplets, ind_j_nln_triplets)

    MPCCModelConCon(nlp, meta)
end

struct MPCCModelVarCon{T,VT} <: AbstractMPCCModel{T,VT}
    nlp::NLPModels.AbstractNLPModel{T, VT}

    meta::MPCCModelMeta{T, VT}
end

# Constructor
function MPCCModelVarCon(nlp::AbstractNLPModel, ind_vcc1::IndexSet, ind_ccc2::IndexSet)
    # compute non-complementarity variables/constraints
    ncc = length(ind_vcc1)
    ind_x = setminus(1:nlp.meta.nvar, ind_vcc1)
    ind_c = setminus(1:nlp.meta.ncon, ind_ccc2)

    # compute jacobian structure indexset
    lin_rows, lin_cols = NLPModels.jac_lin_structure(nlp)
    nln_rows, nln_cols = NLPModels.jac_nln_structure(nlp)
    ind_j_lin_triplets = findall(x->!(x∈ind_ccc2),lin_rows)
    ind_j_nln_triplets = findall(x->!(x∈ind_ccc2),nln_rows)

    # UNUSED
    ind_ccc1 = IndexSet();
    ind_vcc2 = IndexSet();
    ind_cc1 = ind_vcc1;
    ind_cc2 = ind_ccc2;
    cc_types = fill!(Vector{CCType}(undef,ncc), VarCon)


    meta = MPCCModelMeta(Ref(nlp.meta), ncc,
                         ind_vcc1, ind_vcc2,
                         ind_ccc, ind_ccc2,
                         ind_cc1, ind_cc2,
                         cc_types,
                         ind_x, ind_c,
                         ind_j_lin_triplets, ind_j_nln_triplets)

    MPCCModelVarCon(nlp, meta)
end

struct MPCCModelGeneric{T, VT} <: AbstractMPCCModel{T, VT}
    nlp::NLPModels.AbstractNLPModel{T, VT}

    meta::MPCCModelMeta{T, VT}
end

struct MPCCModelVerticalForm{T, VT} <: AbstractMPCCModel{T, VT}
    nlp::NLPModels.AbstractNLPModel{T, VT}

    meta::MPCCModelMeta{T, VT} # TODO(@anton) this needs its own metadata class
end

######################### Implementing NLPModels API #########################
NLPModels.obj(mpcc::AbstractMPCCModel, x::AbstractVector) = NLPModels.obj(mpcc.nlp, x)
NLPModels.grad!(mpcc::AbstractMPCCModel, x::AbstractVector, gx::AbstractVector) = NLPModels.grad!(mpcc.nlp, x, gx)
NLPModels.objgrad!(mpcc::AbstractMPCCModel, x::AbstractVector, g::AbstractVector) = NLPModels.objgrad!(mpcc.nlp, x, g)

# TODO(@anton) This is not currently allocationless and requires initial arrays
# which are the size of nlp.meta.ncon rather than the "true" ncon which is nlp.meta.ncon - ncc,
# and these are then shrunk to size (hence avoiding allocation).
# The only way to make it so in my view is to implement a custom Subarray which allows for "ignored" indices
# Perhaps there is another way however, I haven't found a good one yet.

function NLPModels.cons_lin!(mpcc::AbstractMPCCModel, x::AbstractVector, cx::AbstractVector)
    cons_lin!(mpcc.nlp, x, cx)
    keepat!(cx, intersect(mpcc.meta.ind_c, mpcc.nlp.lin))
    return cx
end

function NLPModels.cons_nln!(mpcc::AbstractMPCCModel, x::AbstractVector, cx::AbstractVector)
    cons_nln!(mpcc.nlp, x, cx)
    keepat!(cx, intersect(mpcc.meta.ind_c, mpcc.nlp.nln))
    return cx
end


function NLPModels.jac_lin_structure!(mpcc::AbstractMPCCModel, rows::Vector{Int}, cols::Vector{Int})
    jac_lin_structure!(mpcc.nlp, rows, cols) # get including complementarities

    keepat!(rows, mpcc.meta.ind_j_lin_structure)
    keepat!(cols, mpcc.meta.ind_j_lin_structure)
    return rows, cols
end

function NLPModels.jac_nln_structure!(mpcc::AbstractMPCCModel, rows::Vector{Int}, cols::Vector{Int})
    jac_nln_structure!(mpcc.nlp, rows, cols) # get including complementarities

    keepat!(rows, mpcc.meta.ind_j_nln_structure)
    keepat!(cols, mpcc.meta.ind_j_nln_structure)
    return rows, cols
end

function NLPModels.jac_lin_coord!(mpcc::AbstractMPCCModel, x::AbstractVector, j::AbstractVector)
    jac_lin_coord!(mpcc.nlp, x, j)

    keepat!(j, mpcc.meta.ind_j_lin_structure)
    return j
end
function NLPModels.jac_nln_coord!(mpcc::AbstractMPCCModel, x::AbstractVector, j::AbstractVector)
    jac_nln_coord!(mpcc.nlp, x, j)

    keepat!(j, mpcc.meta.ind_j_nln_structure)
    return j
end

function NLPModels.jprod_lin!(mpcc::AbstractMPCCModel, x::AbstractVector, v::AbstractVector, jv::AbstractVector)
    jprod_lin!(mpcc.nlp, x, v, jv)

    keepat!(jv, mpcc.meta.ind_c)
    return jv
end

function NLPModels.jprod_nln!(mpcc::AbstractMPCCModel, x::AbstractVector, v::AbstractVector, jv::AbstractVector)
    jprod_lin!(mpcc.nlp, x, v, jv)

    keepat!(jv, mpcc.meta.ind_c)
    return jv
end

function NLPModels.jtprod_lin!(mpcc::AbstractMPCCModel, x::AbstractVector, v::AbstractVector, jtv::AbstractVector)
    error("not implemented")
    # TODO(@anton) this is not correct
    v[setminus(1:mpcc.meta.ncon, mpcc.meta.ind_c)] = 0 # TODO(@anton) this is not a great solution
    jtprod_lin!(mpcc.nlp, x, v, jtv)

    return jtv
end

function NLPModels.jtprod_nln!(mpcc::AbstractMPCCModel, x::AbstractVector, v::AbstractVector, jtv::AbstractVector)
    error("not implemented")
    # TODO(@anton) this is not correct
    jprod_lin!(mpcc.nlp, x, v, jv)

    keepat!(jv, mpcc.meta.ind_c)
    return jv
end

function NLPModels.hess_structure!(mpcc::AbstractMPCCModel, rows::Vector{Int}, cols::Vector{Int})
    return hess_structure!(mpcc.nlp, rows, cols)
end
function NLPModels.hess_coord!(mpcc::AbstractMPCCModel{T,S}, x::AbstractVector{T}, y::AbstractVector{T}, Hv::AbstractVector, obj_weight::Real = one(T))
    return hess_coord!(mpcc.nlp, x, y, Hv, obj_weight)
end
function NLPModels.hprod!(mpcc::AbstractMPCCModel, x::AbstractVector, cx::AbstractVector) error("not implemented") end
######################### MPCCModelVarVar #########################
######################### MPCCModelVarCon #########################
######################### MPCCModelConCon #########################
######################### MPCCModelVerticalForm #########################
# Note: This requires special handling for all values because you also have lifted variables, which are not represented in nlp.

######################### Vertical Form Conversions #########################
function MPCCModelVerticalForm(mpcc::MPCCModelGeneric)
    # TODO
    error("unimplemented")
    MPCCModelVarVar(nlp, ind_vcc1, ind_vcc2, ind_x);
end

function MPCCModelVerticalForm(mpcc::MPCCModelConCon)
    # NOTE: We assume that we have to lift ALL Nonlinear complementarity constraints

    ncc = length(mpcc.ind_ccc1)
    nw = mpcc.nlp.meta.nvar
    ind_cc1_lift = mpcc.ind_ccc1
    ind_cc2_lift = mpcc.ind_ccc2
    ind_vcc1 = collect(nw+1:nw+ncc)
    ind_vcc2 = collect(nw+ncc+1:nw+2*ncc)
    ind_x = collect(1:nw)
    MPCCModelVerticalForm(nlp, ind_cc1_lift, ind_cc2_lift, ind_vcc1, ind_vcc2, ind_x, mpcc.ind_c);
end

function MPCCModelVerticalForm(mpcc::MPCCModelVarCon)
    ncc = length(mpcc.ind_vcc1)
    nw = mpcc.nlp.meta.nvar
    ind_vcc1 = mpcc.ind_vcc1
    ind_vcc2 = collect(nw+1:nw+ncc)
    ind_x = collect(1:nw)
    MPCCModelVerticalForm(nlp, [], mpcc.ind_ccc2, ind_vcc1, ind_vcc2, ind_x);
end

"""
  A noop because the mpcc is already in vertical form
"""
function MPCCModelVerticalForm(mpcc::MPCCModelVarVar)
    return MPCCModelVerticalForm(mpcc, Int[], Int[], mpcc.ind_vcc1, mpcc.ind_vcc2, mpcc.ind_x, mpcc.ind_c)
end

######################### Scholtes Relaxation #########################
struct ScholtesRelaxation{T, VT} <: NLPModels.AbstractNLPModel{T, VT}
    mpcc::AbstractMPCCModel{T, VT}
    𝜎::Ref{T}
end

function Base.getproperty(rnlp::ScholtesRelaxation, sym::Symbol)
    if sym ∈ [:counters]
        getfield(rnlp.mpcc.nlp, sym)
    else
        getfield(rnlp, sym)
    end
end



# TODO(@anton) Add Core.show overload
