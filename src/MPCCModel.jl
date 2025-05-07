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

    # Index Sets of complementarity variables
    ind_vcc1::IndexSet
    ind_vcc2::IndexSet

    # Index Sets of noncomplemntarity variables and constraints
    ind_x::IndexSet
    ind_c::IndexSet
end

struct MPCCModelConCon{T,VT} <: AbstractMPCCModel{T,VT}
    nlp::NLPModels.AbstractNLPModel{T, VT}

    # Index sets of complementarity constraints
    ind_ccc1::IndexSet
    ind_ccc2::IndexSet

    # Also indices

    # Precomputed indices of non-complementarity generic constraints
    ind_x::IndexSet
    ind_c::IndexSet
end

struct MPCCModelVarCon{T,VT} <: AbstractMPCCModel{T,VT}
    nlp::NLPModels.AbstractNLPModel{T, VT}
    # Indices of complementarities
    ind_vcc1::IndexSet
    ind_ccc2::IndexSet
    # Precomputed indices of non-complementarity variables and generic constraints
    ind_x::IndexSet
    ind_c::IndexSet
end

struct MPCCModelGeneric{T, VT} <: AbstractMPCCModel{T, VT}
    nlp::NLPModels.AbstractNLPModel{T, VT}
    ind_cc1::IndexSet # index of G in nlp or variable appearing in cc
    ind_cc2::IndexSet # index of H in nlp or variable appearing in cc
    cc_types::Vector{CCType} # VarCon, VarVar, ConCon

    ind_x::IndexSet
    ind_c::IndexSet
end

struct MPCCModelVerticalForm{T, VT} <: AbstractMPCCModel{T, VT}
    nlp::NLPModels.AbstractNLPModel{T, VT}

    # indices in the nlp that contain the lifted nonlinear constraints
    ind_cc1_lift::IndexSet
    ind_cc2_lift::IndexSet

    # indices in the nlp that contain the lifted comp variables
    ind_vcc1::IndexSet
    ind_vcc2::IndexSet
    ind_c::IndexSet
    ind_x::IndexSet
end

######################### Implementing NLPModels API #########################
NLPModels.obj(mpcc::AbstractMPCCModel, x::AbstractVector) = NLPModels.obj(mpcc.nlp, x)
NLPModels.grad!(mpcc::AbstractMPCCModel, x::AbstractVector, gx::AbstractVector) = NLPModels.grad!(mpcc.nlp, x, gx)
NLPModels.objgrad!(mpcc::AbstractMPCCModel, x::AbstractVector, g::AbstractVector) = NLPModels.objgrad!(mpcc.nlp, x, g)

# TODO(@anton) This is not currently allocationless,
# The only way to make it so in my view is to implement a custom Subarray which allows for "ignored" indices
# Perhaps there is another way however, I haven't found a good one yet.

function NLPModels.cons_lin!(mpcc::AbstractMPCCModel, x::AbstractVector, cx::AbstractVector)
    cons_lin!(mpcc.nlp, x, cx)
    deleteat!(cx, intersect(mpcc.ind_c, mpcc.nlp.lin))
    return cx
end

function NLPModels.cons_nln!(mpcc::AbstractMPCCModel, x::AbstractVector, cx::AbstractVector)
    cons_nln!(mpcc.nlp, x, cx)
    deleteat!(cx, intersect(mpcc.ind_c, mpcc.nlp.nln))
    return cx
end


function NLPModels.jac_lin_structure!(mpcc::AbstractMPCCModel, rows::Vector{int}, cols::Vector{int})
    jac_lin_structure!(mpcc, rows, cols) # get including complementarities

    # now do allocationless search through row,cols and remove
    ii = firstindex(rows);
    while ii ≤ length(rows)
        if rows[ii] ∈ mpcc.ind_ccc1 || rows[ii] ∈ mpcc.ind_ccc2
            # TODO(@anton) Is this even a good idea as this is quadratic. All this to save one memory alloc???
            deleteat!(rows, ii)
            deteteat!(cols, ii)
        end
        ii += 1
    end
    return rows, cols
end

function NLPModels.jac_nln_structure!(mpcc::AbstractMPCCModel, x::AbstractVector, cx::AbstractVector)
    jac_nln_structure!(mpcc, rows, cols) # get including complementarities

    # now do allocationless search through row,cols and remove
    ii = firstindex(rows);
    while ii ≤ length(rows)
        if rows[ii] ∈ mpcc.ind_ccc1 || rows[ii] ∈ mpcc.ind_ccc2
            # TODO(@anton) Is this even a good idea as this is quadratic. All this to save one memory alloc???
            deleteat!(rows, ii)
            deteteat!(cols, ii)
        end
        ii += 1
    end
    return rows, cols
end

function NLPModels.jac_lin_coord!(mpcc::AbstractMPCCModel, x::AbstractVector, cx::AbstractVector) error("not implemented") end
function NLPModels.jac_nln_coord!(mpcc::AbstractMPCCModel, x::AbstractVector, cx::AbstractVector) error("not implemented") end
function NLPModels.jprod_lin!(mpcc::AbstractMPCCModel, x::AbstractVector, cx::AbstractVector) error("not implemented") end
function NLPModels.jprod_nln!(mpcc::AbstractMPCCModel, x::AbstractVector, cx::AbstractVector) error("not implemented") end
function NLPModels.jtprod_lin!(mpcc::AbstractMPCCModel, x::AbstractVector, cx::AbstractVector) error("not implemented") end
function NLPModels.jtprod_nln!(mpcc::AbstractMPCCModel, x::AbstractVector, cx::AbstractVector) error("not implemented") end
function NLPModels.jth_hess_coord!(mpcc::AbstractMPCCModel, x::AbstractVector, cx::AbstractVector) error("not implemented") end
function NLPModels.jth_hprod!(mpcc::AbstractMPCCModel, x::AbstractVector, cx::AbstractVector) error("not implemented") end
function NLPModels.ghjvprod!(mpcc::AbstractMPCCModel, x::AbstractVector, cx::AbstractVector) error("not implemented") end
function NLPModels.hess_structure!(mpcc::AbstractMPCCModel, x::AbstractVector, cx::AbstractVector) error("not implemented") end
function NLPModels.hess_coord!(mpcc::AbstractMPCCModel, x::AbstractVector, cx::AbstractVector) error("not implemented") end
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
    if sym ∈ [:meta, :counters]
        getfield(rnlp.mpcc.nlp, sym)
    else
        getfield(rnlp, sym)
    end
end



# TODO(@anton) Add Core.show overload
