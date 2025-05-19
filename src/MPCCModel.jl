"""
  Abstract type for problems in the form:
  min f(w)
  s.t. lbc ≤ c(w) ≤ ubc
       0 ≤ G(w) ⟂ H(w) ≥ 0
"""
abstract type AbstractMPCCModel{T, VT} end

# TODO(@anton) For a prototype this is a bit of a hack to account for not having meta and counters
#              In principle we want to have a MPCCModelMeta and MPCCModelCounters?
function Base.getproperty(mpcc::AbstractMPCCModel, sym::Symbol)
    if sym ∈ [:counters]
        getfield(mpcc.nlp, sym)
    else
        getfield(mpcc, sym)
    end
end

function Base.show(io::IO, mpcc::AbstractMPCCModel)
    show_header(io, mpcc)
    show(io, mpcc.nlp.meta)
    return show(io, mpcc.nlp.counters)
end

struct MPCCModel{T, VT} <: AbstractMPCCModel{T, VT}
    nlp::NLPModels.AbstractNLPModel{T, VT}
    meta::MPCCModelMeta{T, VT}
end

######################### Helper functions for MPCCModel #########################
function is_vertical(mpcc::MPCCModel)
    return all(map((x)->isa(x, VarVar), mpcc.meta.cc_types))
end

######################### MPCC Types #########################
# Constructor
function MPCCModelVarVar(nlp::AbstractNLPModel, ind_vcc1::IndexSet, ind_vcc2::IndexSet)
    # compute sizes
    ncc = length(ind_vcc1)
    ncon = nlp.meta.ncon
    nlin = nlp.meta.nlin
    nnln = nlp.meta.nnln

    # compute non-complementarity variables/constraints
    ind_x = setdiff(1:nlp.meta.nvar, union(ind_vcc1, ind_vcc2))
    ind_c = collect(1:nlp.meta.ncon)

    # compute jacobian structure indexset
    ind_j_triplets = collect(1:nlp.meta.nnzj)
    ind_j_lin_triplets = collect(1:nlp.meta.lin_nnzj)
    ind_j_nln_triplets = collect(1:nlp.meta.nln_nnzj)
    ind_j_lin_row_map = Dict(zip(1:nlin, 1:nlin))
    ind_j_nln_row_map = Dict(zip(1:nnln, 1:nnln))

    ind_j_comp_left_triplets::IndexSet = []
    ind_j_comp_right_triplets::IndexSet = []
    ind_j_comp_left_row_map = Dict{Int, Int}()
    ind_j_comp_right_row_map = Dict{Int, Int}()
    # compute nln and lin index sets
    lin = nlp.meta.lin
    nln = nlp.meta.nln
    nlin = length(lin)
    nnln = length(nln)
    c_lin = collect(1:nlin)
    c_nln = collect(1:nnln)
    cc_l::IndexSet = []
    cc_r::IndexSet = []

    # Complementarity Constraints
    ind_cc1 = ind_vcc1
    ind_cc2 = ind_vcc2
    cc_types = fill!(Vector{CCType}(undef, ncc), VarVar())

    # nnzj updates:
    nnzj = nlp.meta.nnzj
    lin_nnzj = nlp.meta.lin_nnzj
    nln_nnzj = nlp.meta.nln_nnzj
    comp_left_nnzj = ncc
    comp_right_nnzj = ncc

    meta = MPCCModelMeta(
        Ref(nlp.meta),
        ncc,
        ncon,
        nlin,
        nnln,
        nnzj,
        lin_nnzj,
        nln_nnzj,
        comp_left_nnzj,
        comp_right_nnzj,
        lin,
        nln,
        c_lin,
        c_nln,
        cc_l,
        cc_r,
        ind_cc1,
        ind_cc2,
        cc_types,
        ind_x,
        ind_c,
        ind_j_triplets,
        ind_j_lin_triplets,
        ind_j_nln_triplets,
        ind_j_comp_left_triplets,
        ind_j_comp_right_triplets,
        ind_j_comp_left_row_map,
        ind_j_comp_right_row_map,
        ind_j_lin_row_map,
        ind_j_nln_row_map,
    )

    return MPCCModel(nlp, meta)
end

# Constructor
function MPCCModelConCon(nlp::AbstractNLPModel, ind_ccc1::IndexSet, ind_ccc2::IndexSet)
    # compute sizes
    ncc = length(ind_ccc1)
    ncon = nlp.meta.ncon - 2*ncc

    # compute non-complementarity variables/constraints
    ind_x = collect(1:nlp.meta.nvar)
    ind_c = setdiff(1:nlp.meta.ncon, union(ind_ccc1, ind_ccc2))

    # compute jacobian structure indexset
    rows, cols = NLPModels.jac_structure(nlp)
    ind_j_triplets = findall(x->x ∈ ind_c, rows)
    if hasmethod(jac_lin_structure!, (typeof(nlp), IndexSet, IndexSet))
        lin_rows, lin_cols = NLPModels.jac_lin_structure(nlp)
        nln_rows, nln_cols = NLPModels.jac_nln_structure(nlp)
        for i in 1:nlp.meta.nlin
            lin_rows[i] += count(x < nlp.meta.lin[lin_rows[i]] for x in nlp.meta.nln)
        end
        for i in 1:nlp.meta.nnln
            nln_rows[i] += count(x < nlp.meta.nln[nln_rows[i]] for x in nlp.meta.lin)
        end
        ind_j_lin_triplets = findall(x->!((x∈ind_ccc1) || (x∈ind_ccc2)), lin_rows)
        ind_j_nln_triplets = findall(x->!((x∈ind_ccc1) || (x∈ind_ccc2)), nln_rows)

        # compute nln and lin index sets
        lin = intersect(nlp.meta.lin, ind_c)
        nln = intersect(nlp.meta.nln, ind_c)
        nlin = length(lin)
        nnln = length(nln)
        c_lin = [i for i in 1:nlin if nlp.meta.lin[i] ∈ ind_c]
        cc_lin = [i for i in 1:nlin if nlp.meta.lin[i] ∉ ind_c]
        c_nln = [i for i in 1:nnln if nlp.meta.nln[i] ∈ ind_c]
        cc_nln = [i for i in 1:nnln if nlp.meta.nln[i] ∉ ind_c]

        ind_j_lin_row_map =
            Dict((i, i-count([x < j for x in cc_lin])) for i in 1:nlp.meta.nlin)
        ind_j_nln_row_map =
            Dict((i, i-count([x < j for x in cc_nln])) for i in 1:nlp.meta.nnln)
    else
        @warn "NLPModel with which this MPCC is created doesn't support linear API"
        ind_j_lin_triplets::IndexSet = []
        ind_j_nln_triplets = ind_j_triplets
        lin::IndexSet = []
        nln = nlp.meta.nln
        nlin = nlp.meta.nlin
        nnln = nlp.meta.nnln
        c_lin::IndexSet = []
        cc_lin::IndexSet = []
        c_nln = [i for i in 1:nnln if nlp.meta.nln[i] ∈ ind_c]
        cc_nln = [i for i in 1:nnln if nlp.meta.nln[i] ∉ ind_c]

        ind_j_lin_row_map = Dict{Int, Int}()
        ind_j_nln_row_map::Dict{Int, Int} =
            Dict((i, i-count([x < j for x in cc_nln])) for i in 1:nlp.meta.nnln)
    end
    ind_j_comp_left_triplets = findall(x->x∈ind_ccc1, rows);
    ind_j_comp_right_triplets = findall(x->x∈ind_ccc2, rows);
    ind_j_comp_left_row_map = Dict{Int, Int}(zip(ind_ccc1, 1:ncc))
    ind_j_comp_right_row_map = Dict{Int, Int}(zip(ind_ccc2, 1:ncc))

    # Complementarity Constraints
    ind_cc1 = ind_ccc1;
    ind_cc2 = ind_ccc2;
    cc_types = fill!(Vector{CCType}(undef, ncc), ConCon())
    cc_l = [i for i in 1:nlp.meta.ncon if i ∈ ind_cc1]
    cc_r = [i for i in 1:nlp.meta.ncon if i ∈ ind_cc2]

    # nnzj updates:
    lin_nnzj = length(ind_j_lin_triplets)
    nln_nnzj = length(ind_j_nln_triplets)
    nnzj = lin_nnzj + nln_nnzj
    comp_left_nnzj = length(ind_j_comp_left_triplets)
    comp_right_nnzj = length(ind_j_comp_right_triplets)

    meta = MPCCModelMeta(
        Ref(nlp.meta),
        ncc,
        ncon,
        nlin,
        nnln,
        nnzj,
        lin_nnzj,
        nln_nnzj,
        comp_left_nnzj,
        comp_right_nnzj,
        lin,
        nln,
        c_lin,
        c_nln,
        cc_l,
        cc_r,
        ind_cc1,
        ind_cc2,
        cc_types,
        ind_x,
        ind_c,
        ind_j_triplets,
        ind_j_lin_triplets,
        ind_j_nln_triplets,
        ind_j_comp_left_triplets,
        ind_j_comp_right_triplets,
        ind_j_comp_left_row_map,
        ind_j_comp_right_row_map,
        ind_j_lin_row_map,
        ind_j_nln_row_map,
    )

    return MPCCModel(nlp, meta)
end

# Constructor
function MPCCModelVarCon(nlp::AbstractNLPModel, ind_vcc1::IndexSet, ind_ccc2::IndexSet)
    # compute sizes
    ncc = length(ind_vcc1)
    ncon = nlp.meta.ncon - ncc
    # compute non-complementarity variables/constraints
    ind_x = setdiff(1:nlp.meta.nvar, ind_vcc1)
    ind_c = setdiff(1:nlp.meta.ncon, ind_ccc2)

    # compute jacobian structure indexset
    rows, cols = NLPModels.jac_structure(nlp)
    ind_j_triplets = findall(x->x ∈ ind_c, rows)
    if hasmethod(jac_lin_structure!, (typeof(nlp), IndexSet, IndexSet))
        lin_rows, lin_cols = NLPModels.jac_lin_structure(nlp)
        nln_rows, nln_cols = NLPModels.jac_nln_structure(nlp)
        # Convert to true row numbers
        for i in 1:nlp.meta.nlin
            lin_rows[i] += count(x < nlp.meta.lin[lin_rows[i]] for x in nlp.meta.nln)
        end
        for i in 1:nlp.meta.nnln
            nln_rows[i] += count(x < nlp.meta.nln[nln_rows[i]] for x in nlp.meta.lin)
        end
        # Keep only the "correct" indices
        ind_j_lin_triplets = findall(x->!(x∈ind_ccc2), lin_rows)
        ind_j_nln_triplets = findall(x->!(x∈ind_ccc2), nln_rows)

        # compute nln and lin index sets
        lin = intersect(nlp.meta.lin, ind_c)
        nln = intersect(nlp.meta.nln, ind_c)
        nlin = length(lin)
        nnln = length(nln)
        c_lin = [i for i in 1:nlin if nlp.meta.lin[i] ∈ ind_c]
        cc_lin = [i for i in 1:nlin if nlp.meta.lin[i] ∉ ind_c]
        c_nln = [i for i in 1:nnln if nlp.meta.nln[i] ∈ ind_c]
        cc_nln = [i for i in 1:nnln if nlp.meta.nln[i] ∉ ind_c]

        ind_j_lin_row_map =
            Dict((i, i-count([x < j for x in cc_lin])) for i in 1:nlp.meta.nlin)
        ind_j_nln_row_map =
            Dict((i, i-count([x < j for x in cc_nln])) for i in 1:nlp.meta.nnln)
    else
        @warn "NLPModel with which this MPCC is created doesn't support linear API"
        ind_j_lin_triplets::IndexSet = []
        ind_j_nln_triplets = ind_j_triplets
        lin::IndexSet = []
        nln = nlp.meta.nln
        nlin = nlp.meta.nlin
        nnln = nlp.meta.nnln
        c_lin::IndexSet = []
        cc_lin::IndexSet = []
        c_nln = [i for i in 1:nnln if nlp.meta.nln[i] ∈ ind_c]
        cc_nln = [i for i in 1:nnln if nlp.meta.nln[i] ∉ ind_c]

        ind_j_lin_row_map = Dict{Int, Int}()
        ind_j_nln_row_map::Dict{Int, Int} =
            Dict((i, i-count([x < j for x in cc_nln])) for i in 1:nlp.meta.nnln)
    end
    ind_j_comp_left_triplets::IndexSet = [];
    ind_j_comp_right_triplets = findall(x->x∈ind_ccc2, rows);
    ind_j_comp_left_row_map = Dict{Int, Int}()
    ind_j_comp_right_row_map = Dict{Int, Int}(zip(ind_ccc2, 1:ncc))

    # UNUSED
    ind_cc1 = ind_vcc1;
    ind_cc2 = ind_ccc2;
    cc_types = fill!(Vector{CCType}(undef, ncc), VarCon())
    cc_l::IndexSet = [];
    cc_r = [i for i in 1:nlp.meta.ncon if i ∈ ind_cc2]

    # nnzj updates:
    lin_nnzj = length(ind_j_lin_triplets)
    nln_nnzj = length(ind_j_nln_triplets)
    nnzj = lin_nnzj + nln_nnzj
    comp_left_nnzj = ncc
    comp_right_nnzj = length(ind_j_comp_right_triplets)

    meta = MPCCModelMeta(
        Ref(nlp.meta),
        ncc,
        ncon,
        nlin,
        nnln,
        nnzj,
        lin_nnzj,
        nln_nnzj,
        comp_left_nnzj,
        comp_right_nnzj,
        lin,
        nln,
        c_lin,
        c_nln,
        cc_l,
        cc_r,
        ind_cc1,
        ind_cc2,
        cc_types,
        ind_x,
        ind_c,
        ind_j_triplets,
        ind_j_lin_triplets,
        ind_j_nln_triplets,
        ind_j_comp_left_triplets,
        ind_j_comp_right_triplets,
        ind_j_comp_left_row_map,
        ind_j_comp_right_row_map,
        ind_j_lin_row_map,
        ind_j_nln_row_map,
    )

    return MPCCModel(nlp, meta)
end

######################### Implementing NLPModels API #########################
NLPModels.obj(mpcc::AbstractMPCCModel, x::AbstractVector) = NLPModels.obj(mpcc.nlp, x)
function NLPModels.grad!(mpcc::AbstractMPCCModel, x::AbstractVector, gx::AbstractVector)
    return NLPModels.grad!(mpcc.nlp, x, gx)
end
function NLPModels.objgrad!(mpcc::AbstractMPCCModel, x::AbstractVector, g::AbstractVector)
    return NLPModels.objgrad!(mpcc.nlp, x, g)
end

function NLPModels.cons!(mpcc::AbstractMPCCModel, x::AbstractVector, cx::AbstractVector)
    # TODO(@anton) do we want to use a SFINAE style definition only for nlpmodel types which do not support
    #              the linear interface
    mcx = MappedVector(cx, mpcc.meta.ind_c, mpcc.nlp.meta.ncon)
    cons!(mpcc.nlp, x, mcx)
    return cx
end

function NLPModels.cons_lin!(mpcc::AbstractMPCCModel, x::AbstractVector, cx::AbstractVector)
    mcx = MappedVector(cx, mpcc.meta.c_lin, mpcc.nlp.meta.nlin)
    cons_lin!(mpcc.nlp, x, mcx)
    return cx
end

function NLPModels.cons_nln!(mpcc::AbstractMPCCModel, x::AbstractVector, cx::AbstractVector)
    mcx = MappedVector(cx, mpcc.meta.c_nln, mpcc.nlp.meta.nnln)
    cons_nln!(mpcc.nlp, x, mcx)
    return cx
end

function NLPModels.jac_structure!(
    mpcc::AbstractMPCCModel,
    rows::AbstractVector{Int},
    cols::AbstractVector{Int},
)
    mrows = MappedVector(rows, mpcc.meta.ind_j_triplets, mpcc.nlp.meta.nnzj)
    mcols = MappedVector(cols, mpcc.meta.ind_j_triplets, mpcc.nlp.meta.nnzj)
    jac_structure!(mpcc.nlp, mrows, mcols) # get including complementarities
    return rows, cols
end

function NLPModels.jac_lin_structure!(
    mpcc::AbstractMPCCModel,
    rows::AbstractVector{Int},
    cols::AbstractVector{Int},
)
    mrows = MappedVector(rows, mpcc.meta.ind_j_lin_triplets, mpcc.nlp.meta.lin_nnzj)
    mcols = MappedVector(cols, mpcc.meta.ind_j_lin_triplets, mpcc.nlp.meta.lin_nnzj)
    jac_lin_structure!(mpcc.nlp, mrows, mcols) # get including complementarities

    # Convert row values adjusting for the number of linear complementarities
    map!((x) -> mpcc.meta.ind_j_lin_row_map[x], rows, rows)

    return rows, cols
end

function NLPModels.jac_nln_structure!(
    mpcc::AbstractMPCCModel,
    rows::AbstractVector{Int},
    cols::AbstractVector{Int},
)
    mrows = MappedVector(rows, mpcc.meta.ind_j_lin_triplets, mpcc.nlp.meta.lin_nnzj)
    mcols = MappedVector(cols, mpcc.meta.ind_j_lin_triplets, mpcc.nlp.meta.lin_nnzj)
    jac_nln_structure!(mpcc.nlp, mrows, mcols) # get including complementarities

    map!((x) -> mpcc.meta.ind_j_nln_row_map[x], rows, rows)
    return rows, cols
end

function NLPModels.jac_coord!(mpcc::AbstractMPCCModel, x::AbstractVector, j::AbstractVector)
    mj = MappedVector(j, mpcc.meta.ind_j_triplets, mpcc.nlp.meta.nnzj)
    jac_coord!(mpcc.nlp, x, mj)

    return j
end

function NLPModels.jac_lin_coord!(
    mpcc::AbstractMPCCModel,
    x::AbstractVector,
    j::AbstractVector,
)
    mj = MappedVector(j, mpcc.meta.ind_j_lin_triplets, mpcc.nlp.meta.lin_nnzj)
    jac_lin_coord!(mpcc.nlp, x, mj)

    return j
end

function NLPModels.jac_nln_coord!(
    mpcc::AbstractMPCCModel,
    x::AbstractVector,
    j::AbstractVector,
)
    mj = MappedVector(j, mpcc.meta.ind_j_nln_triplets, mpcc.nlp.meta.nln_nnzj)
    jac_lin_coord!(mpcc.nlp, x, mj)

    return j
end

function NLPModels.jprod!(
    mpcc::AbstractMPCCModel,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    Jv[1:mpcc.meta.ncon] = jac(mpcc, x) * v
    return Jv
end

function NLPModels.jprod_lin!(
    mpcc::AbstractMPCCModel,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way?
    Jv[1:mpcc.meta.nlin] = jac_lin(mpcc, x) * v
    return Jv
end

function NLPModels.jprod_nln!(
    mpcc::AbstractMPCCModel,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way?
    Jv[1:mpcc.meta.nnln] = jac_nln(mpcc, x) * v
    return Jv
end

function NLPModels.jtprod!(
    mpcc::AbstractMPCCModel,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way?
    Jtv[1:mpcc.meta.nvar] = jac(mpcc, x)' * v
    return Jtv
end

function NLPModels.jtprod_lin!(
    mpcc::AbstractMPCCModel,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way?
    Jtv[1:mpcc.meta.nvar] = jac_lin(mpcc, x)' * v
    return Jtv
end

function NLPModels.jtprod_nln!(
    mpcc::AbstractMPCCModel,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way?
    Jv[1:mpcc.meta.nvar] = jac_nln(mpcc, x)' * v
    return Jtv
end

function NLPModels.hess_structure!(
    mpcc::AbstractMPCCModel,
    rows::AbstractVector{Int},
    cols::AbstractVector{Int},
)
    # TODO(@anton) This currently includes the contribution from the nonlinear complementarity constraint multipliers
    #              which is not correct, but this is hard to mask out so it is fine for now.
    return hess_structure!(mpcc.nlp, rows, cols)
end
function NLPModels.hess_coord!(
    mpcc::AbstractMPCCModel{T, VT},
    x::AbstractVector{T},
    y::AbstractVector{T},
    H::AbstractVector;
    obj_weight::Real=one(T),
) where {T, VT}
    my = MappedVector(y, mpcc.meta.ind_c, mpcc.nlp.meta.ncon)
    return hess_coord!(mpcc.nlp, x, y, H; obj_weight=obj_weight)
end
function NLPModels.hprod!(
    mpcc::AbstractMPCCModel{T, VT},
    x::AbstractVector{T},
    y::AbstractVector{T},
    v::AbstractVector{T},
    Hv::AbstractVector;
    obj_weight::Real=one(T),
) where {T, VT}
    my = MappedVector(y, mpcc.meta.ind_c, mpcc.nlp.meta.ncon)
    return hprod!(mpcc.nlp, x, my, v, Hv; obj_weight=obj_weight)
end

function comp_left(mpcc::AbstractMPCCModel{T, VT}, x::AbstractVector) where {T, VT}
    ccx = VT(undef, mpcc.meta.ncc)
    return comp_left!(mpcc, x, ccx)
end

function comp_left!(mpcc::AbstractMPCCModel, x::AbstractVector, ccx::AbstractVector)
    @lencheck mpcc.meta.ncc ccx
    @lencheck mpcc.meta.nvar x
    # First get variables:
    for i in 1:mpcc.meta.ncc
        if isa(mpcc.meta.cc_types[i], Union{VarVar, VarCon})
            ccx[i] = x[mpcc.meta.ind_cc1[i]]
        end
    end

    mccx = MappedVector(ccx, mpcc.meta.cc_l, mpcc.nlp.meta.ncon)
    NLPModels.cons!(mpcc.nlp, x, mccx)
    return ccx
end

function comp_right(mpcc::AbstractMPCCModel{T, VT}, x::AbstractVector) where {T, VT}
    ccx = VT(undef, mpcc.meta.ncc)
    return comp_right!(mpcc, x, ccx)
end

function comp_right!(mpcc::AbstractMPCCModel, x::AbstractVector, ccx::AbstractVector)
    @lencheck mpcc.meta.ncc ccx
    @lencheck mpcc.meta.nvar x

    # First get variables:
    for i in 1:mpcc.meta.ncc
        if isa(mpcc.meta.cc_types[i], VarVar)
            ccx[i] = x[mpcc.meta.ind_cc2[i]]
        end
    end

    mccx = MappedVector(ccx, mpcc.meta.cc_r, mpcc.nlp.meta.ncon)
    NLPModels.cons!(mpcc.nlp, x, mccx)
    return ccx
end

function lcomp_left(mpcc::AbstractMPCCModel{T, VT}) where {T, VT}
    lccx = VT(undef, mpcc.meta.ncc)
    return lcomp_left!(mpcc, lccx)
end

function lcomp_left!(mpcc::AbstractMPCCModel, lccx::AbstractVector)
    @lencheck mpcc.meta.ncc lccx

    for i in 1:mpcc.meta.ncc
        if isa(mpcc.meta.cc_types[i], Union{VarVar, VarCon})
            lccx[i] = mpcc.nlp.meta.lvar[mpcc.meta.ind_cc1[i]]
        else
            lccx[i] = mpcc.nlp.meta.lcon[mpcc.meta.ind_cc1[i]]
        end
    end
    return lccx
end

function lcomp_right(mpcc::AbstractMPCCModel{T, VT}) where {T, VT}
    lccx = VT(undef, mpcc.meta.ncc)
    return lcomp_right!(mpcc, lccx)
end

function lcomp_right!(mpcc::AbstractMPCCModel, lccx::AbstractVector)
    @lencheck mpcc.meta.ncc lccx

    for i in 1:mpcc.meta.ncc
        if isa(mpcc.meta.cc_types[i], VarVar)
            lccx[i] = mpcc.nlp.meta.lvar[mpcc.meta.ind_cc2[i]]
        else
            lccx[i] = mpcc.nlp.meta.lcon[mpcc.meta.ind_cc2[i]]
        end
    end
    return lccx
end

function comp_res_left!(mpcc::AbstractMPCCModel, x::AbstractVector, rccx::AbstractVector)
    @lencheck mpcc.meta.ncc rccx
    @lencheck mpcc.meta.nvar x

    comp_left!(mpcc, x, rccx)

    for i in 1:mpcc.meta.ncc
        if isa(mpcc.meta.cc_types[i], Union{VarVar, VarCon})
            rccx[i] -= mpcc.nlp.meta.lvar[mpcc.meta.ind_cc1[i]]
        else
            rccx[i] -= mpcc.nlp.meta.lcon[mpcc.meta.ind_cc1[i]]
        end
    end
end

function comp_res_right!(mpcc::AbstractMPCCModel, x::AbstractVector, rccx::AbstractVector)
    @lencheck mpcc.meta.ncc rccx
    @lencheck mpcc.meta.nvar x

    comp_right!(mpcc, x, rccx)

    for i in 1:mpcc.meta.ncc
        if isa(mpcc.meta.cc_types[i], VarVar)
            rccx[i] -= mpcc.nlp.meta.lvar[mpcc.meta.ind_cc2[i]]
        else
            rccx[i] -= mpcc.nlp.meta.lcon[mpcc.meta.ind_cc2[i]]
        end
    end
end

function jac_comp_left_structure(mpcc::AbstractMPCCModel)
    rows = IndexSet(undef, mpcc.meta.comp_left_nnzj)
    cols = IndexSet(undef, mpcc.meta.comp_left_nnzj)

    return jac_comp_left_structure!(mpcc, rows, cols)
end

function jac_comp_left_structure!(
    mpcc::AbstractMPCCModel,
    rows::AbstractVector{Int},
    cols::AbstractVector{Int},
)
    # NOTE: Var type nnz triples come at end ALWAYS
    mrows = MappedVector(rows, mpcc.meta.ind_j_comp_left_triplets, mpcc.nlp.meta.nnzj)
    mcols = MappedVector(cols, mpcc.meta.ind_j_comp_left_triplets, mpcc.nlp.meta.nnzj)

    NLPModels.jac_structure!(mpcc.nlp, mrows, mcols)

    map!(x -> (x ≠ 0 ? mpcc.meta.ind_j_comp_left_row_map[x] : 0), mrows, mrows)

    i_var_comp = length(mpcc.meta.ind_j_comp_left_triplets) + 1
    # TODO(@anton) maybe vectorize
    for i in 1:mpcc.meta.ncc
        if isa(mpcc.meta.cc_types[i], Union{VarVar, VarCon})
            rows[i_var_comp] = i;
            cols[i_var_comp] = mpcc.meta.ind_cc1[i]
            i_var_comp += 1
        end
    end
    return rows, cols
end

function jac_comp_right_structure(mpcc::AbstractMPCCModel)
    rows = IndexSet(undef, mpcc.meta.comp_right_nnzj)
    cols = IndexSet(undef, mpcc.meta.comp_right_nnzj)

    return jac_comp_right_structure!(mpcc, rows, cols)
end

function jac_comp_right_structure!(
    mpcc::AbstractMPCCModel,
    rows::AbstractVector{Int},
    cols::AbstractVector{Int},
)
    # NOTE: Var type nnz triples come at end ALWAYS
    mrows = MappedVector(rows, mpcc.meta.ind_j_comp_right_triplets, mpcc.nlp.meta.nnzj)
    mcols = MappedVector(cols, mpcc.meta.ind_j_comp_right_triplets, mpcc.nlp.meta.nnzj)

    NLPModels.jac_structure!(mpcc.nlp, mrows, mcols)

    map!(x -> (x ≠ 0 ? mpcc.meta.ind_j_comp_right_row_map[x] : 0), mrows, mrows)

    i_var_comp = length(mpcc.meta.ind_j_comp_right_triplets) + 1
    # TODO(@anton) maybe vectorize
    for i in 1:mpcc.meta.ncc
        if isa(mpcc.meta.cc_types[i], VarVar)
            rows[i_var_comp] = i;
            cols[i_var_comp] = mpcc.meta.ind_cc2[i]
            i_var_comp += 1
        end
    end
    return rows, cols
end

function jac_comp_left_coord(
    mpcc::AbstractMPCCModel{T, VT},
    x::AbstractVector,
) where {T, VT}
    vals = VT(undef, mpcc.meta.comp_left_nnzj)

    return jac_comp_left_coord!(mpcc, x, vals)
end

function jac_comp_left_coord!(
    mpcc::AbstractMPCCModel,
    x::AbstractVector,
    vals::AbstractVector,
)
    # NOTE: Var type nnz triples come at end ALWAYS
    mvals = MappedVector(vals, mpcc.meta.ind_j_comp_left_triplets, mpcc.nlp.meta.nnzj)

    NLPModels.jac_coord!(mpcc.nlp, x, mvals)

    i_var_comp = length(mpcc.meta.ind_j_comp_left_triplets) + 1
    # TODO(@anton) maybe vectorize
    for i in 1:mpcc.meta.ncc
        if isa(mpcc.meta.cc_types[i], Union{VarVar, VarCon})
            vals[i_var_comp] = 1.0;
            i_var_comp += 1
        end
    end
    return vals
end

function jac_comp_right_coord(
    mpcc::AbstractMPCCModel{T, VT},
    x::AbstractVector,
) where {T, VT}
    vals = VT(undef, mpcc.meta.comp_left_nnzj)

    return jac_comp_right_coord!(mpcc, x, vals)
end

function jac_comp_right_coord!(
    mpcc::AbstractMPCCModel,
    x::AbstractVector,
    vals::AbstractVector,
)
    # NOTE: Var type nnz triples come at end ALWAYS
    mvals = MappedVector(vals, mpcc.meta.ind_j_comp_right_triplets, mpcc.nlp.meta.nnzj)

    NLPModels.jac_coord!(mpcc.nlp, x, mvals)

    i_var_comp = length(mpcc.meta.ind_j_comp_right_triplets) + 1
    # TODO(@anton) maybe vectorize
    for i in 1:mpcc.meta.ncc
        if isa(mpcc.meta.cc_types[i], VarVar)
            vals[i_var_comp] = 1.0;
            i_var_comp += 1
        end
    end
    return vals
end

function comp_residual(mpcc::AbstractMPCCModel{T, VT}, x::AbstractVector) where {T, VT}
    G = VT(undef, mpcc.meta.ncc)
    H = VT(undef, mpcc.meta.ncc)
    comp_res_left!(mpcc, x, G)
    comp_res_right!(mpcc, x, H)

    map!(min, G, G, H)
    return maximum(G)
end

######################### Vertical Form Conversions #########################
function vertical_form(mpcc::AbstractMPCCModel)
    ind_var1 = [
        mpcc.meta.ind_cc1[i] for i in 1:mpcc.meta.ncc if isa(mpcc.meta.cc_types[i], ConCon)
    ]

    ind_lift1 = [i for i in 1:mpcc.meta.ncc if isa(mpcc.meta.cc_types[i], ConCon)]
    ind_lift2 =
        [i for i in 1:mpcc.meta.ncc if isa(mpcc.meta.cc_types[i], Union{VarCon, ConCon})]
    nlift1 = length(ind_lift1)
    nlift2 = length(ind_lift2)

    ind_lift = vcat(
        map((i) -> mpcc.meta.ind_cc1[i], ind_lift1),
        map((i) -> mpcc.meta.ind_cc2[i], ind_lift2),
    )
    vnlp = LiftedNLPModel(mpcc.nlp, ind_lift)

    lift1 = (mpcc.nlp.meta.nvar+1):(mpcc.nlp.meta.nvar+nlift1)
    lift2 = (mpcc.nlp.meta.nvar+nlift1+1):(mpcc.nlp.meta.nvar+nlift1+nlift2)

    ind_vcc1 = mpcc.meta.ind_cc1
    ind_vcc1[ind_lift1] = lift1
    ind_vcc2 = mpcc.meta.ind_cc2
    ind_vcc2[ind_lift2] = lift2

    return MPCCModelVarVar(vnlp, ind_vcc1, ind_vcc2)
end
# TODO(@anton) Add Core.show overload
