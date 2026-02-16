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

# Taken from NLPModels
show_header(io::IO, mpcc::AbstractMPCCModel) = println(io, typeof(mpcc))

function Base.show(io::IO, mpcc::AbstractMPCCModel)
    show_header(io, mpcc)
    show(io, mpcc.nlp.meta)
    return show(io, mpcc.nlp.counters)
end
# NOTE: This is no longer threadsafe :)
mutable struct MPCCModel{T, VT} <: AbstractMPCCModel{T, VT}
    const nlp::NLPModels.AbstractNLPModel{T, VT}
    const meta::MPCCModelMeta{T, VT}
    const _c1::VT       # [nlp.ncon]
    const _j1::VT       # [nlp.nnzj]
    const _i1::IndexSet # [nlp.nnzj]
    const _i2::IndexSet # [nlp.nnzj]
    const _cc1::VT      # [ncc]
    const _cc2::VT      # [ncc]
end

######################### Helper functions for MPCCModel #########################
function is_vertical(mpcc::MPCCModel)
    return all(map((x)->x==VarVar, mpcc.meta.cc_types))
end

######################### MPCC Types #########################
# Constructor
function MPCCModelVarVar(
    nlp::AbstractNLPModel{T, VT},
    ind_vcc1::IndexSet,
    ind_vcc2::IndexSet,
) where {T, VT}
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
    cc_types = fill!(Vector{CCType}(undef, ncc), VarVar)

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

    # Build work vectors
    _c1 = VT(undef, nlp.meta.ncon)
    _j1 = VT(undef, nlp.meta.nnzj)
    _i1 = IndexSet(undef, nlp.meta.nnzj)
    _i2 = IndexSet(undef, nlp.meta.nnzj)
    _cc1 = VT(undef, ncc)
    _cc2 = VT(undef, ncc)

    return MPCCModel(nlp, meta, _c1, _j1, _i1, _i2, _cc1, _cc2)
end

# Constructor
function MPCCModelConCon(
    nlp::AbstractNLPModel{T, VT},
    ind_ccc1::IndexSet,
    ind_ccc2::IndexSet,
) where {T, VT}
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
            Dict((i, i-count([x < i for x in cc_lin])) for i in 1:nlp.meta.nlin)
        ind_j_nln_row_map =
            Dict((i, i-count([x < i for x in cc_nln])) for i in 1:nlp.meta.nnln)
    else
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
            Dict((i, i-count([x < i for x in cc_nln])) for i in 1:nlp.meta.nnln)
    end
    ind_j_comp_left_triplets = findall(x->x∈ind_ccc1, rows);
    ind_j_comp_right_triplets = findall(x->x∈ind_ccc2, rows);
    ind_j_comp_left_row_map = Dict{Int, Int}(zip(ind_ccc1, 1:ncc))
    ind_j_comp_right_row_map = Dict{Int, Int}(zip(ind_ccc2, 1:ncc))

    # Complementarity Constraints
    ind_cc1 = ind_ccc1;
    ind_cc2 = ind_ccc2;
    cc_types = fill!(Vector{CCType}(undef, ncc), ConCon)
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

    # Build work vectors
    _c1 = VT(undef, nlp.meta.ncon)
    _j1 = VT(undef, nlp.meta.nnzj)
    _i1 = IndexSet(undef, nlp.meta.nnzj)
    _i2 = IndexSet(undef, nlp.meta.nnzj)
    _cc1 = VT(undef, ncc)
    _cc2 = VT(undef, ncc)

    return MPCCModel(nlp, meta, _c1, _j1, _i1, _i2, _cc1, _cc2)
end

# Constructor
function MPCCModelVarCon(
    nlp::AbstractNLPModel{T, VT},
    ind_vcc1::IndexSet,
    ind_ccc2::IndexSet,
) where {T, VT}
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
            Dict((i, i-count([x < i for x in cc_lin])) for i in 1:nlp.meta.nlin)
        ind_j_nln_row_map =
            Dict((i, i-count([x < i for x in cc_nln])) for i in 1:nlp.meta.nnln)
    else
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
            Dict((i, i-count([x < i for x in cc_nln])) for i in 1:nlp.meta.nnln)
    end
    ind_j_comp_left_triplets::IndexSet = [];
    ind_j_comp_right_triplets = findall(x->x∈ind_ccc2, rows);
    ind_j_comp_left_row_map = Dict{Int, Int}()
    ind_j_comp_right_row_map = Dict{Int, Int}(zip(ind_ccc2, 1:ncc))

    # UNUSED
    ind_cc1 = ind_vcc1;
    ind_cc2 = ind_ccc2;
    cc_types = fill!(Vector{CCType}(undef, ncc), VarCon)
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

    # Build work vectors
    _c1 = VT(undef, nlp.meta.ncon)
    _j1 = VT(undef, nlp.meta.nnzj)
    _i1 = IndexSet(undef, nlp.meta.nnzj)
    _i2 = IndexSet(undef, nlp.meta.nnzj)
    _cc1 = VT(undef, ncc)
    _cc2 = VT(undef, ncc)

    return MPCCModel(nlp, meta, _c1, _j1, _i1, _i2, _cc1, _cc2)
end

######################### Implementing NLPModels API #########################
NLPModels.obj(mpcc::AbstractMPCCModel, x::AbstractVector) = NLPModels.obj(mpcc.nlp, x)
function NLPModels.grad!(mpcc::AbstractMPCCModel, x::AbstractVector, gx::AbstractVector)
    NLPModels.grad!(mpcc.nlp, x, gx)
    return gx
end
function NLPModels.grad(mpcc::AbstractMPCCModel{T}, x::AbstractVector{T}) where {T}
    g = Vector{T}(undef, mpcc.meta.nvar)
    return grad!(mpcc, x, g)
end

function NLPModels.objgrad!(mpcc::AbstractMPCCModel, x::AbstractVector, g::AbstractVector)
    return NLPModels.objgrad!(mpcc.nlp, x, g)
end

function NLPModels.cons!(mpcc::AbstractMPCCModel, x::AbstractVector, cx::AbstractVector)
    # TODO(@anton) do we want to use a SFINAE style definition only for nlpmodel types which do not support
    #              the linear interface
    cons!(mpcc.nlp, x, mpcc._c1)
    @views cx .= mpcc._c1[mpcc.meta.ind_c]
    return cx
end

function NLPModels.cons_lin!(mpcc::AbstractMPCCModel, x::AbstractVector, cx::AbstractVector)
    _c_lin = view(mpcc._c1, 1:mpcc.nlp.meta.nlin)
    cons_lin!(mpcc.nlp, x, _c_lin)
    @views cx .= _c_lin[mpcc.meta.c_lin]
    return cx
end

function NLPModels.cons_nln!(mpcc::AbstractMPCCModel, x::AbstractVector, cx::AbstractVector)
    _c_nln = view(mpcc._c1, 1:mpcc.nlp.meta.nnln)
    cons_nln!(mpcc.nlp, x, _c_nln)
    @views cx .= _c_lin[mpcc.meta.c_nln]
    return cx
end

function NLPModels.jac_structure!(
    mpcc::AbstractMPCCModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    jac_structure!(mpcc.nlp, mpcc._i1, mpcc._i2) # get including complementarities
    @views begin
        rows .= mpcc._i1[mpcc.meta.ind_j_triplets]
        cols .= mpcc._i2[mpcc.meta.ind_j_triplets]
    end
    return rows, cols
end

function NLPModels.jac_structure(mpcc::AbstractMPCCModel)
    rows = Vector{Int}(undef, mpcc.meta.nnzj)
    cols = Vector{Int}(undef, mpcc.meta.nnzj)
    return jac_structure!(mpcc, rows, cols)
end

function NLPModels.jac_lin_structure!(
    mpcc::AbstractMPCCModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    _rows = view(mpcc._i1, 1:mpcc.nlp.meta.lin_nnzj)
    _cols = view(mpcc._i2, 1:mpcc.nlp.meta.lin_nnzj)
    jac_lin_structure!(mpcc.nlp, _rows, _cols) # get including complementarities
    @views begin
        rows .= _lin_rows[mpcc.meta.ind_j_lin_triplets]
        cols .= _lin_cols[mpcc.meta.ind_j_lin_triplets]
    end
    # Convert row values adjusting for the number of linear complementarities
    map!((x) -> mpcc.meta.ind_j_lin_row_map[x], rows, rows)

    return rows, cols
end

function NLPModels.jac_nln_structure!(
    mpcc::AbstractMPCCModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    _rows = view(mpcc._i1, 1:mpcc.nlp.meta.nln_nnzj)
    _cols = view(mpcc._i2, 1:mpcc.nlp.meta.nln_nnzj)
    jac_lin_structure!(mpcc.nlp, _rows, _cols) # get including complementarities
    @views begin
        rows .= _rows[mpcc.meta.ind_j_nln_triplets]
        cols .= _cols[mpcc.meta.ind_j_nln_triplets]
    end
    # Convert row values adjusting for the number of nonlinear complementarities
    map!((x) -> mpcc.meta.ind_j_nln_row_map[x], rows, rows)

    return rows, cols
end

function NLPModels.jac_coord!(mpcc::AbstractMPCCModel, x::AbstractVector, j::AbstractVector)
    jac_coord!(mpcc.nlp, x, mpcc._j1)
    @views j .= mpcc._j1[mpcc.meta.ind_j_triplets]
    return j
end

function NLPModels.jac_coord(mpcc::AbstractMPCCModel{T}, x::AbstractVector) where {T}
    vals = Vector{T}(undef, mpcc.meta.nnzj)
    return jac_coord!(mpcc, x, vals)
end

function NLPModels.jac_lin_coord!(
    mpcc::AbstractMPCCModel,
    x::AbstractVector,
    j::AbstractVector,
)
    _j = view(mpcc._j1, 1:mpcc.nlp.meta.lin_nnzj)
    jac_lin_coord!(mpcc.nlp, x, _j)
    @views j .= _j[mpcc.meta.ind_j_lin_triplets]
    return j
end

function NLPModels.jac_nln_coord!(
    mpcc::AbstractMPCCModel,
    x::AbstractVector,
    j::AbstractVector,
)
    _j = view(mpcc._j1, 1:mpcc.nlp.meta.nln_nnzj)
    jac_lin_coord!(mpcc.nlp, x, _j)
    @views j .= _j[mpcc.meta.ind_j_nln_triplets]
    return j
end

function NLPModels.jprod!(
    mpcc::AbstractMPCCModel,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    Jv[1:mpcc.meta.ncon] .= jac(mpcc, x) * v
    return Jv
end

function NLPModels.jprod_lin!(
    mpcc::AbstractMPCCModel,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way?
    Jv[1:mpcc.meta.nlin] .= jac_lin(mpcc, x) * v
    return Jv
end

function NLPModels.jprod_nln!(
    mpcc::AbstractMPCCModel,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way?
    Jv[1:mpcc.meta.nnln] .= jac_nln(mpcc, x) * v
    return Jv
end

function NLPModels.jtprod!(
    mpcc::AbstractMPCCModel,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way?
    Jtv[1:mpcc.meta.nvar] .= jac(mpcc, x)' * v
    return Jtv
end

function NLPModels.jtprod_lin!(
    mpcc::AbstractMPCCModel,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way?
    Jtv[1:mpcc.meta.nvar] .= jac_lin(mpcc, x)' * v
    return Jtv
end

function NLPModels.jtprod_nln!(
    mpcc::AbstractMPCCModel,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way?
    Jv[1:mpcc.meta.nvar] .= jac_nln(mpcc, x)' * v
    return Jtv
end

function NLPModels.hess_structure!(
    mpcc::AbstractMPCCModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
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
    mpcc._c1 .= T(0.0)
    mpcc._c1[mpcc.meta.ind_c] .= y
    return hess_coord!(mpcc.nlp, x, mpcc._c1, H; obj_weight=obj_weight)
end
function NLPModels.hprod!(
    mpcc::AbstractMPCCModel{T, VT},
    x::AbstractVector{T},
    y::AbstractVector{T},
    v::AbstractVector{T},
    Hv::AbstractVector;
    obj_weight::Real=one(T),
) where {T, VT}
    mpcc._c1 .= T(0.0)
    mpcc._c1[mpcc.meta.ind_c] .= y
    return hprod!(mpcc.nlp, x, mpcc._c1, v, Hv; obj_weight=obj_weight)
end

function comp_left(mpcc::AbstractMPCCModel{T, VT}, x::AbstractVector) where {T, VT}
    ccx = VT(undef, mpcc.meta.ncc)
    return comp_left!(mpcc, x, ccx)
end

function comp_left!(mpcc::AbstractMPCCModel, x::AbstractVector, ccx::AbstractVector)
    @lencheck mpcc.meta.ncc ccx
    @lencheck mpcc.meta.nvar x
    cvar = 0
    # First get variables:
    for i in 1:mpcc.meta.ncc
        if mpcc.meta.cc_types[i] ∈ [VarVar, VarCon]
            ccx[i] = x[mpcc.meta.ind_cc1[i]]
            cvar += 1
        end
    end

    # TODO(@anton) I am not sure anymore if this is correct for non-vertical form
    cons!(mpcc.nlp, x, mpcc._c1)
    @views ccx[(cvar+1):end] .= mpcc._c1[mpcc.meta.cc_l]
    return ccx
end

function comp_right(mpcc::AbstractMPCCModel{T, VT}, x::AbstractVector) where {T, VT}
    ccx = VT(undef, mpcc.meta.ncc)
    return comp_right!(mpcc, x, ccx)
end

function comp_right!(mpcc::AbstractMPCCModel, x::AbstractVector, ccx::AbstractVector)
    @lencheck mpcc.meta.ncc ccx
    @lencheck mpcc.meta.nvar x
    cvar = 0
    # First get variables:
    for i in 1:mpcc.meta.ncc
        if mpcc.meta.cc_types[i] ∈ [VarVar, ConVar]
            ccx[i] = x[mpcc.meta.ind_cc2[i]]
            cvar += 1
        end
    end

    # TODO(@anton) I am not sure anymore if this is correct for non-vertical form
    cons!(mpcc.nlp, x, mpcc._c1)
    @views ccx[(cvar+1):end] .= mpcc._c1[mpcc.meta.cc_r]
    return ccx
end

function lcomp_left(mpcc::AbstractMPCCModel{T, VT}) where {T, VT}
    lccx = VT(undef, mpcc.meta.ncc)
    return lcomp_left!(mpcc, lccx)
end

function lcomp_left!(mpcc::AbstractMPCCModel{T, VT}, lccx::AbstractVector) where {T, VT}
    @lencheck mpcc.meta.ncc lccx

    for i in 1:mpcc.meta.ncc
        if mpcc.meta.cc_types[i] ∈ [VarVar, VarCon]
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

function lcomp_right!(mpcc::AbstractMPCCModel{T, VT}, lccx::AbstractVector) where {T, VT}
    @lencheck mpcc.meta.ncc lccx

    for i in 1:mpcc.meta.ncc
        if mpcc.meta.cc_types[i] ∈ [VarVar, ConVar]
            lccx[i] = mpcc.nlp.meta.lvar[mpcc.meta.ind_cc2[i]]
        else
            lccx[i] = mpcc.nlp.meta.lcon[mpcc.meta.ind_cc2[i]]
        end
    end
    return lccx
end

function comp_res_left(mpcc::AbstractMPCCModel{T, VT}, x::AbstractVector) where {T, VT}
    lccx = VT(undef, mpcc.meta.ncc)
    return comp_res_left!(mpcc, x, lccx)
end

function comp_res_left!(mpcc::AbstractMPCCModel, x::AbstractVector, lccx::AbstractVector)
    @lencheck mpcc.meta.ncc lccx
    @lencheck mpcc.meta.nvar x

    comp_left!(mpcc, x, lccx)

    for i in 1:mpcc.meta.ncc
        if mpcc.meta.cc_types[i] ∈ [VarVar, VarCon]
            lccx[i] -= mpcc.nlp.meta.lvar[mpcc.meta.ind_cc1[i]]
        else
            lccx[i] -= mpcc.nlp.meta.lcon[mpcc.meta.ind_cc1[i]]
        end
    end
    return lccx
end

function comp_res_right(mpcc::AbstractMPCCModel{T, VT}, x::AbstractVector) where {T, VT}
    rccx = VT(undef, mpcc.meta.ncc)
    return comp_res_right!(mpcc, x, rccx)
end

function comp_res_right!(mpcc::AbstractMPCCModel, x::AbstractVector, rccx::AbstractVector)
    @lencheck mpcc.meta.ncc rccx
    @lencheck mpcc.meta.nvar x

    comp_right!(mpcc, x, rccx)

    for i in 1:mpcc.meta.ncc
        if mpcc.meta.cc_types[i] ∈ [VarVar, ConVar]
            rccx[i] -= mpcc.nlp.meta.lvar[mpcc.meta.ind_cc2[i]]
        else
            rccx[i] -= mpcc.nlp.meta.lcon[mpcc.meta.ind_cc2[i]]
        end
    end
    return rccx
end

function jac_comp_left_structure(mpcc::AbstractMPCCModel)
    rows = IndexSet(undef, mpcc.meta.comp_left_nnzj)
    cols = IndexSet(undef, mpcc.meta.comp_left_nnzj)

    return jac_comp_left_structure!(mpcc, rows, cols)
end

function jac_comp_left_structure!(
    mpcc::AbstractMPCCModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    _rows = mpcc._i1
    _cols = mpcc._i2
    jac_lin_structure!(mpcc.nlp, _rows, _cols) # get including complementarities

    @views begin
        rows[1:length(mpcc.meta.ind_j_comp_left_triplets)] .=
            _rows[mpcc.meta.ind_j_comp_left_triplets]
        cols[1:length(mpcc.meta.ind_j_comp_left_triplets)] .=
            _cols[mpcc.meta.ind_j_comp_left_triplets]
        map!(
            x -> mpcc.meta.ind_j_comp_left_row_map[x],
            rows[1:length(mpcc.meta.ind_j_comp_left_triplets)],
            rows[1:length(mpcc.meta.ind_j_comp_left_triplets)],
        )
    end

    i_var_comp = length(mpcc.meta.ind_j_comp_left_triplets) + 1
    # TODO(@anton) maybe vectorize
    for i in 1:mpcc.meta.ncc
        if mpcc.meta.cc_types[i] ∈ [VarVar, VarCon]
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
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    _rows = mpcc._i1
    _cols = mpcc._i2
    jac_lin_structure!(mpcc.nlp, _rows, _cols) # get including complementarities

    @views begin
        rows[1:length(mpcc.meta.ind_j_comp_right_triplets)] .=
            _rows[mpcc.meta.ind_j_comp_right_triplets]
        cols[1:length(mpcc.meta.ind_j_comp_right_triplets)] .=
            _cols[mpcc.meta.ind_j_comp_right_triplets]

        map!(
            x -> mpcc.meta.ind_j_comp_right_row_map[x],
            rows[1:length(mpcc.meta.ind_j_comp_right_triplets)],
            rows[1:length(mpcc.meta.ind_j_comp_right_triplets)],
        )
    end

    i_var_comp = length(mpcc.meta.ind_j_comp_right_triplets) + 1
    # TODO(@anton) maybe vectorize
    for i in 1:mpcc.meta.ncc
        if mpcc.meta.cc_types[i] ∈ [VarVar, ConVar]
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
    _vals = mpcc._j1
    NLPModels.jac_coord!(mpcc.nlp, x, _vals)
    @views vals[1:length(mpcc.meta.ind_j_comp_left_triplets)] .=
        _vals[mpcc.meta.ind_j_comp_left_triplets]

    i_var_comp = length(mpcc.meta.ind_j_comp_left_triplets) + 1
    # TODO(@anton) maybe vectorize
    for i in 1:mpcc.meta.ncc
        if mpcc.meta.cc_types[i] ∈ [VarVar, VarCon]
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
    _vals = mpcc._j1
    NLPModels.jac_coord!(mpcc.nlp, x, _vals)
    @views vals[1:length(mpcc.meta.ind_j_comp_right_triplets)] .=
        _vals[mpcc.meta.ind_j_comp_right_triplets]

    i_var_comp = length(mpcc.meta.ind_j_comp_right_triplets) + 1
    # TODO(@anton) maybe vectorize
    for i in 1:mpcc.meta.ncc
        if mpcc.meta.cc_types[i] ∈ [VarVar, ConVar]
            vals[i_var_comp] = 1.0;
            i_var_comp += 1
        end
    end
    return vals
end

function comp_residual(mpcc::AbstractMPCCModel{T, VT}, x::AbstractVector) where {T, VT}
    # TODO(@anton): This can be done more efficiently in vertical form
    G = mpcc._cc1
    H = mpcc._cc2
    comp_res_left!(mpcc, x, G)
    comp_res_right!(mpcc, x, H)

    map!(min, G, G, H)
    return maximum(G)
end

function comp_residual_product(
    mpcc::AbstractMPCCModel{T, VT},
    x::AbstractVector,
) where {T, VT}
    # TODO(@anton): This can be done more efficiently in vertical form
    G = mpcc._cc1
    H = mpcc._cc2
    comp_res_left!(mpcc, x, G)
    comp_res_right!(mpcc, x, H)

    G .*= H
    return maximum(G)
end

function comp_residual_sum(mpcc::AbstractMPCCModel{T, VT}, x::AbstractVector) where {T, VT}
    # TODO(@anton): This can be done more efficiently in vertical form
    G = mpcc._cc1
    H = mpcc._cc2
    comp_res_left!(mpcc, x, G)
    comp_res_right!(mpcc, x, H)
    return dot(G, H)
end

######################### Vertical Form Conversions #########################
function vertical_form(mpcc::AbstractMPCCModel)
    ind_var1 = [
        mpcc.meta.ind_cc1[i] for
        i in 1:mpcc.meta.ncc if mpcc.meta.cc_types[i]∈[ConVar, ConCon]
    ]

    ind_lift1::IndexSet =
        [i for i in 1:mpcc.meta.ncc if mpcc.meta.cc_types[i]∈[ConVar, ConCon]]
    ind_lift2::IndexSet =
        [i for i in 1:mpcc.meta.ncc if mpcc.meta.cc_types[i]∈[VarCon, ConCon]]
    nlift1 = length(ind_lift1)
    nlift2 = length(ind_lift2)

    ind_lift::IndexSet = vcat(
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
