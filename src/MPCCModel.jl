"""
  Abstract type for problems in the form:
  min f(w)
  s.t. lbc ≤ c(w) ≤ ubc
       0 ≤ G(w) ⟂ H(w) ≥ 0
"""
abstract type AbstractMPCCModel{T, VT} end

# TODO(@anton) For a prototype this is a bit of a hack to account for not having meta and counters
#              In principle we want to have a MPCCModelMeta and MPCCModelCounters?
# function Base.getproperty(mpcc::AbstractMPCCModel, sym::Symbol)
#     if sym ∈ [:counters]
#         getfield(mpcc.nlp, sym)
#     else
#         getfield(mpcc, sym)
#     end
# end

# Taken from NLPModels
show_header(io::IO, mpcc::AbstractMPCCModel) = println(io, typeof(mpcc))

function Base.show(io::IO, mpcc::AbstractMPCCModel)
    show_header(io, mpcc)
    show(io, mpcc.nlp.meta)
    return show(io, mpcc.nlp.counters)
end
# NOTE: This is no longer threadsafe :)
struct MPCCModel{T, VT, NLP <: NLPModels.AbstractNLPModel{T, VT}, NMT} <:
       AbstractMPCCModel{T, VT}
    nlp::NLP
    meta::MPCCModelMeta{T, VT, NMT}
    _c1::VT       # [nlp.ncon]
    _j1::VT       # [nlp.nnzj]
    _i1::IndexSet # [nlp.nnzj]
    _i2::IndexSet # [nlp.nnzj]
    _cc1::VT      # [ncc]
    _cc2::VT      # [ncc]
    counters::NLPModels.Counters
end

######################### Helper functions for MPCCModel #########################
function is_vertical(mpcc::MPCCModel)
    return all(map((x)->x==VarVar, get_cc_types(mpcc)))
end

######################### MPCC Types #########################
# Constructor
function MPCCModel(
    nlp::AbstractNLPModel{T, VT},
    ind_vcc1::IndexSet,
    ind_vcc2::IndexSet,
) where {T, VT}
    # compute sizes
    ncc = length(ind_vcc1)
    ncon = get_ncon(nlp)
    nlin = get_nlin(nlp)
    nnln = get_nnln(nlp)

    # compute non-complementarity variables/constraints
    ind_x = setdiff(1:get_nvar(nlp), union(ind_vcc1, ind_vcc2))
    ind_c = collect(1:get_ncon(nlp))

    # compute jacobian structure indexset
    ind_j_triplets = collect(1:get_nnzj(nlp))
    ind_j_lin_triplets = collect(1:get_lin_nnzj(nlp))
    ind_j_nln_triplets = collect(1:get_nln_nnzj(nlp))
    ind_j_lin_row_map = Dict(zip(1:nlin, 1:nlin))
    ind_j_nln_row_map = Dict(zip(1:nnln, 1:nnln))

    ind_j_comp_left_triplets::IndexSet = []
    ind_j_comp_right_triplets::IndexSet = []
    ind_j_comp_left_row_map = Dict{Int, Int}()
    ind_j_comp_right_row_map = Dict{Int, Int}()
    # compute nln and lin index sets
    lin = get_lin(nlp)
    nln = get_nln(nlp)
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
    nnzj = get_nnzj(nlp)
    lin_nnzj = get_lin_nnzj(nlp)
    nln_nnzj = get_nln_nnzj(nlp)
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
    _c1 = VT(undef, get_ncon(nlp))
    _j1 = VT(undef, get_nnzj(nlp))
    _i1 = IndexSet(undef, get_nnzj(nlp))
    _i2 = IndexSet(undef, get_nnzj(nlp))
    _cc1 = VT(undef, ncc)
    _cc2 = VT(undef, ncc)

    return MPCCModel(nlp, meta, _c1, _j1, _i1, _i2, _cc1, _cc2, nlp.counters)
end

# Constructor
function MPCCModelConCon(
    nlp::AbstractNLPModel{T, VT},
    ind_ccc1::IndexSet,
    ind_ccc2::IndexSet,
) where {T, VT}
    # compute sizes
    ncc = length(ind_ccc1)
    ncon = get_ncon(nlp) - 2*ncc

    # compute non-complementarity variables/constraints
    ind_x = collect(1:get_nvar(nlp))
    ind_c = setdiff(1:get_ncon(nlp), union(ind_ccc1, ind_ccc2))

    # compute jacobian structure indexset
    rows, cols = NLPModels.jac_structure(nlp)
    ind_j_triplets = findall(x->x ∈ ind_c, rows)
    if hasmethod(jac_lin_structure!, (typeof(nlp), IndexSet, IndexSet))
        lin_rows, lin_cols = NLPModels.jac_lin_structure(nlp)
        nln_rows, nln_cols = NLPModels.jac_nln_structure(nlp)
        for i in 1:get_nlin(nlp)
            lin_rows[i] += count(x < get_lin(nlp)[lin_rows[i]] for x in get_nln(nlp))
        end
        for i in 1:get_nnln(nlp)
            nln_rows[i] += count(x < get_nln(nlp)[nln_rows[i]] for x in get_lin(nlp))
        end
        ind_j_lin_triplets = findall(x->!((x∈ind_ccc1) || (x∈ind_ccc2)), lin_rows)
        ind_j_nln_triplets = findall(x->!((x∈ind_ccc1) || (x∈ind_ccc2)), nln_rows)

        # compute nln and lin index sets
        lin = intersect(get_lin(nlp), ind_c)
        nln = intersect(get_nln(nlp), ind_c)
        nlin = length(lin)
        nnln = length(nln)
        c_lin = [i for i in 1:nlin if get_lin(nlp)[i] ∈ ind_c]
        cc_lin = [i for i in 1:nlin if get_lin(nlp)[i] ∉ ind_c]
        c_nln = [i for i in 1:nnln if get_nln(nlp)[i] ∈ ind_c]
        cc_nln = [i for i in 1:nnln if get_nln(nlp)[i] ∉ ind_c]

        ind_j_lin_row_map =
            Dict((i, i-count([x < i for x in cc_lin])) for i in 1:get_nlin(nlp))
        ind_j_nln_row_map =
            Dict((i, i-count([x < i for x in cc_nln])) for i in 1:get_nnln(nlp))
    else
        ind_j_lin_triplets::IndexSet = []
        ind_j_nln_triplets = ind_j_triplets
        lin::IndexSet = []
        nln = get_nln(nlp)
        nlin = get_nlin(nlp)
        nnln = get_nnln(nlp)
        c_lin::IndexSet = []
        cc_lin::IndexSet = []
        c_nln = [i for i in 1:nnln if get_nln(nlp)[i] ∈ ind_c]
        cc_nln = [i for i in 1:nnln if get_nln(nlp)[i] ∉ ind_c]

        ind_j_lin_row_map = Dict{Int, Int}()
        ind_j_nln_row_map::Dict{Int, Int} =
            Dict((i, i-count([x < i for x in cc_nln])) for i in 1:get_nnln(nlp))
    end
    ind_j_comp_left_triplets = findall(x->x∈ind_ccc1, rows);
    ind_j_comp_right_triplets = findall(x->x∈ind_ccc2, rows);
    ind_j_comp_left_row_map = Dict{Int, Int}(zip(ind_ccc1, 1:ncc))
    ind_j_comp_right_row_map = Dict{Int, Int}(zip(ind_ccc2, 1:ncc))

    # Complementarity Constraints
    ind_cc1 = ind_ccc1;
    ind_cc2 = ind_ccc2;
    cc_types = fill!(Vector{CCType}(undef, ncc), ConCon)
    cc_l = [i for i in 1:get_ncon(nlp) if i ∈ ind_cc1]
    cc_r = [i for i in 1:get_ncon(nlp) if i ∈ ind_cc2]

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
    _c1 = VT(undef, get_ncon(nlp))
    _j1 = VT(undef, get_nnzj(nlp))
    _i1 = IndexSet(undef, get_nnzj(nlp))
    _i2 = IndexSet(undef, get_nnzj(nlp))
    _cc1 = VT(undef, ncc)
    _cc2 = VT(undef, ncc)

    return MPCCModel(nlp, meta, _c1, _j1, _i1, _i2, _cc1, _cc2, nlp.counters)
end

# Constructor
function MPCCModelVarCon(
    nlp::AbstractNLPModel{T, VT},
    ind_vcc1::IndexSet,
    ind_ccc2::IndexSet,
) where {T, VT}
    # compute sizes
    ncc = length(ind_vcc1)
    ncon = get_ncon(nlp) - ncc
    # compute non-complementarity variables/constraints
    ind_x = setdiff(1:get_nvar(nlp), ind_vcc1)
    ind_c = setdiff(1:get_ncon(nlp), ind_ccc2)

    # compute jacobian structure indexset
    rows, cols = NLPModels.jac_structure(nlp)
    ind_j_triplets = findall(x->x ∈ ind_c, rows)
    if hasmethod(jac_lin_structure!, (typeof(nlp), IndexSet, IndexSet))
        lin_rows, lin_cols = NLPModels.jac_lin_structure(nlp)
        nln_rows, nln_cols = NLPModels.jac_nln_structure(nlp)
        # Convert to true row numbers
        for i in 1:get_nlin(nlp)
            lin_rows[i] += count(x < get_lin(nlp)[lin_rows[i]] for x in get_nln(nlp))
        end
        for i in 1:get_nnln(nlp)
            nln_rows[i] += count(x < get_nln(nlp)[nln_rows[i]] for x in get_lin(nlp))
        end
        # Keep only the "correct" indices
        ind_j_lin_triplets = findall(x->!(x∈ind_ccc2), lin_rows)
        ind_j_nln_triplets = findall(x->!(x∈ind_ccc2), nln_rows)

        # compute nln and lin index sets
        lin = intersect(get_lin(nlp), ind_c)
        nln = intersect(get_nln(nlp), ind_c)
        nlin = length(lin)
        nnln = length(nln)
        c_lin = [i for i in 1:nlin if get_lin(nlp)[i] ∈ ind_c]
        cc_lin = [i for i in 1:nlin if get_lin(nlp)[i] ∉ ind_c]
        c_nln = [i for i in 1:nnln if get_nln(nlp)[i] ∈ ind_c]
        cc_nln = [i for i in 1:nnln if get_nln(nlp)[i] ∉ ind_c]

        ind_j_lin_row_map =
            Dict((i, i-count([x < i for x in cc_lin])) for i in 1:get_nlin(nlp))
        ind_j_nln_row_map =
            Dict((i, i-count([x < i for x in cc_nln])) for i in 1:get_nnln(nlp))
    else
        ind_j_lin_triplets::IndexSet = []
        ind_j_nln_triplets = ind_j_triplets
        lin::IndexSet = []
        nln = get_nln(nlp)
        nlin = get_nlin(nlp)
        nnln = get_nnln(nlp)
        c_lin::IndexSet = []
        cc_lin::IndexSet = []
        c_nln = [i for i in 1:nnln if get_nln(nlp)[i] ∈ ind_c]
        cc_nln = [i for i in 1:nnln if get_nln(nlp)[i] ∉ ind_c]

        ind_j_lin_row_map = Dict{Int, Int}()
        ind_j_nln_row_map::Dict{Int, Int} =
            Dict((i, i-count([x < i for x in cc_nln])) for i in 1:get_nnln(nlp))
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
    cc_r = [i for i in 1:get_ncon(nlp) if i ∈ ind_cc2]

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
    _c1 = VT(undef, get_ncon(nlp))
    _j1 = VT(undef, get_nnzj(nlp))
    _i1 = IndexSet(undef, get_nnzj(nlp))
    _i2 = IndexSet(undef, get_nnzj(nlp))
    _cc1 = VT(undef, ncc)
    _cc2 = VT(undef, ncc)

    return MPCCModel(nlp, meta, _c1, _j1, _i1, _i2, _cc1, _cc2, nlp.counters)
end

# Verticalize generic CC types. Returns a vertical form MPCC
# TODO(@anton) we do no checks here :)
function MPCCModel(
    nlp::AbstractNLPModel,
    ind_cc1::IndexSet,
    ind_cc2::IndexSet,
    cc_types::AbstractVector{CCType},
)
    ncc = length(ind_cc1)
    nvar = get_nvar(nlp)

    ind_lift1::IndexSet = [i for i in 1:ncc if cc_types[i]∈[ConVar, ConCon]]
    ind_lift2::IndexSet = [i for i in 1:ncc if cc_types[i]∈[VarCon, ConCon]]
    nlift1 = length(ind_lift1)
    nlift2 = length(ind_lift2)

    ind_lift::IndexSet =
        vcat(map((i) -> ind_cc1[i], ind_lift1), map((i) -> ind_cc2[i], ind_lift2))
    vnlp = LiftedNLPModel(nlp, ind_lift)

    lift1 = (nvar+1):(nvar+nlift1)
    lift2 = (nvar+nlift1+1):(nvar+nlift1+nlift2)

    ind_vcc1 = ind_cc1
    ind_vcc1[ind_lift1] = lift1
    ind_vcc2 = ind_cc2
    ind_vcc2[ind_lift2] = lift2

    return MPCCModel(vnlp, ind_vcc1, ind_vcc2)
end

######################### Implementing NLPModels API #########################
NLPModels.obj(mpcc::AbstractMPCCModel, x::AbstractVector) = NLPModels.obj(mpcc.nlp, x)
function NLPModels.grad!(mpcc::AbstractMPCCModel, x::AbstractVector, gx::AbstractVector)
    NLPModels.grad!(mpcc.nlp, x, gx)
    return gx
end
function NLPModels.grad(mpcc::AbstractMPCCModel{T}, x::AbstractVector{T}) where {T}
    g = Vector{T}(undef, get_nvar(mpcc))
    return grad!(mpcc, x, g)
end

function NLPModels.objgrad!(mpcc::AbstractMPCCModel, x::AbstractVector, g::AbstractVector)
    return NLPModels.objgrad!(mpcc.nlp, x, g)
end

function NLPModels.cons!(mpcc::AbstractMPCCModel, x::AbstractVector, cx::AbstractVector)
    # TODO(@anton) do we want to use a SFINAE style definition only for nlpmodel types which do not support
    #              the linear interface
    cons!(mpcc.nlp, x, mpcc._c1)
    @views cx .= mpcc._c1[get_ind_c(mpcc)]
    return cx
end

function NLPModels.cons_lin!(mpcc::AbstractMPCCModel, x::AbstractVector, cx::AbstractVector)
    _c_lin = view(mpcc._c1, 1:get_nlin(mpcc.nlp))
    cons_lin!(mpcc.nlp, x, _c_lin)
    @views cx .= _c_lin[get_c_lin(mpcc)]
    return cx
end

function NLPModels.cons_nln!(mpcc::AbstractMPCCModel, x::AbstractVector, cx::AbstractVector)
    _c_nln = view(mpcc._c1, 1:get_nnln(mpcc.nlp))
    cons_nln!(mpcc.nlp, x, _c_nln)
    @views cx .= _c_lin[get_c_nln(mpcc)]
    return cx
end

function NLPModels.jac_structure!(
    mpcc::AbstractMPCCModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    jac_structure!(mpcc.nlp, mpcc._i1, mpcc._i2) # get including complementarities
    @views begin
        rows .= mpcc._i1[get_ind_j_triplets(mpcc)]
        cols .= mpcc._i2[get_ind_j_triplets(mpcc)]
    end
    return rows, cols
end

function NLPModels.jac_structure(mpcc::AbstractMPCCModel)
    rows = Vector{Int}(undef, get_nnzj(mpcc))
    cols = Vector{Int}(undef, get_nnzj(mpcc))
    return jac_structure!(mpcc, rows, cols)
end

function NLPModels.jac_lin_structure!(
    mpcc::AbstractMPCCModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    _rows = view(mpcc._i1, 1:get_lin_nnzj(mpcc.nlp))
    _cols = view(mpcc._i2, 1:get_lin_nnzj(mpcc.nlp))
    jac_lin_structure!(mpcc.nlp, _rows, _cols) # get including complementarities
    @views begin
        rows .= _rows[get_ind_j_lin_triplets(mpcc)]
        cols .= _cols[get_ind_j_lin_triplets(mpcc)]
    end
    # Convert row values adjusting for the number of linear complementarities
    map!((x) -> get_ind_j_lin_row_map(mpcc)[x], rows, rows)

    return rows, cols
end

function NLPModels.jac_nln_structure!(
    mpcc::AbstractMPCCModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    _rows = view(mpcc._i1, 1:get_nln_nnzj(mpcc.nlp))
    _cols = view(mpcc._i2, 1:get_nln_nnzj(mpcc.nlp))
    jac_nln_structure!(mpcc.nlp, _rows, _cols) # get including complementarities
    @views begin
        rows .= _rows[get_ind_j_nln_triplets(mpcc)]
        cols .= _cols[get_ind_j_nln_triplets(mpcc)]
    end
    # Convert row values adjusting for the number of nonlinear complementarities
    map!((x) -> get_ind_j_nln_row_map(mpcc)[x], rows, rows)

    return rows, cols
end

function NLPModels.jac_coord!(mpcc::AbstractMPCCModel, x::AbstractVector, j::AbstractVector)
    jac_coord!(mpcc.nlp, x, mpcc._j1)
    @views j .= mpcc._j1[get_ind_j_triplets(mpcc)]
    return j
end

function NLPModels.jac_coord(mpcc::AbstractMPCCModel{T}, x::AbstractVector) where {T}
    vals = Vector{T}(undef, get_nnzj(mpcc))
    return jac_coord!(mpcc, x, vals)
end

function NLPModels.jac_lin_coord!(
    mpcc::AbstractMPCCModel,
    x::AbstractVector,
    j::AbstractVector,
)
    _j = view(mpcc._j1, 1:get_lin_nnzj(mpcc.nlp))
    jac_lin_coord!(mpcc.nlp, x, _j)
    @views j .= _j[get_ind_j_lin_triplets(mpcc)]
    return j
end

function NLPModels.jac_nln_coord!(
    mpcc::AbstractMPCCModel,
    x::AbstractVector,
    j::AbstractVector,
)
    _j = view(mpcc._j1, 1:get_nln_nnzj(mpcc.nlp))
    jac_lin_coord!(mpcc.nlp, x, _j)
    @views j .= _j[get_ind_j_nln_triplets(mpcc)]
    return j
end

function NLPModels.jprod!(
    mpcc::AbstractMPCCModel,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    Jv[1:get_ncon(mpcc)] .= jac(mpcc, x) * v
    return Jv
end

function NLPModels.jprod_lin!(
    mpcc::AbstractMPCCModel,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way?
    Jv[1:get_nlin(mpcc)] .= jac_lin(mpcc, x) * v
    return Jv
end

function NLPModels.jprod_nln!(
    mpcc::AbstractMPCCModel,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way?
    Jv[1:get_nnln(mpcc)] .= jac_nln(mpcc, x) * v
    return Jv
end

function NLPModels.jtprod!(
    mpcc::AbstractMPCCModel,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way?
    Jtv[1:get_nvar(mpcc)] .= jac(mpcc, x)' * v
    return Jtv
end

function NLPModels.jtprod_lin!(
    mpcc::AbstractMPCCModel,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way?
    Jtv[1:get_nvar(mpcc)] .= jac_lin(mpcc, x)' * v
    return Jtv
end

function NLPModels.jtprod_nln!(
    mpcc::AbstractMPCCModel,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way?
    Jv[1:get_nvar(mpcc)] .= jac_nln(mpcc, x)' * v
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
    mpcc._c1[get_ind_c(mpcc)] .= y
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
    mpcc._c1[get_ind_c(mpcc)] .= y
    return hprod!(mpcc.nlp, x, mpcc._c1, v, Hv; obj_weight=obj_weight)
end

function comp_left(mpcc::AbstractMPCCModel{T, VT}, x::AbstractVector{T}) where {T, VT}
    ccx = VT(undef, get_ncc(mpcc))
    return comp_left!(mpcc, x, ccx)
end

function comp_left!(
    mpcc::AbstractMPCCModel{T, VT},
    x::AbstractVector{T},
    ccx::AbstractVector{T},
) where {T, VT}
    @lencheck get_ncc(mpcc) ccx
    @lencheck get_nvar(mpcc) x
    cvar = 0
    vert = true
    # First get variables:
    for i in 1:get_ncc(mpcc)
        if get_cc_types(mpcc)[i] ∈ [VarVar, VarCon]
            ccx[i] = x[get_ind_cc1(mpcc)[i]]
            cvar += 1
        else
            vert = false
        end
    end
    # TODO(@anton) this should be done via multiple dispatch probably
    if !vert
        # TODO(@anton) I am not sure anymore if this is correct for non-vertical form
        cons!(mpcc.nlp, x, mpcc._c1)
        @views ccx[(cvar+1):end] .= mpcc._c1[get_cc_l(mpcc)]
    end
    return ccx
end

function comp_right(mpcc::AbstractMPCCModel{T, VT}, x::AbstractVector{T}) where {T, VT}
    ccx = VT(undef, get_ncc(mpcc))
    return comp_right!(mpcc, x, ccx)
end

function comp_right!(
    mpcc::AbstractMPCCModel{T, VT},
    x::AbstractVector{T},
    ccx::AbstractVector{T},
) where {T, VT}
    @lencheck get_ncc(mpcc) ccx
    @lencheck get_nvar(mpcc) x
    cvar = 0
    vert = true
    # First get variables:
    for i in 1:get_ncc(mpcc)
        if get_cc_types(mpcc)[i] ∈ [VarVar, ConVar]
            ccx[i] = x[get_ind_cc2(mpcc)[i]]
            cvar += 1
        else
            vert = false
        end
    end

    if !vert
        # TODO(@anton) I am not sure anymore if this is correct for non-vertical form
        cons!(mpcc.nlp, x, mpcc._c1)
        @views ccx[(cvar+1):end] .= mpcc._c1[get_cc_r(mpcc)]
    end
    return ccx
end

function lcomp_left(mpcc::AbstractMPCCModel{T, VT}) where {T, VT}
    lccx = VT(undef, get_ncc(mpcc))
    return lcomp_left!(mpcc, lccx)
end

function lcomp_left!(mpcc::AbstractMPCCModel{T, VT}, lccx::AbstractVector{T}) where {T, VT}
    @lencheck get_ncc(mpcc) lccx

    for i in 1:get_ncc(mpcc)
        if get_cc_types(mpcc)[i] ∈ [VarVar, VarCon]
            lccx[i] = get_lvar(mpcc.nlp)[get_ind_cc1(mpcc)[i]]
        else
            lccx[i] = get_lcon(mpcc.nlp)[get_ind_cc1(mpcc)[i]]
        end
    end
    return lccx
end

function lcomp_right(mpcc::AbstractMPCCModel{T, VT}) where {T, VT}
    lccx = VT(undef, get_ncc(mpcc))
    return lcomp_right!(mpcc, lccx)
end

function lcomp_right!(mpcc::AbstractMPCCModel{T, VT}, lccx::AbstractVector{T}) where {T, VT}
    @lencheck get_ncc(mpcc) lccx

    for i in 1:get_ncc(mpcc)
        if get_cc_types(mpcc)[i] ∈ [VarVar, ConVar]
            lccx[i] = get_lvar(mpcc.nlp)[get_ind_cc2(mpcc)[i]]
        else
            lccx[i] = get_lcon(mpcc.nlp)[get_ind_cc2(mpcc)[i]]
        end
    end
    return lccx
end

function comp_res_left(mpcc::AbstractMPCCModel{T, VT}, x::AbstractVector{T}) where {T, VT}
    lccx = VT(undef, get_ncc(mpcc))
    return comp_res_left!(mpcc, x, lccx)
end

function comp_res_left!(
    mpcc::AbstractMPCCModel{T, VT},
    x::AbstractVector{T},
    lccx::AbstractVector{T},
) where {T, VT}
    @lencheck get_ncc(mpcc) lccx
    @lencheck get_nvar(mpcc) x

    comp_left!(mpcc, x, lccx)

    for i in 1:get_ncc(mpcc)
        if get_cc_types(mpcc)[i] ∈ [VarVar, VarCon]
            lccx[i] -= get_lvar(mpcc.nlp)[get_ind_cc1(mpcc)[i]]
        else
            lccx[i] -= get_lcon(mpcc.nlp)[get_ind_cc1(mpcc)[i]]
        end
    end
    return lccx
end

function comp_res_right(mpcc::AbstractMPCCModel{T, VT}, x::AbstractVector{T}) where {T, VT}
    rccx = similar(mpcc._cc1, get_ncc(mpcc))
    return comp_res_right!(mpcc, x, rccx)
end

function comp_res_right!(
    mpcc::AbstractMPCCModel{T, VT},
    x::AbstractVector{T},
    rccx::AbstractVector{T},
) where {T, VT}
    @lencheck get_ncc(mpcc) rccx
    @lencheck get_nvar(mpcc) x

    comp_right!(mpcc, x, rccx)

    for i in 1:get_ncc(mpcc)
        if get_cc_types(mpcc)[i] ∈ [VarVar, ConVar]
            rccx[i] -= get_lvar(mpcc.nlp)[get_ind_cc2(mpcc)[i]]
        else
            rccx[i] -= get_lcon(mpcc.nlp)[get_ind_cc2(mpcc)[i]]
        end
    end
    return rccx
end

function comp_res_prod!(
    mpcc::AbstractMPCCModel{T, VT},
    x::AbstractVector{T},
    ccx::AbstractVector{T},
) where {T, VT}
    comp_res_left!(mpcc, x, mpcc._cc1)
    comp_res_right!(mpcc, x, mpcc._cc2)
    ccx .= mpcc._cc1 .* mpcc._cc2
    return ccx
end

function jac_comp_left_structure(mpcc::AbstractMPCCModel)
    rows = IndexSet(undef, get_comp_left_nnzj(mpcc))
    cols = IndexSet(undef, get_comp_left_nnzj(mpcc))

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
        rows[1:length(get_ind_j_comp_left_triplets(mpcc))] .=
            _rows[get_ind_j_comp_left_triplets(mpcc)]
        cols[1:length(get_ind_j_comp_left_triplets(mpcc))] .=
            _cols[get_ind_j_comp_left_triplets(mpcc)]
        map!(
            x -> get_ind_j_comp_left_row_map(mpcc)[x],
            rows[1:length(get_ind_j_comp_left_triplets(mpcc))],
            rows[1:length(get_ind_j_comp_left_triplets(mpcc))],
        )
    end

    i_var_comp = length(get_ind_j_comp_left_triplets(mpcc)) + 1
    # TODO(@anton) maybe vectorize
    for i in 1:get_ncc(mpcc)
        if get_cc_types(mpcc)[i] ∈ [VarVar, VarCon]
            rows[i_var_comp] = i;
            cols[i_var_comp] = get_ind_cc1(mpcc)[i]
            i_var_comp += 1
        end
    end
    return rows, cols
end

function jac_comp_right_structure(mpcc::AbstractMPCCModel)
    rows = IndexSet(undef, get_comp_right_nnzj(mpcc))
    cols = IndexSet(undef, get_comp_right_nnzj(mpcc))

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
        rows[1:length(get_ind_j_comp_right_triplets(mpcc))] .=
            _rows[get_ind_j_comp_right_triplets(mpcc)]
        cols[1:length(get_ind_j_comp_right_triplets(mpcc))] .=
            _cols[get_ind_j_comp_right_triplets(mpcc)]

        map!(
            x -> get_ind_j_comp_right_row_map(mpcc)[x],
            rows[1:length(get_ind_j_comp_right_triplets(mpcc))],
            rows[1:length(get_ind_j_comp_right_triplets(mpcc))],
        )
    end

    i_var_comp = length(get_ind_j_comp_right_triplets(mpcc)) + 1
    # TODO(@anton) maybe vectorize
    for i in 1:get_ncc(mpcc)
        if get_cc_types(mpcc)[i] ∈ [VarVar, ConVar]
            rows[i_var_comp] = i;
            cols[i_var_comp] = get_ind_cc2(mpcc)[i]
            i_var_comp += 1
        end
    end
    return rows, cols
end

function jac_comp_left_coord(
    mpcc::AbstractMPCCModel{T, VT},
    x::AbstractVector,
) where {T, VT}
    vals = VT(undef, get_comp_left_nnzj(mpcc))

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
    @views vals[1:length(get_ind_j_comp_left_triplets(mpcc))] .=
        _vals[get_ind_j_comp_left_triplets(mpcc)]

    i_var_comp = length(get_ind_j_comp_left_triplets(mpcc)) + 1
    # TODO(@anton) maybe vectorize
    for i in 1:get_ncc(mpcc)
        if get_cc_types(mpcc)[i] ∈ [VarVar, VarCon]
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
    vals = VT(undef, get_comp_left_nnzj(mpcc))

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
    @views vals[1:length(get_ind_j_comp_right_triplets(mpcc))] .=
        _vals[get_ind_j_comp_right_triplets(mpcc)]

    i_var_comp = length(get_ind_j_comp_right_triplets(mpcc)) + 1
    # TODO(@anton) maybe vectorize
    for i in 1:get_ncc(mpcc)
        if get_cc_types(mpcc)[i] ∈ [VarVar, ConVar]
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
        get_ind_cc1(mpcc)[i] for
        i in 1:get_ncc(mpcc) if get_cc_types(mpcc)[i]∈[ConVar, ConCon]
    ]

    ind_lift1::IndexSet =
        [i for i in 1:get_ncc(mpcc) if get_cc_types(mpcc)[i]∈[ConVar, ConCon]]
    ind_lift2::IndexSet =
        [i for i in 1:get_ncc(mpcc) if get_cc_types(mpcc)[i]∈[VarCon, ConCon]]
    nlift1 = length(ind_lift1)
    nlift2 = length(ind_lift2)

    ind_lift::IndexSet = vcat(
        map((i) -> get_ind_cc1(mpcc)[i], ind_lift1),
        map((i) -> get_ind_cc2(mpcc)[i], ind_lift2),
    )
    vnlp = LiftedNLPModel(mpcc.nlp, ind_lift)

    lift1 = (get_nvar(mpcc.nlp)+1):(get_nvar(mpcc.nlp)+nlift1)
    lift2 = (get_nvar(mpcc.nlp)+nlift1+1):(get_nvar(mpcc.nlp)+nlift1+nlift2)

    ind_vcc1 = get_ind_cc1(mpcc)
    ind_vcc1[ind_lift1] = lift1
    ind_vcc2 = get_ind_cc2(mpcc)
    ind_vcc2[ind_lift2] = lift2

    return MPCCModel(vnlp, ind_vcc1, ind_vcc2)
end
# TODO(@anton) Add Core.show overload

######################## Typed gets #######################
for field in fieldnames(NLPModelMeta) ∪ fieldnames(MPCCModelMeta)
    meth = Symbol("get_", field)
    if field ∈ fieldnames(NLPModelMeta)
        if field ∈ fieldnames(MPCCModelMeta)
            @eval NLPModels.$meth(meta::MPCCModelMeta) =
                getproperty(meta, $(QuoteNode(field)))
        else
            @eval NLPModels.$meth(meta::MPCCModelMeta) = $meth(meta.nlp_meta[])
        end
        @eval NLPModels.$meth(mpcc::AbstractMPCCModel) = $meth(mpcc.meta)
    else
        @eval begin
            @doc """
                  $($meth)(nlp)
                  $($meth)(meta)
                  Return the value $($(QuoteNode(field))) from meta or nlp.meta.
      """
            $meth(meta::MPCCModelMeta) = getproperty(meta, $(QuoteNode(field)))
        end
        @eval $meth(mpcc::AbstractMPCCModel) = $meth(mpcc.meta)
        @eval export $meth
    end
end
