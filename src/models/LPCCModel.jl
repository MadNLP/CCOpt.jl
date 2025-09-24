# TODO(@anton) this can actually in principle piggyback off of the already evaluated values
#              in the solver. Think about efficiencies here.
######################### types #########################
abstract type AbstractLPCCModel{T, VT} <: AbstractMPCCModel{T, VT} end

struct LPCCModel{T, VT} <: AbstractLPCCModel{T, VT}
    nlp::NLPModels.AbstractNLPModel{T, VT}
    meta::MPCCModelMeta{T, VT}
    fixed_map::Vector{Int}
    A::SparseMatrixCOO{T}
    klasttouch::Vector{Int}
    csrrowptr::Vector{Int}
    csrcolval::Vector{Int}
    csrnzval::Vector{T}
end

######################### linearize #########################
# TODO(@anton) cleanup allocations here.
function linearize(mpcc::AbstractMPCCModel{T, VT}, x::VT; tr=0.0) where {T, VT}
    if !is_vertical(mpcc)
        error("linearization is supported for only vertical form MPCCs")
    end
    # Linearize the underiyling nlp
    lp = QuadraticModels.linearize(mpcc.nlp, x; tr=tr)
    m, n = size(lp.data.A)

    # Build inplace COO matrix
    A = SparseMatrixCOO(
        m,
        n,
        Vector{Int}(undef, mpcc.nlp.meta.nnzj),
        Vector{Int}(undef, mpcc.nlp.meta.nnzj),
        VT(undef, mpcc.nlp.meta.nnzj),
    )
    jac_structure!(mpcc.nlp, A.rows, A.cols)

    # build inplace vectors
    klasttouch = Vector{Int}(undef, n)
    csrrowptr = Vector{Int}(undef, m + 1)
    csrcolval = Vector{Int}(undef, length(A.rows))
    csrnzval = VT(undef, length(A.rows))

    ## CREATE NEW META ##
    ind_cc1 = mpcc.meta.ind_cc1
    ind_cc2 = mpcc.meta.ind_cc2
    # compute sizes
    ncc = length(mpcc.meta.ind_cc1)
    ncon = lp.meta.ncon
    nlin = lp.meta.nlin
    nnln = lp.meta.nnln

    # Do trust region prunning
    fixed_map = zeros(Int, mpcc.meta.ncc)
    n_fixed = 0
    if tr > 0.0
        for ii in 1:ncc
            icc1 = ind_cc1[ii]
            icc2 = ind_cc2[ii]
            # Assumes at least one branch is in the trust region
            if x[icc1] - tr > mpcc.nlp.meta.lvar[icc1]
                n_fixed += 1
                fixed_map[ii] = 2
                # update bounds in the lp
                lp.meta.uvar[icc2] = lp.meta.lvar[icc2]
                lp.meta.x0[icc2] = lp.meta.lvar[icc2]
            elseif x[icc2] - tr > mpcc.nlp.meta.lvar[icc2]
                n_fixed += 1
                fixed_map[ii] = 1
                # update bounds in the lp by fixing the variable
                lp.meta.uvar[icc1] = lp.meta.lvar[icc1]
                lp.meta.x0[icc1] = lp.meta.lvar[icc1]
            end
        end
    end
    # Prune the new ind_cc
    ind_cc1 = ind_cc1[fixed_map .== 0]
    ind_cc2 = ind_cc2[fixed_map .== 0]
    cc_types = mpcc.meta.cc_types[fixed_map .== 0]

    # compute non-complementarity variables/constraints
    ind_c = collect(1:lp.meta.ncon)
    ind_x = setdiff(1:mpcc.meta.nvar, union(ind_cc1, ind_cc2))

    # compute jacobian structure indexset
    ind_j_triplets = collect(1:lp.meta.nnzj)
    ind_j_lin_triplets = collect(1:lp.meta.lin_nnzj)
    ind_j_nln_triplets = collect(1:lp.meta.nln_nnzj)
    ind_j_lin_row_map = Dict(zip(1:nlin, 1:nlin))
    ind_j_nln_row_map = Dict(zip(1:nnln, 1:nnln))

    ind_j_comp_left_triplets::IndexSet = []
    ind_j_comp_right_triplets::IndexSet = []
    ind_j_comp_left_row_map = Dict{Int, Int}()
    ind_j_comp_right_row_map = Dict{Int, Int}()
    # compute nln and lin index sets
    lin = lp.meta.lin
    nln = lp.meta.nln
    nlin = length(lin)
    nnln = length(nln)
    c_lin = collect(1:nlin)
    c_nln = collect(1:nnln)
    cc_l::IndexSet = []
    cc_r::IndexSet = []

    # nnzj updates:
    nnzj = lp.meta.nnzj
    lin_nnzj = lp.meta.lin_nnzj
    nln_nnzj = lp.meta.nln_nnzj
    comp_left_nnzj = ncc
    comp_right_nnzj = ncc

    meta = MPCCModelMeta(
        Ref(lp.meta),
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
    return LPCCModel(lp, meta, fixed_map, A, klasttouch, csrrowptr, csrcolval, csrnzval)
end

######################### linearize! #########################
# update an lpcc with data in place
# TODO(@anton) cleanup allocations here.
function linearize!(
    lpcc::AbstractMPCCModel{T, VT},
    mpcc::AbstractMPCCModel{T, VT},
    x::VT;
    tr=0.0,
) where {T, VT}
    # Reimplement linearize for quadratic model here because
    # we need tighter control of the nlp meta
    QuadraticModels.linearize!(
        lpcc.nlp,
        mpcc.nlp,
        x,
        lpcc.A,
        lpcc.klasttouch,
        lpcc.csrrowptr,
        lpcc.csrcolval,
        lpcc.csrnzval;
        tr=tr,
    )
    lp = lpcc.nlp

    fixed_map = lpcc.fixed_map
    fixed_map .= 0
    n_fixed = 0
    if tr > 0.0
        for ii in 1:mpcc.meta.ncc
            icc1 = mpcc.meta.ind_cc1[ii]
            icc2 = mpcc.meta.ind_cc2[ii]
            # Assumes at least one branch is in the trust region
            if x[icc1] - tr > mpcc.nlp.meta.lvar[icc1]
                n_fixed += 1
                fixed_map[ii] = 2
                # update bounds in the lp
                lp.meta.uvar[icc2] = lp.meta.lvar[icc2]
                lp.meta.x0[icc2] = lp.meta.lvar[icc2]
            elseif x[icc2] - tr > mpcc.nlp.meta.lvar[icc2]
                n_fixed += 1
                fixed_map[ii] = 1
                # update bounds in the lp by fixing the variable
                lp.meta.uvar[icc1] = lp.meta.lvar[icc1]
                lp.meta.x0[icc1] = lp.meta.lvar[icc1]
            end
        end
    end
    # Prune the new ind_cc
    ind_cc1 = mpcc.meta.ind_cc1[fixed_map .== 0]
    ind_cc2 = mpcc.meta.ind_cc2[fixed_map .== 0]
    cc_types = mpcc.meta.cc_types[fixed_map .== 0]
    ind_x = setdiff(1:mpcc.meta.nvar, union(ind_cc1, ind_cc2))

    lpcc.meta.ncc = mpcc.meta.ncc - n_fixed
    lpcc.meta.ind_cc1 = ind_cc1
    lpcc.meta.ind_cc2 = ind_cc2
    lpcc.meta.cc_types = cc_types
    lpcc.meta.ind_x = ind_x
    return lpcc
end

function tr!(
    lpcc::AbstractMPCCModel{T, VT},
    mpcc::AbstractMPCCModel{T, VT},
    x::VT;
    tr=0.0,
) where {T, VT}
    # Reimplement linearize for quadratic model here because
    # we need tighter control of the nlp meta
    lp = lpcc.nlp
    lp.meta.lvar .= max.(mpcc.meta.lvar .- x, -tr)
    lp.meta.uvar .= max.(mpcc.meta.uvar .- x, tr)

    fixed_map = lpcc.fixed_map
    fixed_map .= 0
    n_fixed = 0
    if tr > 0.0
        for ii in 1:mpcc.meta.ncc
            icc1 = mpcc.meta.ind_cc1[ii]
            icc2 = mpcc.meta.ind_cc2[ii]
            # Assumes at least one branch is in the trust region
            if x[icc1] - tr > mpcc.nlp.meta.lvar[icc1]
                n_fixed += 1
                fixed_map[ii] = 2
                # update bounds in the lp
                lp.meta.uvar[icc2] = lp.meta.lvar[icc2]
                lp.meta.x0[icc2] = lp.meta.lvar[icc2]
            elseif x[icc2] - tr > mpcc.nlp.meta.lvar[icc2]
                n_fixed += 1
                fixed_map[ii] = 1
                # update bounds in the lp by fixing the variable
                lp.meta.uvar[icc1] = lp.meta.lvar[icc1]
                lp.meta.x0[icc1] = lp.meta.lvar[icc1]
            end
        end
    end
    # Prune the new ind_cc
    ind_cc1 = mpcc.meta.ind_cc1[fixed_map .== 0]
    ind_cc2 = mpcc.meta.ind_cc2[fixed_map .== 0]
    cc_types = mpcc.meta.cc_types[fixed_map .== 0]
    ind_x = setdiff(1:mpcc.meta.nvar, union(ind_cc1, ind_cc2))

    lpcc.meta.ncc = mpcc.meta.ncc - n_fixed
    lpcc.meta.ind_cc1 = ind_cc1
    lpcc.meta.ind_cc2 = ind_cc2
    lpcc.meta.cc_types = cc_types
    lpcc.meta.ind_x = ind_x

    return lpcc
end
