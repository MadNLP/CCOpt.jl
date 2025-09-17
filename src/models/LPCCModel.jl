######################### types #########################
abstract type AbstractLPCCModel{T, VT} <: AbstractMPCCModel{T, VT} end

struct LPCCModel{T, VT} <: AbstractLPCCModel{T, VT}
    nlp::NLPModels.AbstractNLPModel{T, VT}
    meta::MPCCModelMeta{T, VT}
    fixed_map::Vector{Int}
end

######################### linearize #########################
# TODO(@anton) cleanup allocations here.
function linearize(mpcc::AbstractMPCCModel{T, VT}, x::VT; tr=0.0) where {T, VT}
    if !is_vertical(mpcc)
        error("linearization is supported for only vertical form MPCCs")
    end
    # Linearize the underiyling nlp
    lp = QuadraticModels.linearize(mpcc.nlp, x; tr=tr)

    # Do trust region prunning
    fixed_map = zeros(Int, mpcc.meta.ncc)
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

    meta = MPCCModelMeta(
        Ref(lp.meta),
        mpcc.meta.ncc - n_fixed,
        mpcc.meta.ncon,
        mpcc.meta.ncon, # nlin
        0, # nnln
        mpcc.meta.nnzj,
        mpcc.meta.nnzj, # lin_nnzj
        0, # nln_nnzj
        mpcc.meta.comp_left_nnzj,
        mpcc.meta.comp_right_nnzj,
        sort(union(mpcc.meta.lin, mpcc.meta.nln)), # lin TODO(@anton): is sort needed?
        IndexSet(), # nln
        sort(union(mpcc.meta.c_lin, mpcc.meta.c_nln)), # c_lin TODO(@anton) is sort needed?
        IndexSet(), # c_nln
        mpcc.meta.cc_l,
        mpcc.meta.cc_r,
        ind_cc1,
        ind_cc2,
        cc_types,
        ind_x,
        mpcc.meta.ind_c,
        mpcc.meta.ind_j_triplets,
        sort(union(mpcc.meta.ind_j_lin_triplets, mpcc.meta.ind_j_nln_triplets)), # ind_j_lin_triplets
        IndexSet(), # ind_j_nln_triplets
        mpcc.meta.ind_j_comp_left_triplets,
        mpcc.meta.ind_j_comp_right_triplets,
        mpcc.meta.ind_j_comp_left_row_map,
        mpcc.meta.ind_j_comp_right_row_map,
        merge(mpcc.meta.ind_j_lin_row_map, mpcc.meta.ind_j_nln_row_map), # ind_j_lin_row_map TODO(@anton) merge?
        Dict{Int, Int}(), #ind_j_nln_row_map
    )

    return LPCCModel(lp, meta, fixed_map)
end

######################### linearize! #########################
# update an lpcc with data in place
function linearize!(
    lpcc::AbstractMPCCModel{T, VT},
    mpcc::AbstractMPCCModel{T, VT},
    x::VT;
    tr=0.0,
) where {T, VT}
    # Reimplement linearize for quadratic model here because
    # we need tighter control of the nlp meta
    QuadraticModels.linearize!(lpcc.nlp, mpcc.nlp, x; tr=tr)

    return lpcc
end
