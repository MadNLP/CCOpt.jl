######################### linearize #########################
function linearize(mpcc::AbstractMPCCModel{T, VT}, x::VT; tr=0.0) where {T, VT}
    lp = QuadraticModels.linearize(mpcc.nlp, x; tr=tr)

    meta = MPCCModelMeta(
        Ref(lp.meta),
        mpcc.meta.ncc,
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
        mpcc.meta.ind_cc1,
        mpcc.meta.ind_cc2,
        mpcc.meta.cc_types,
        mpcc.meta.ind_x,
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

    return MPCCModel(lp, meta)
end

######################### linearize! #########################
# update an lpcc with data in place
function linearize!(
    lpcc::AbstractMPCCModel{T, VT},
    mpcc::AbstractMPCCModel{T, VT},
    x::VT;
    tr=0.0,
) where {T, VT}
    lp = QuadraticModels.linearize(mpcc.nlp, x; tr=tr)

    meta = MPCCModelMeta(
        Ref(lp.meta),
        mpcc.meta.ncc,
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
        mpcc.meta.ind_cc1,
        mpcc.meta.ind_cc2,
        mpcc.meta.cc_types,
        mpcc.meta.ind_x,
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

    return MPCCModel(lp, meta)
end
