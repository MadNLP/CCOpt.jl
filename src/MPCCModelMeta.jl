# TODO(@anton) Most of the index sets should perhaps live in a metadata field as in NLPModels
struct MPCCModelMeta{T, VT} <: AbstractNLPModelMeta{T, VT}
    nlp_meta::Ref{AbstractNLPModelMeta{T, VT}}

    ncc::Int

    # Index Sets of complementarity variables
    ind_vcc1::IndexSet
    ind_vcc2::IndexSet
    ind_ccc1::IndexSet
    ind_ccc2::IndexSet
    ind_cc1::IndexSet
    ind_cc2::IndexSet
    cc_types::Vector{CCType} # VarCon, VarVar, ConCon

    # Index Sets of noncomplemntarity variables and constraints
    ind_x::IndexSet
    ind_c::IndexSet

    # Index set of the jacobian triplets to keep.
    ind_j_lin_triplets::IndexSet
    ind_j_nln_triplets::IndexSet
end

function Base.getproperty(meta::MPCCModelMeta, sym::Symbol)
    if sym ∈ [:ind_vcc1, :ind_vcc2,
              :ind_ccc1, :ind_ccc2,
              :ind_cc1, :ind_cc2,
              :ind_x, :ind_c,
              :ind_j_lin_triplets,
              :ind_j_nln_triplets]
        getfield(meta, sym)
    else
        getfield(meta.nlp_meta[], sym)
    end
end
