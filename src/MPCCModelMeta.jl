struct MPCCModelMeta{T, VT, MT<: AbstractNLPModelMeta{T, VT}} <: AbstractNLPModelMeta{T, VT}
    nlp_meta::Base.RefValue{MT}

    ncc::Int
    ncon::Int # This may or may not be different depending on the type of constraints
    nlin::Int # This may or may not be different depending on the type of constraints
    nnln::Int # This may or may not be different depending on the type of constraints

    lin::IndexSet
    nln::IndexSet

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
    if sym ∈ fieldnames(MPCCModelMeta) # NOTE: This is either elegant or EXTREMELY not, depending on how "static" the field names are
        getfield(meta, sym)
    else
        getfield(meta.nlp_meta[], sym)
    end
end
