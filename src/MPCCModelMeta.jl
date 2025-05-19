struct MPCCModelMeta{T, VT, MT <: AbstractNLPModelMeta{T, VT}} <:
       AbstractNLPModelMeta{T, VT}
    nlp_meta::Base.RefValue{MT}

    ncc::Int
    ncon::Int # This may or may not be different depending on the type of constraints
    nlin::Int # This may or may not be different depending on the type of constraints
    nnln::Int # This may or may not be different depending on the type of constraints

    nnzj::Int
    lin_nnzj::Int
    nln_nnzj::Int
    comp_left_nnzj::Int
    comp_right_nnzj::Int

    lin::IndexSet
    nln::IndexSet

    c_lin::IndexSet
    c_nln::IndexSet
    cc_l::IndexSet
    cc_r::IndexSet

    # Index Sets of complementarity variables
    ind_cc1::IndexSet
    ind_cc2::IndexSet
    cc_types::Vector{CCType} # VarCon, VarVar, ConCon

    # Index Sets of noncomplemntarity variables and constraints
    ind_x::IndexSet
    ind_c::IndexSet

    # Index set of the jacobian triplets to keep.
    ind_j_triplets::IndexSet
    ind_j_lin_triplets::IndexSet
    ind_j_nln_triplets::IndexSet
    ind_j_comp_left_triplets::IndexSet
    ind_j_comp_right_triplets::IndexSet
    ind_j_comp_left_row_map::Dict{Int, Int}
    ind_j_comp_right_row_map::Dict{Int, Int}

    ind_j_lin_row_map::Dict{Int, Int}
    ind_j_nln_row_map::Dict{Int, Int}
end

function Base.getproperty(meta::MPCCModelMeta, sym::Symbol)
    if sym ∈ fieldnames(MPCCModelMeta) # NOTE: This is either elegant or EXTREMELY not, depending on how "static" the field names are
        getfield(meta, sym)
    else
        getproperty(meta.nlp_meta[], sym)
    end
end
