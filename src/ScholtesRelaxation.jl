######################### Scholtes Relaxation #########################
struct ScholtesRelaxation{T, VT} <: NLPModels.AbstractNLPModel{T, VT}
    mpcc::AbstractMPCCModel{T, VT}
    meta::NLPModels.NLPModelMeta{T,VT}
    𝜎::Base.RefValue{T}
end

function ScholtesRelaxation(mpcc::AbstractMPCCModel{T,VT}) where {T,VT}
    if !is_vertical(mpcc)
        # TODO(@anton) Perhaps we should do this automatically in the future or we can support non-vertical form scholtes
        #              though this makes the callbacks a bit more complicated
        error("Scholtes Relaxation currently expects a vertical form MPCC, use vertical_form(mpcc) to convert it.")
    end

    # Update only what needs to be updated
    ncon = mpcc.meta.ncon + mpcc.meta.ncc
    lcon = append!(mpcc.meta.lcon, -Inf*ones(mpcc.meta.ncc))
    ucon = append!(mpcc.meta.ucon, zeros(mpcc.meta.ncc))
    y0 = append!(mpcc.meta.y0, zeros(mpcc.meta.ncc))
    # TODO(@anton) this is an upper bound: we can calculate this exactly if we store
    #              the nnz for the jacobians of G, and H
    nnzj = mpcc.meta.nnzj + 2*mpcc.meta.ncc
    nln_nnzj = mpcc.meta.nln_nnzj + 2*mpcc.meta.ncc

    # TODO(@anton) this is a bug actually. we need to check the structure of the mpcc (and the underlying nlp) to
    #              figure out if the nnzh is correct as if the off diagonals are not already in the nonzeros.
    nnzh = mpcc.meta.nnzh + mpcc.meta.ncc
    # TODO(@anton) We may need to change how nlv(b,o,c) are handled because we actually cannot
    #              backcalculate how these need to change necessarily.
    #              However these seem to not be used anywhere in the NLPModels API so I am ignoring them.

    meta = NLPModels.NLPModelMeta(mpcc.nlp.meta,
                                  ncon=ncon,
                                  lcon=lcon,
                                  y0=y0,
                                  nnzj=nnzj,
                                  nln_nnzj=nln_nnzj,
                                  nnzh=nnzh)
    𝜎= zero(T)
    ScholtesRelaxation(mpcc, meta, Ref(𝜎))
end

# Counters should be forwarded
function Base.getproperty(rnlp::ScholtesRelaxation, sym::Symbol)
    if sym ∈ [:counters]
        getfield(rnlp.mpcc.nlp, sym)
    else
        getfield(rnlp, sym)
    end
end

######################### NLPModels Callbacks #########################

function NLPModels.cons_lin!(rnlp::ScholtesRelaxation{T,VT}, x::VT, cx::VT) where {T, VT}
    if get_ncon(rnlp.mpcc.nlp) > 0
        return cons_lin!(rnlp.mpcc, x, cx)
    else
        return cx
    end
end

function NLPModels.cons_nln!(rnlp::ScholtesRelaxation{T,VT}, x::VT, cx::VT) where {T, VT}
    mpcc_nnln = rnlp.mpcc.meta.nnln
    # This if statement is necessary as it seems that without it c!(cx,x) does not exist in a possible underlying ADNLPModel
    if get_ncon(rnlp.mpcc.nlp) > 0
        cons_nln!(rnlp.mpcc, x, view(cx, 1:mpcc_nnln))
    end
    # TODO(@anton) figure out if the intermediate outputs cause allocations
    cx[(mpcc_nnln+1):(rnlp.meta.nnln)] = (comp_left(rnlp.mpcc, x) .- lcomp_left(rnlp.mpcc)).*(comp_right(rnlp.mpcc, x) .- lcomp_right(rnlp.mpcc)) .- rnlp.𝜎[]
end

function NLPModels.jac_lin_structure!(mpcc::ScholtesRelaxation, rows::Vector{Int}, cols::Vector{Int}) where {T, VT}
    jac_lin_structure!(mpcc, rows, cols) # get including complementarities
    return rows, cols
end

function NLPModels.jac_nln_structure!(mpcc::ScholtesRelaxation, rows::Vector{Int}, cols::Vector{Int}) where {T, VT}
    jac_nln_structure!(mpcc, rows, cols) # get including complementarities


    return rows, cols
end

function NLPModels.jac_lin_coord!(mpcc::ScholtesRelaxation, x::VT, j::VT) where {T, VT}
    return j
end
function NLPModels.jac_nln_coord!(mpcc::ScholtesRelaxation, x::VT, j::VT) where {T, VT}
    jac_nln_coord!(mpcc.nlp, x, j)

    keepat!(j, mpcc.meta.ind_j_nln_structure)
    return j
end

function NLPModels.jprod_lin!(mpcc::ScholtesRelaxation, x::VT, v::VT, jv::VT) where {T, VT}
    mjv = MappedVector(jv, mpcc.meta.c_lin, mpcc.nlp.meta.nlin)
    jprod_lin!(mpcc.nlp, x, v, mjv)
    return jv
end

function NLPModels.jprod_nln!(mpcc::ScholtesRelaxation, x::VT, v::VT, jv::VT) where {T, VT}
    mjv = MappedVector(jv, mpcc.meta.c_nln, mpcc.nlp.meta.nnln)
    jprod_lin!(mpcc.nlp, x, v, mjv)
    return jv
end

function NLPModels.jtprod_lin!(mpcc::ScholtesRelaxation, x::VT, v::VT, jtv::VT) where {T, VT}
    error("not implemented")
    # TODO(@anton) this is not correct
    v[setdiff(1:mpcc.meta.ncon, mpcc.meta.ind_c)] = 0 # TODO(@anton) this is not a great solution
    jtprod_lin!(mpcc.nlp, x, v, jtv)

    return jtv
end

function NLPModels.jtprod_nln!(mpcc::ScholtesRelaxation, x::VT, v::VT, jtv::VT) where {T, VT}
    error("not implemented")
    # TODO(@anton) this is not correct
    jprod_lin!(mpcc.nlp, x, v, jv)

    keepat!(jv, mpcc.meta.ind_c)
    return jv
end

function NLPModels.hess_structure!(mpcc::ScholtesRelaxation, rows::Vector{Int}, cols::Vector{Int}) where {T, VT}
    return hess_structure!(mpcc.nlp, rows, cols)
end
function NLPModels.hess_coord!(mpcc::ScholtesRelaxation{T,VT}, x::VT, y::VT, Hv::VT, obj_weight::Real = one(T)) where {T, VT}
    return hess_coord!(mpcc.nlp, x, y, Hv, obj_weight)
end
