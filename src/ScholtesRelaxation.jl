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
    nnzh = mpcc.meta.nnzh + 2*mpcc.meta.ncc
    # TODO(@anton) We may need to change how nlv(b,o,c) are handled because we actually cannot
    #              backcalculate how these need to change necessarily.

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
