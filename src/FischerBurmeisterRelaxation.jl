######################### FischerBurmeister Relaxation #########################
struct FischerBurmeisterRelaxation{T, VT} <: AbstractMPCCRelaxation{T, VT}
    mpcc::AbstractMPCCModel{T, VT}
    meta::NLPModels.NLPModelMeta{T, VT}
    cc1_buf::VT
    cc2_buf::VT
    σ::Base.RefValue{T}
end

function FischerBurmeisterRelaxation(mpcc::AbstractMPCCModel{T, VT}) where {T, VT}
    if !is_vertical(mpcc)
        # TODO(@anton) Perhaps we should do this automatically in the future or we can support non-vertical form scholtes
        #              though this makes the callbacks a bit more complicated
        error(
            "Fischer-Burmeister Relaxation currently expects a vertical form MPCC, use vertical_form(mpcc) to convert it.",
        )
    end

    # Update only what needs to be updated
    ncon = mpcc.meta.ncon + mpcc.meta.ncc
    lcon = vcat(mpcc.meta.lcon, -typemax(T)*ones(T, mpcc.meta.ncc))
    ucon = vcat(mpcc.meta.ucon, zeros(T, mpcc.meta.ncc))
    y0 = vcat(mpcc.meta.y0, zeros(T, mpcc.meta.ncc))
    # TODO(@anton) this is a lower bound only accurate for vertical form: we can calculate this exactly if we store
    #              the nnz for the jacobians of G, and H, which we now do.
    nnzj = mpcc.meta.nnzj + 2*mpcc.meta.ncc
    nln_nnzj = mpcc.meta.nln_nnzj + 2*mpcc.meta.ncc # All the nln values are

    # TODO(@anton) this is a bug actually. we need to check the structure of the mpcc (and the underlying nlp) to
    #              figure out if the nnzh is correct as if the off diagonals are not already in the nonzeros.
    #
    # TODO(@anton) This may or may not break the assumptions made by show(::NLPModelMeta)
    nnzh = mpcc.meta.nnzh + 3*mpcc.meta.ncc
    # TODO(@anton) We may need to change how nlv(b,o,c) are handled because we actually cannot
    #              backcalculate how these need to change necessarily.
    #              However these seem to not be used anywhere in the NLPModels API so I am ignoring them.

    meta = NLPModels.NLPModelMeta(
        mpcc.nlp.meta,
        ncon=ncon,
        lcon=lcon,
        ucon=ucon,
        y0=y0,
        nnzj=nnzj,
        nln_nnzj=nln_nnzj,
        nnzh=nnzh,
    )

    cc1_buf = VT(undef, mpcc.meta.ncc)
    cc2_buf = VT(undef, mpcc.meta.ncc)

    σ = zero(T)
    return FischerBurmeisterRelaxation(mpcc, meta, cc1_buf, cc2_buf, Ref(σ))
end

# Counters should be forwarded
function Base.getproperty(rnlp::FischerBurmeisterRelaxation, sym::Symbol)
    if sym ∈ [:counters]
        getproperty(rnlp.mpcc.nlp, sym)
    else
        getfield(rnlp, sym)
    end
end

######################### NLPModels Callbacks #########################
function NLPModels.obj(rnlp::FischerBurmeisterRelaxation, x::AbstractVector)
    return NLPModels.obj(rnlp.mpcc, x)
end

function NLPModels.grad!(
    rnlp::FischerBurmeisterRelaxation,
    x::AbstractVector,
    gx::AbstractVector,
)
    return NLPModels.grad!(rnlp.mpcc, x, gx)
end

function NLPModels.objgrad!(
    rnlp::FischerBurmeisterRelaxation,
    x::AbstractVector,
    g::AbstractVector,
)
    return NLPModels.objgrad!(rnlp.mpcc, x, g)
end

function NLPModels.cons!(
    rnlp::FischerBurmeisterRelaxation,
    x::AbstractVector,
    cx::AbstractVector,
)
    mpcc_ncon = rnlp.mpcc.meta.ncon
    if get_ncon(rnlp.mpcc.nlp) > 0
        cons!(rnlp.mpcc, x, view(cx, 1:mpcc_ncon))
    end
    comp_res_left!(rnlp.mpcc, x, rnlp.cc1_buf)
    comp_res_right!(rnlp.mpcc, x, rnlp.cc2_buf)
    @views(
        map!(
            (a, b) -> a + b - sqrt(a^2 + b^2 + 2*rnlp.σ[]),
            cx[(mpcc_ncon+1):(rnlp.meta.ncon)],
            rnlp.cc1_buf,
            rnlp.cc2_buf,
        )
    )
    return cx
end

function NLPModels.cons_lin!(
    rnlp::FischerBurmeisterRelaxation,
    x::AbstractVector,
    cx::AbstractVector,
)
    if get_ncon(rnlp.mpcc.nlp) > 0
        return cons_lin!(rnlp.mpcc, x, cx)
    else
        return cx
    end
end

function NLPModels.cons_nln!(
    rnlp::FischerBurmeisterRelaxation,
    x::AbstractVector,
    cx::AbstractVector,
)
    mpcc_nnln = rnlp.mpcc.meta.nnln
    # This if statement is necessary as it seems that without it c!(cx,x) does not exist in a possible underlying ADNLPModel
    if get_ncon(rnlp.mpcc.nlp) > 0
        cons_nln!(rnlp.mpcc, x, view(cx, 1:mpcc_nnln))
    end
    # TODO(@anton) figure out if the intermediate outputs cause allocations
    comp_res_left!(rnlp.mpcc, x, rnlp.cc1_buf)
    comp_res_right!(rnlp.mpcc, x, rnlp.cc2_buf)
    @views(
        map!(
            (a, b) -> a + b - sqrt(a^2 + b^2 + 2*rnlp.σ[]),
            cx[(mpcc_nnln+1):(rnlp.meta.nnln)],
            rnlp.cc1_buf,
            rnlp.cc2_buf,
        )
    )
    return cx
end

function NLPModels.jac_structure!(
    rnlp::FischerBurmeisterRelaxation,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    @views jac_structure!(
        rnlp.mpcc,
        rows[1:rnlp.mpcc.meta.nnzj],
        cols[1:rnlp.mpcc.meta.nnzj],
    ) # get including complementarities

    for i in 1:rnlp.mpcc.meta.ncc
        rows[i+rnlp.mpcc.meta.nnzj] = i + rnlp.mpcc.meta.ncon
        cols[i+rnlp.mpcc.meta.nnzj] = rnlp.mpcc.meta.ind_cc1[i]
    end
    for i in 1:rnlp.mpcc.meta.ncc
        rows[i+rnlp.mpcc.meta.nnzj+rnlp.mpcc.meta.ncc] = i + rnlp.mpcc.meta.ncon
        cols[i+rnlp.mpcc.meta.nnzj+rnlp.mpcc.meta.ncc] = rnlp.mpcc.meta.ind_cc2[i]
    end

    return rows, cols
end

function NLPModels.jac_lin_structure!(
    rnlp::FischerBurmeisterRelaxation,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    jac_lin_structure!(rnlp.mpcc, rows, cols) # get including complementarities
    return rows, cols
end

function NLPModels.jac_nln_structure!(
    rnlp::FischerBurmeisterRelaxation,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    @views jac_nln_structure!(
        rnlp.mpcc,
        rows[1:rnlp.mpcc.meta.nln_nnzj],
        cols[1:rnlp.mpcc.meta.nln_nnzj],
    ) # get including complementarities

    for i in 1:rnlp.mpcc.meta.ncc
        rows[i+rnlp.mpcc.meta.nln_nnzj] = i + rnlp.mpcc.meta.nnln
        cols[i+rnlp.mpcc.meta.nln_nnzj] = rnlp.mpcc.meta.ind_cc1[i]
    end
    for i in 1:rnlp.mpcc.meta.ncc
        rows[i+rnlp.mpcc.meta.nln_nnzj+rnlp.mpcc.meta.ncc] = i + rnlp.mpcc.meta.nnln
        cols[i+rnlp.mpcc.meta.nln_nnzj+rnlp.mpcc.meta.ncc] = rnlp.mpcc.meta.ind_cc2[i]
    end

    return rows, cols
end

function NLPModels.jac_coord!(
    rnlp::FischerBurmeisterRelaxation,
    x::AbstractVector,
    j::AbstractVector,
)
    # TODO(@anton) might be useful to special case the devision operation at exactly 0
    #              or in some region around 0.
    @views begin
        jac_coord!(rnlp.mpcc, x, j[1:rnlp.mpcc.meta.nnzj])
        comp_res_left!(rnlp.mpcc, x, rnlp.cc1_buf)
        comp_res_right!(rnlp.mpcc, x, rnlp.cc2_buf)
        map!(
            (a, b) -> 1 - a/(sqrt(a^2 + b^2)),
            j[(rnlp.mpcc.meta.nnzj+1):(rnlp.mpcc.meta.nnzj+rnlp.mpcc.meta.ncc)],
            rnlp.cc1_buf,
            rnlp.cc2_buf,
        )
        map!(
            (a, b) -> 1 - b/(sqrt(a^2 + b^2)),
            j[(rnlp.mpcc.meta.nnzj+rnlp.mpcc.meta.ncc+1):(rnlp.mpcc.meta.nnzj+2*rnlp.mpcc.meta.ncc)],
            rnlp.cc1_buf,
            rnlp.cc2_buf,
        )
    end
    return j
end

function NLPModels.jac_lin_coord!(
    rnlp::FischerBurmeisterRelaxation,
    x::AbstractVector,
    jac::AbstractVector,
)
    return jac_lin_coord!(rnlp.mpcc, x, jac)
end

function NLPModels.jac_nln_coord!(
    rnlp::FischerBurmeisterRelaxation,
    x::AbstractVector,
    jac::AbstractVector,
)
    # TODO(@anton) might be useful to special case the devision operation at exactly 0
    #              or in some region around 0.
    @views begin
        jac_coord!(rnlp.mpcc, x, j[1:rnlp.mpcc.meta.nln_nnzj])
        comp_res_left!(rnlp.mpcc, x, rnlp.cc1_buf)
        comp_res_right!(rnlp.mpcc, x, rnlp.cc2_buf)
        map!(
            (a, b) -> 1 - a/(sqrt(a^2 + b^2)),
            jac[(rnlp.mpcc.meta.nln_nnzj+1):(rnlp.mpcc.meta.nln_nnzj+rnlp.mpcc.meta.ncc)],
            rnlp.cc1_buf,
            rnlp.cc2_buf,
        )
        map!(
            (a, b) -> 1 - b/(sqrt(a^2 + b^2)),
            jac[(rnlp.mpcc.meta.nln_nnzj+rnlp.mpcc.meta.ncc+1):(rnlp.mpcc.meta.nln_nnzj+2*rnlp.mpcc.meta.ncc)],
            rnlp.cc1_buf,
            rnlp.cc2_buf,
        )
    end
    return jac
end

function NLPModels.jprod_lin!(
    rnlp::FischerBurmeisterRelaxation,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jv[1:rnlp.meta.nlin] = jac_lin(rnlp, x) * v
    return Jv
end

function NLPModels.jprod_nln!(
    rnlp::FischerBurmeisterRelaxation,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jv[1:rnlp.meta.nnln] = jac_nln(rnlp, x) * v
    return Jv
end

function NLPModels.jtprod_lin!(
    rnlp::FischerBurmeisterRelaxation,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jtv[1:rnlp.meta.nvar] = jac_lin(rnlp, x)' * v
    return Jtv
end

function NLPModels.jtprod_nln!(
    rnlp::FischerBurmeisterRelaxation,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jtv[1:rnlp.meta.nvar] = jac_nln(rnlp, x)' * v
    return Jtv
end

function NLPModels.hess_structure!(
    rnlp::FischerBurmeisterRelaxation,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    @views hess_structure!(
        rnlp.mpcc,
        rows[1:rnlp.mpcc.meta.nnzh],
        cols[1:rnlp.mpcc.meta.nnzh],
    )
    # TODO(@anton) it seems hard to vectorize in one operation this because there is no efficient unzip in Base:
    #              See https://github.com/JuliaLang/julia/issues/13942 for details
    nnzh = rnlp.mpcc.meta.nnzh
    ncc = rnlp.mpcc.meta.ncc
    # Off diagonal terms
    for i in 1:ncc
        cols[i+nnzh], rows[i+nnzh] =
            minmax(rnlp.mpcc.meta.ind_cc1[i], rnlp.mpcc.meta.ind_cc2[i])
    end
    # Diagonal terms
    for i in 1:ncc
        cols[i+nnzh+ncc], rows[i+nnzh+ncc] =
            rnlp.mpcc.meta.ind_cc1[i], rnlp.mpcc.meta.ind_cc1[i]
    end
    for i in 1:ncc
        cols[i+nnzh+2*ncc], rows[i+nnzh+2*ncc] =
            rnlp.mpcc.meta.ind_cc2[i], rnlp.mpcc.meta.ind_cc2[i]
    end
    return rows, cols
end
function NLPModels.hess_coord!(
    rnlp::FischerBurmeisterRelaxation{T, VT},
    x::AbstractVector{T},
    y::AbstractVector{T},
    H::AbstractVector{T};
    obj_weight::Real=one(T),
) where {T, VT}
    @views hess_coord!(
        rnlp.mpcc,
        x,
        y[1:rnlp.mpcc.meta.ncon],
        H[1:rnlp.mpcc.meta.nnzh];
        obj_weight=obj_weight,
    )
    # TODO(@anton) deduplicate by maybe having one more buffer for a^2 + b^2
    nnzh = rnlp.mpcc.meta.nnzh
    ncc = rnlp.mpcc.meta.ncc
    ncon = rnlp.mpcc.meta.ncon
    comp_res_left!(rnlp.mpcc, x, rnlp.cc1_buf)
    comp_res_right!(rnlp.mpcc, x, rnlp.cc2_buf)
    # xy
    for i in 1:ncc
        H[i+nnzh] =
            y[i+ncon]*(
                rnlp.cc1_buf[i]*rnlp.cc2_buf[i]
            )/(rnlp.cc1_buf[i]^2 + rnlp.cc2_buf[i]^2)^(3/2)
    end
    # xx
    for i in 1:ncc
        H[i+nnzh+ncc] =
            -y[i+ncon]*(rnlp.cc2_buf[i]^2)/(rnlp.cc1_buf[i]^2 + rnlp.cc2_buf[i]^2)^(3/2)
    end
    # yy
    for i in 1:ncc
        H[i+nnzh+2*ncc] =
            -y[i+ncon]*(rnlp.cc1_buf[i]^2)/(rnlp.cc1_buf[i]^2 + rnlp.cc2_buf[i]^2)^(3/2)
    end
    return H
end

function NLPModels.hprod!(
    rnlp::FischerBurmeisterRelaxation{T, VT},
    x::AbstractVector{T},
    y::AbstractVector{T},
    v::AbstractVector{T},
    Hv::AbstractVector;
    obj_weight::Real=one(T),
) where {T, VT}

    # TODO(@anton) like jprod this is inefficient memory allocation wise
    #              in principle we can do the same efficiency as with the Scholtes relaxation
    #              but it is slightly more complicated.
    #              As MadMPEC doesn't use hprod we wait for user need to implement this.
    Hv .= hess(rnlp, x, obj_weight=obj_weight) * v
    return Hv
end
