######################### BigM Relaxation of LPCCs #########################
# TODO(@anton) enforce mpcc is an lpcc
struct BigMModel{T, VT} <: AbstractQuadraticModel{T, VT}
    lpcc::AbstractMPCCModel{T, VT}
    meta::NLPModels.NLPModelMeta{T, VT}
    counters::NLPModels.Counters{T, VT}
    Mrows::IndexSet
    Mcols::IndexSet
    Mvals::VT
end

function BigMModel(lpcc::AbstractMPCCModel{T, VT}, M::T) where {T, VT}
    if !is_vertical(lpcc)
        error(
            "BigM Relaxation currently expects a vertical form LPCC, use vertical_form(lpcc) to convert it.",
        )
    end
    nvar = lpcc.meta.nvar
    ncon = lpcc.meta.ncon
    ncc = lpcc.meta.ncc
    ind_cc1 = lpcc.meta.ind_cc1
    ind_cc2 = lpcc.meta.ind_cc2

    # Update only what needs to be updated
    ncon = ncon + 2*ncc
    lcon = vcat(lpcc.meta.lcon, .-lpcc.meta.lcon[ind_cc1], .-lpcc.meta.lcon[ind_cc2] - M)
    ucon = vcat(lpcc.meta.ucon, typemax(T)*ones(T, 2*ncc))
    y0 = vcat(mpcc.meta.y0, zeros(T, 2*ncc))

    nnzj = mpcc.meta.nnzj + 2*mpcc.meta.ncc
    lin_nnzj = mpcc.meta.nnzj + 2*mpcc.meta.ncc
    # TODO(@anton) this is a bug actually. we need to check the structure of the mpcc (and the underlying nlp) to
    #              figure out if the nnzh is correct as if the off diagonals are not already in the nonzeros.
    #
    # TODO(@anton) This may or may not break the assumptions made by show(::NLPModelMeta)
    nnzh = mpcc.meta.nnzh + mpcc.meta.ncc
    # TODO(@anton) We may need to change how nlv(b,o,c) are handled because we actually cannot
    #              backcalculate how these need to change necessarily.
    #              However these seem to not be used anywhere in the NLPModels API so I am ignoring them.

    meta = NLPModels.NLPModelMeta(
        mpcc.nlp.meta,
        nvar=nvar+ncc
        ncon=ncon+2*ncc,
        lcon=lcon,
        ucon=ucon,
        y0=y0,
        nnzj=nnzj,
        nln_nnzj=nln_nnzj,
        nnzh=nnzh,
    )
    σ = zero(T)
    return ScholtesRelaxation(mpcc, meta, Ref(σ))
end

# Counters should be forwarded
function Base.getproperty(rnlp::ScholtesRelaxation, sym::Symbol)
    if sym ∈ [:counters]
        getproperty(rnlp.mpcc.nlp, sym)
    else
        getfield(rnlp, sym)
    end
end

######################### NLPModels Callbacks #########################
NLPModels.obj(rnlp::ScholtesRelaxation, x::AbstractVector) = NLPModels.obj(rnlp.mpcc, x)

function NLPModels.grad!(rnlp::ScholtesRelaxation, x::AbstractVector, gx::AbstractVector)
    return NLPModels.grad!(rnlp.mpcc, x, gx)
end

function NLPModels.objgrad!(rnlp::ScholtesRelaxation, x::AbstractVector, g::AbstractVector)
    return NLPModels.objgrad!(rnlp.mpcc, x, g)
end

function NLPModels.cons!(rnlp::ScholtesRelaxation, x::AbstractVector, cx::AbstractVector)
    mpcc_ncon = rnlp.mpcc.meta.ncon
    if get_ncon(rnlp.mpcc.nlp) > 0
        cons!(rnlp.mpcc, x, view(cx, 1:mpcc_ncon))
    end
    cx[(mpcc_ncon+1):(rnlp.meta.ncon)] =
        (comp_left(rnlp.mpcc, x) .- lcomp_left(rnlp.mpcc)) .*
        (comp_right(rnlp.mpcc, x) .- lcomp_right(rnlp.mpcc)) .- rnlp.σ[]
    return cx
end

function NLPModels.cons_lin!(
    rnlp::ScholtesRelaxation,
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
    rnlp::ScholtesRelaxation,
    x::AbstractVector,
    cx::AbstractVector,
)
    mpcc_nnln = rnlp.mpcc.meta.nnln
    # This if statement is necessary as it seems that without it c!(cx,x) does not exist in a possible underlying ADNLPModel
    if get_ncon(rnlp.mpcc.nlp) > 0
        cons_nln!(rnlp.mpcc, x, view(cx, 1:mpcc_nnln))
    end
    # TODO(@anton) figure out if the intermediate outputs cause allocations
    cx[(mpcc_nnln+1):(rnlp.meta.nnln)] .=
        (comp_left(rnlp.mpcc, x) .- lcomp_left(rnlp.mpcc)) .*
        (comp_right(rnlp.mpcc, x) .- lcomp_right(rnlp.mpcc)) .- rnlp.σ[]
    return cx
end

function NLPModels.jac_structure!(
    rnlp::ScholtesRelaxation,
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
    rnlp::ScholtesRelaxation,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    jac_lin_structure!(rnlp.mpcc, rows, cols) # get including complementarities
    return rows, cols
end

function NLPModels.jac_nln_structure!(
    rnlp::ScholtesRelaxation,
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
    rnlp::ScholtesRelaxation,
    x::AbstractVector,
    j::AbstractVector,
)
    jac_coord!(rnlp.mpcc, x, @view(j[1:rnlp.mpcc.meta.nnzj]))

    comp_res_right!(
        rnlp.mpcc,
        x,
        @view(j[(rnlp.mpcc.meta.nnzj+1):(rnlp.mpcc.meta.nnzj+rnlp.mpcc.meta.ncc)])
    )
    comp_res_left!(
        rnlp.mpcc,
        x,
        @view(
            j[(rnlp.mpcc.meta.nnzj+rnlp.mpcc.meta.ncc+1):(rnlp.mpcc.meta.nnzj+2*rnlp.mpcc.meta.ncc)]
        )
    )
    return j
end

function NLPModels.jac_lin_coord!(
    rnlp::ScholtesRelaxation,
    x::AbstractVector,
    jac::AbstractVector,
)
    return jac_lin_coord!(rnlp.mpcc, x, jac)
end

function NLPModels.jac_nln_coord!(
    rnlp::ScholtesRelaxation,
    x::AbstractVector,
    jac::AbstractVector,
)
    jac_nln_coord!(rnlp.mpcc, x, @view(jac[1:rnlp.mpcc.meta.nln_nnzj]))

    comp_res_right!(
        rnlp.mpcc,
        x,
        @view(
            jac[(rnlp.mpcc.meta.nln_nnzj+1):(rnlp.mpcc.meta.nln_nnzj+rnlp.mpcc.meta.ncc)]
        )
    )
    comp_res_left!(
        rnlp.mpcc,
        x,
        @view(
            jac[(rnlp.mpcc.meta.nln_nnzj+rnlp.mpcc.meta.ncc+1):(rnlp.mpcc.meta.nln_nnzj+2*rnlp.mpcc.meta.ncc)]
        )
    )
    return jac
end

function NLPModels.jprod_lin!(
    rnlp::ScholtesRelaxation,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jv[1:rnlp.meta.nlin] = jac_lin(rnlp, x) * v
    return Jv
end

function NLPModels.jprod_nln!(
    rnlp::ScholtesRelaxation,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jv[1:rnlp.meta.nnln] = jac_nln(rnlp, x) * v
    return Jv
end

function NLPModels.jtprod_lin!(
    rnlp::ScholtesRelaxation,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jtv[1:rnlp.meta.nvar] = jac_lin(rnlp, x)' * v
    return Jtv
end

function NLPModels.jtprod_nln!(
    rnlp::ScholtesRelaxation,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jtv[1:rnlp.meta.nvar] = jac_nln(rnlp, x)' * v
    return Jtv
end

function NLPModels.hess_structure!(
    rnlp::ScholtesRelaxation,
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
    for i in 1:rnlp.mpcc.meta.ncc
        cols[i+rnlp.mpcc.meta.nnzh], rows[i+rnlp.mpcc.meta.nnzh] =
            minmax(rnlp.mpcc.meta.ind_cc1[i], rnlp.mpcc.meta.ind_cc2[i])
    end
    return rows, cols
end
function NLPModels.hess_coord!(
    rnlp::ScholtesRelaxation{T, VT},
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
    for i in 1:rnlp.mpcc.meta.ncc
        H[i+rnlp.mpcc.meta.nnzh] = y[i+rnlp.mpcc.meta.ncon]
    end
    return H
end

function NLPModels.hprod!(
    rnlp::ScholtesRelaxation{T, VT},
    x::AbstractVector{T},
    y::AbstractVector{T},
    v::AbstractVector{T},
    Hv::AbstractVector;
    obj_weight::Real=one(T),
) where {T, VT}
    @views hprod!(rnlp.mpcc, x, y[1:rnlp.mpcc.meta.ncon], v, Hv; obj_weight=obj_weight)
    for i in 1:rnlp.mpcc.meta.ncc
        Hv[rnlp.mpcc.meta.ind_cc1[i]] +=
            v[rnlp.mpcc.meta.ind_cc2[i]]*y[i+rnlp.mpcc.meta.ncon]
        Hv[rnlp.mpcc.meta.ind_cc2[i]] +=
            v[rnlp.mpcc.meta.ind_cc1[i]]*y[i+rnlp.mpcc.meta.ncon]
    end
    return Hv
end
