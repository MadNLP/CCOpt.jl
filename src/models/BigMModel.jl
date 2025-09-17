######################### BigM Relaxation of LPCCs #########################
# TODO(@anton) implement a x-less eval of grad etc.
struct BigMModel{T, VT} <: AbstractNLPModel{T, VT}
    lpcc::AbstractMPCCModel{T, VT}
    meta::NLPModels.NLPModelMeta{T, VT}
    Mrows::IndexSet
    Mcols::IndexSet
    Mvals::VT
end

function BigMModel(lpcc::AbstractLPCCModel{T, VT}, M::T) where {T, VT}
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
    lcon = vcat(lpcc.meta.lcon, .-lpcc.meta.lvar[ind_cc1], .-lpcc.meta.lvar[ind_cc2] .- M)
    ucon = vcat(lpcc.meta.ucon, typemax(T)*ones(T, 2*ncc))
    y0 = vcat(lpcc.meta.y0, zeros(T, 2*ncc))
    x0 = vcat(lpcc.meta.x0, zeros(T, ncc))
    lvar = vcat(lpcc.meta.lvar, zeros(T, ncc))
    uvar = vcat(lpcc.meta.uvar, ones(T, ncc))

    nnzj = lpcc.meta.nnzj + 4*lpcc.meta.ncc
    lin_nnzj = lpcc.meta.nnzj + 4*lpcc.meta.ncc

    meta = NLPModels.NLPModelMeta(
        lpcc.nlp.meta;
        nvar=nvar+ncc,
        ncon=ncon+2*ncc,
        lcon=lcon,
        ucon=ucon,
        lvar=lvar,
        uvar=uvar,
        y0=y0,
        x0=x0,
        nnzj=nnzj,
        lin_nnzj=lin_nnzj,
    )
    # build bigm addition
    Mrows = IndexSet(undef, 4*ncc)
    Mcols = IndexSet(undef, 4*ncc)
    Mvals = VT(undef, 4*ncc)

    for ii in 1:ncc
        # Ms-x_1 > -lbx_2
        Mrows[4*ii-3] = ncon + ii
        Mrows[4*ii-2] = ncon + ii
        Mcols[4*ii-3] = lpcc.meta.ind_cc1[ii]
        Mcols[4*ii-2] = nvar + ii
        Mvals[4*ii-3] = -1.0
        Mvals[4*ii-2] = M
        # -Ms-x_2 > -M - lbx_2
        Mrows[4*ii-1] = ncon + ncc + ii
        Mrows[4*ii] = ncon + ncc + ii
        Mcols[4*ii-1] = lpcc.meta.ind_cc2[ii]
        Mcols[4*ii] = nvar + ii
        Mvals[4*ii-1] = -1.0
        Mvals[4*ii] = -M
    end

    return BigMModel(lpcc, meta, Mrows, Mcols, Mvals)
end

# Counters should be forwarded
function Base.getproperty(bigm::BigMModel, sym::Symbol)
    if sym ∈ [:counters]
        getproperty(bigm.lpcc.nlp, sym)
    else
        getfield(bigm, sym)
    end
end

######################### NLPModels Callbacks #########################
function NLPModels.obj(bigm::BigMModel, x::AbstractVector)
    @views(NLPModels.obj(bigm.lpcc, x[1:bigm.lpcc.meta.nvar]))
end

function NLPModels.grad!(bigm::BigMModel, x::AbstractVector, gx::AbstractVector)
    @views(NLPModels.grad!(bigm.lpcc, x[1:bigm.lpcc.meta.nvar], gx[1:bigm.lpcc.meta.nvar]))
    gx[(bigm.lpcc.meta.nvar+1):bigm.meta.nvar] .= 0
    return gx
end

function NLPModels.objgrad!(bigm::BigMModel, x::AbstractVector, g::AbstractVector)
    (obj, g) = @views(
        NLPModels.objgrad!(bigm.lpcc, x[1:bigm.lpcc.meta.nvar], g[1:bigm.lpcc.meta.nvar])
    )
    g[(bigm.lpcc.meta.nvar+1):bigm.meta.nvar] .= 0
    return obj, g
end

function NLPModels.cons!(bigm::BigMModel, x::AbstractVector, cx::AbstractVector)
    lpcc_ncon = bigm.lpcc.meta.ncon
    if get_ncon(bigm.lpcc.nlp) > 0
        @views(cons!(bigm.lpcc, x[1:bigm.lpcc.meta.nvar], cx[1:lpcc_ncon]))
    end
    @views begin
        cx[(lpcc_ncon+1):(lpcc_ncon+bigm.lpcc.meta.ncc)] .=
            M .- comp_res_left(bigm.lpcc, x[1:bigm.lpcc.meta.nvar])
        cx[(lpcc_ncon+bigm.lpcc.meta.ncc+1):(lpcc_ncon+2*bigm.lpcc.meta.ncc)] .=
            -M .- comp_res_left(bigm.lpcc, x[1:bigm.lpcc.meta.nvar])
    end
    return cx
end

function NLPModels.cons_lin!(bigm::BigMModel, x::AbstractVector, cx::AbstractVector)
    @views begin
        if get_ncon(bigm.lpcc.nlp) > 0
            return cons_lin!(bigm.lpcc, x[1:bigm.lpcc.meta.nvar], cx)
        end
        cx[(lpcc_ncon+1):(lpcc_ncon+bigm.lpcc.meta.ncc)] .=
            M .- comp_res_left(bigm.lpcc, x[1:bigm.lpcc.meta.nvar])
        cx[(lpcc_ncon+bigm.lpcc.meta.ncc+1):(lpcc_ncon+2*bigm.lpcc.meta.ncc)] .=
            -M .- comp_res_left(bigm.lpcc, x[1:bigm.lpcc.meta.nvar])
    end
    return cx
end

function NLPModels.jac_structure!(
    bigm::BigMModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    @views jac_structure!(
        bigm.lpcc,
        rows[1:bigm.lpcc.meta.nnzj],
        cols[1:bigm.lpcc.meta.nnzj],
    ) # get including complementarities

    rows[(bigm.lpcc.meta.nnzj+1):end] .= bigm.Mrows
    cols[(bigm.lpcc.meta.nnzj+1):end] .= bigm.Mcols

    return rows, cols
end

function NLPModels.jac_lin_structure!(
    bigm::BigMModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    @views jac_lin_structure!(
        bigm.lpcc,
        rows[1:bigm.lpcc.meta.lin_nnzj],
        cols[1:bigm.lpcc.meta.lin_nnzj],
    ) # get including complementarities

    rows[(bigm.lpcc.meta.lin_nnzj+1):end] .= bigm.Mrows
    cols[(bigm.lpcc.meta.lin_nnzj+1):end] .= bigm.Mcols

    return rows, cols
end

function NLPModels.jac_coord!(bigm::BigMModel, x::AbstractVector, j::AbstractVector)
    @views(jac_coord!(bigm.lpcc, x[1:bigm.lpcc.meta.nvar], j[1:bigm.lpcc.meta.nnzj]))

    j[(bigm.lpcc.meta.nnzj+1):end] .= bigm.Mvals
    return j
end

function NLPModels.jac_lin_coord!(bigm::BigMModel, x::AbstractVector, j::AbstractVector)
    @views(
        jac_lin_coord!(bigm.lpcc, x[1:bigm.lpcc.meta.nvar], j[1:bigm.lpcc.meta.lin_nnzj])
    )

    j[(bigm.lpcc.meta.lin_nnzj+1):end] .= bigm.Mvals

    return j
end
