######################### BigM Relaxation of LPCCs #########################
# TODO(@anton) implement a x-less eval of grad etc.
struct BigMModel{T,VT,MT<:LPCCModel} <: AbstractNLPModel{T,VT}
    lpcc::MT
    meta::NLPModels.NLPModelMeta{T,VT}
    Mrows::IndexSet
    Mcols::IndexSet
    Mvals::VT
end

function BigMModel(lpcc::LPCCModel{T,VT}, M::T) where {T,VT}
    if !is_vertical(lpcc)
        error(
            "BigM Relaxation currently expects a vertical form LPCC, use vertical_form(lpcc) to convert it.",
        )
    end
    nvar = get_nvar(lpcc)
    ncon = get_ncon(lpcc)
    ncc = get_ncc(lpcc)
    ind_cc1 = get_ind_cc1(lpcc)
    ind_cc2 = get_ind_cc2(lpcc)

    # Update only what needs to be updated
    lcon = vcat(get_lcon(lpcc), .-get_lvar(lpcc)[ind_cc1], .-get_lvar(lpcc)[ind_cc2] .- M)
    ucon = vcat(get_ucon(lpcc), typemax(T)*ones(T, 2*ncc))
    y0 = vcat(get_y0(lpcc), zeros(T, 2*ncc))
    x0 = vcat(get_x0(lpcc), zeros(T, ncc))
    lvar = vcat(get_lvar(lpcc), zeros(T, ncc))
    uvar = vcat(get_uvar(lpcc), ones(T, ncc))

    nnzj = get_nnzj(lpcc) + 4*get_ncc(lpcc)
    lin_nnzj = get_nnzj(lpcc) + 4*get_ncc(lpcc)

    meta = NLPModels.NLPModelMeta(
        lpcc.nlp.meta;
        nvar = nvar+ncc,
        ncon = ncon+2*ncc,
        lcon = lcon,
        ucon = ucon,
        lvar = lvar,
        uvar = uvar,
        y0 = y0,
        x0 = x0,
        nnzj = nnzj,
        lin_nnzj = lin_nnzj,
    )
    # build bigm addition
    Mrows = IndexSet(undef, 4*ncc)
    Mcols = IndexSet(undef, 4*ncc)
    Mvals = VT(undef, 4*ncc)

    for ii in 1:ncc
        # Ms-x_1 > -lbx_2
        Mrows[4*ii-3] = ncon + ii
        Mrows[4*ii-2] = ncon + ii
        Mcols[4*ii-3] = get_ind_cc1(lpcc)[ii]
        Mcols[4*ii-2] = nvar + ii
        Mvals[4*ii-3] = -1.0
        Mvals[4*ii-2] = M
        # -Ms-x_2 > -M - lbx_2
        Mrows[4*ii-1] = ncon + ncc + ii
        Mrows[4*ii] = ncon + ncc + ii
        Mcols[4*ii-1] = get_ind_cc2(lpcc)[ii]
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
    @views(NLPModels.obj(bigm.lpcc, x[1:get_nvar(bigm.lpcc)]))
end

function NLPModels.grad!(bigm::BigMModel, x::AbstractVector, gx::AbstractVector)
    @views(NLPModels.grad!(bigm.lpcc, x[1:get_nvar(bigm.lpcc)], gx[1:get_nvar(bigm.lpcc)]))
    gx[(get_nvar(bigm.lpcc)+1):bigm.meta.nvar] .= 0
    return gx
end

function NLPModels.objgrad!(bigm::BigMModel, x::AbstractVector, g::AbstractVector)
    (obj, g) = @views(
        NLPModels.objgrad!(bigm.lpcc, x[1:get_nvar(bigm.lpcc)], g[1:get_nvar(bigm.lpcc)])
    )
    g[(get_nvar(bigm.lpcc)+1):bigm.meta.nvar] .= 0
    return obj, g
end

function NLPModels.cons!(bigm::BigMModel, x::AbstractVector, cx::AbstractVector)
    lpcc_ncon = get_ncon(bigm.lpcc)
    if get_ncon(bigm.lpcc.nlp) > 0
        @views(cons!(bigm.lpcc, x[1:get_nvar(bigm.lpcc)], cx[1:lpcc_ncon]))
    end
    @views begin
        cx[(lpcc_ncon+1):(lpcc_ncon+get_ncc(bigm.lpcc))] .=
            M .- comp_res_left(bigm.lpcc, x[1:get_nvar(bigm.lpcc)])
        cx[(lpcc_ncon+get_ncc(bigm.lpcc)+1):(lpcc_ncon+2*get_ncc(bigm.lpcc))] .=
            -M .- comp_res_left(bigm.lpcc, x[1:get_nvar(bigm.lpcc)])
    end
    return cx
end

function NLPModels.cons_lin!(bigm::BigMModel, x::AbstractVector, cx::AbstractVector)
    @views begin
        if get_ncon(bigm.lpcc.nlp) > 0
            return cons_lin!(bigm.lpcc, x[1:get_nvar(bigm.lpcc)], cx)
        end
        cx[(lpcc_ncon+1):(lpcc_ncon+get_ncc(bigm.lpcc))] .=
            M .- comp_res_left(bigm.lpcc, x[1:get_nvar(bigm.lpcc)])
        cx[(lpcc_ncon+get_ncc(bigm.lpcc)+1):(lpcc_ncon+2*get_ncc(bigm.lpcc))] .=
            -M .- comp_res_left(bigm.lpcc, x[1:get_nvar(bigm.lpcc)])
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
        rows[1:get_nnzj(bigm.lpcc)],
        cols[1:get_nnzj(bigm.lpcc)],
    ) # get including complementarities

    rows[(get_nnzj(bigm.lpcc)+1):end] .= bigm.Mrows
    cols[(get_nnzj(bigm.lpcc)+1):end] .= bigm.Mcols

    return rows, cols
end

function NLPModels.jac_lin_structure!(
    bigm::BigMModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    @views jac_lin_structure!(
        bigm.lpcc,
        rows[1:get_lin_nnzj(bigm.lpcc)],
        cols[1:get_lin_nnzj(bigm.lpcc)],
    ) # get including complementarities

    rows[(get_lin_nnzj(bigm.lpcc)+1):end] .= bigm.Mrows
    cols[(get_lin_nnzj(bigm.lpcc)+1):end] .= bigm.Mcols

    return rows, cols
end

function NLPModels.jac_coord!(bigm::BigMModel, x::AbstractVector, j::AbstractVector)
    @views(jac_coord!(bigm.lpcc, x[1:get_nvar(bigm.lpcc)], j[1:get_nnzj(bigm.lpcc)]))

    j[(get_nnzj(bigm.lpcc)+1):end] .= bigm.Mvals
    return j
end

function NLPModels.jac_lin_coord!(bigm::BigMModel, x::AbstractVector, j::AbstractVector)
    @views(
        jac_lin_coord!(bigm.lpcc, x[1:get_nvar(bigm.lpcc)], j[1:get_lin_nnzj(bigm.lpcc)])
    )

    j[(get_lin_nnzj(bigm.lpcc)+1):end] .= bigm.Mvals

    return j
end
