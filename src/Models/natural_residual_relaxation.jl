######################### NaturalResidual Relaxation #########################
struct NaturalResidualRelaxation{T, VT} <: AbstractMPCCRelaxation{T, VT}
    mpcc::AbstractMPCCModel{T, VT}
    meta::NLPModels.NLPModelMeta{T, VT}
    cc1_buf::VT
    cc2_buf::VT
    σ::Base.RefValue{T}
end

function NaturalResidualRelaxation(mpcc::AbstractMPCCModel{T, VT}) where {T, VT}
    if !is_vertical(mpcc)
        # TODO(@anton) Perhaps we should do this automatically in the future or we can support non-vertical form scholtes
        #              though this makes the callbacks a bit more complicated
        error(
            "Fischer-Burmeister Relaxation currently expects a vertical form MPCC, use vertical_form(mpcc) to convert it.",
        )
    end

    # Update only what needs to be updated
    ncon = get_ncon(mpcc) + get_ncc(mpcc)
    lcon = vcat(get_lcon(mpcc), -typemax(T)*ones(T, get_ncc(mpcc)))
    ucon = vcat(get_ucon(mpcc), zeros(T, get_ncc(mpcc)))
    y0 = vcat(get_y0(mpcc), zeros(T, get_ncc(mpcc)))
    # TODO(@anton) this is a lower bound only accurate for vertical form: we can calculate this exactly if we store
    #              the nnz for the jacobians of G, and H, which we now do.
    nnzj = get_nnzj(mpcc) + 2*get_ncc(mpcc)
    nln_nnzj = get_nln_nnzj(mpcc) + 2*get_ncc(mpcc) # All the nln values are

    # TODO(@anton) this is a bug actually. we need to check the structure of the mpcc (and the underlying nlp) to
    #              figure out if the nnzh is correct as if the off diagonals are not already in the nonzeros.
    #
    # TODO(@anton) This may or may not break the assumptions made by show(::NLPModelMeta)
    nnzh = get_nnzh(mpcc) + 3*get_ncc(mpcc)
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
        grad_available=true,
        jac_available=true,
        hess_available=true,
        jprod_available=true,
        jtprod_available=true,
        hprod_available=true,
    )

    cc1_buf = VT(undef, get_ncc(mpcc))
    cc2_buf = VT(undef, get_ncc(mpcc))

    σ = zero(T)
    return NaturalResidualRelaxation(mpcc, meta, cc1_buf, cc2_buf, Ref(σ))
end

# Counters should be forwarded
function Base.getproperty(rnlp::NaturalResidualRelaxation, sym::Symbol)
    if sym ∈ [:counters]
        getproperty(rnlp.mpcc.nlp, sym)
    else
        getfield(rnlp, sym)
    end
end

######################### NLPModels Callbacks #########################
function NLPModels.obj(rnlp::NaturalResidualRelaxation, x::AbstractVector)
    return NLPModels.obj(rnlp.mpcc, x)
end

function NLPModels.grad!(
    rnlp::NaturalResidualRelaxation,
    x::AbstractVector,
    gx::AbstractVector,
)
    return NLPModels.grad!(rnlp.mpcc, x, gx)
end

function NLPModels.objgrad!(
    rnlp::NaturalResidualRelaxation,
    x::AbstractVector,
    g::AbstractVector,
)
    return NLPModels.objgrad!(rnlp.mpcc, x, g)
end

function NLPModels.cons!(
    rnlp::NaturalResidualRelaxation,
    x::AbstractVector,
    cx::AbstractVector,
)
    mpcc_ncon = get_ncon(rnlp.mpcc)
    if get_ncon(rnlp.mpcc.nlp) > 0
        cons!(rnlp.mpcc, x, view(cx, 1:mpcc_ncon))
    end
    comp_res_left!(rnlp.mpcc, x, rnlp.cc1_buf)
    comp_res_right!(rnlp.mpcc, x, rnlp.cc2_buf)
    @views(
        map!(
            (a, b) -> a + b - sqrt((a-b)^2 + 4*rnlp.σ[]),
            cx[(mpcc_ncon+1):(rnlp.meta.ncon)],
            rnlp.cc1_buf,
            rnlp.cc2_buf,
        )
    )
    return cx
end

function NLPModels.cons_lin!(
    rnlp::NaturalResidualRelaxation,
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
    rnlp::NaturalResidualRelaxation,
    x::AbstractVector,
    cx::AbstractVector,
)
    mpcc_nnln = get_nnln(rnlp.mpcc)
    # This if statement is necessary as it seems that without it c!(cx,x) does not exist in a possible underlying ADNLPModel
    if get_ncon(rnlp.mpcc.nlp) > 0
        cons_nln!(rnlp.mpcc, x, view(cx, 1:mpcc_nnln))
    end
    # TODO(@anton) figure out if the intermediate outputs cause allocations
    comp_res_left!(rnlp.mpcc, x, rnlp.cc1_buf)
    comp_res_right!(rnlp.mpcc, x, rnlp.cc2_buf)
    @views(
        map!(
            (a, b) -> a + b - sqrt((a-b)^2 + 4*rnlp.σ[]),
            cx[(mpcc_nnln+1):(rnlp.meta.nnln)],
            rnlp.cc1_buf,
            rnlp.cc2_buf,
        )
    )
    return cx
end

function NLPModels.jac_structure!(
    rnlp::NaturalResidualRelaxation,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    @views jac_structure!(
        rnlp.mpcc,
        rows[1:get_nnzj(rnlp.mpcc)],
        cols[1:get_nnzj(rnlp.mpcc)],
    ) # get including complementarities

    for i in 1:get_ncc(rnlp.mpcc)
        rows[i+get_nnzj(rnlp.mpcc)] = i + get_ncon(rnlp.mpcc)
        cols[i+get_nnzj(rnlp.mpcc)] = get_ind_cc1(rnlp.mpcc)[i]
    end
    for i in 1:get_ncc(rnlp.mpcc)
        rows[i+get_nnzj(rnlp.mpcc)+get_ncc(rnlp.mpcc)] = i + get_ncon(rnlp.mpcc)
        cols[i+get_nnzj(rnlp.mpcc)+get_ncc(rnlp.mpcc)] = get_ind_cc2(rnlp.mpcc)[i]
    end

    return rows, cols
end

function NLPModels.jac_lin_structure!(
    rnlp::NaturalResidualRelaxation,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    jac_lin_structure!(rnlp.mpcc, rows, cols) # get including complementarities
    return rows, cols
end

function NLPModels.jac_nln_structure!(
    rnlp::NaturalResidualRelaxation,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    @views jac_nln_structure!(
        rnlp.mpcc,
        rows[1:get_nln_nnzj(rnlp.mpcc)],
        cols[1:get_nln_nnzj(rnlp.mpcc)],
    ) # get including complementarities

    for i in 1:get_ncc(rnlp.mpcc)
        rows[i+get_nln_nnzj(rnlp.mpcc)] = i + get_nnln(rnlp.mpcc)
        cols[i+get_nln_nnzj(rnlp.mpcc)] = get_ind_cc1(rnlp.mpcc)[i]
    end
    for i in 1:get_ncc(rnlp.mpcc)
        rows[i+get_nln_nnzj(rnlp.mpcc)+get_ncc(rnlp.mpcc)] = i + get_nnln(rnlp.mpcc)
        cols[i+get_nln_nnzj(rnlp.mpcc)+get_ncc(rnlp.mpcc)] = get_ind_cc2(rnlp.mpcc)[i]
    end

    return rows, cols
end

function NLPModels.jac_coord!(
    rnlp::NaturalResidualRelaxation,
    x::AbstractVector,
    j::AbstractVector,
)
    # TODO(@anton) might be useful to special case the division operation at exactly 0
    #              or in some region around 0.
    @views begin
        jac_coord!(rnlp.mpcc, x, j[1:get_nnzj(rnlp.mpcc)])
        comp_res_left!(rnlp.mpcc, x, rnlp.cc1_buf)
        comp_res_right!(rnlp.mpcc, x, rnlp.cc2_buf)
        map!(
            (a, b) -> 1 - (a-b)/(sqrt((a-b)^2 + 4*rnlp.σ[])),
            j[(get_nnzj(rnlp.mpcc)+1):(get_nnzj(rnlp.mpcc)+get_ncc(rnlp.mpcc))],
            rnlp.cc1_buf,
            rnlp.cc2_buf,
        )
        map!(
            (a, b) -> 1 + (a-b)/(sqrt((a-b)^2 + 4*rnlp.σ[])),
            j[(get_nnzj(rnlp.mpcc)+get_ncc(
                rnlp.mpcc,
            )+1):(get_nnzj(rnlp.mpcc)+2*get_ncc(rnlp.mpcc))],
            rnlp.cc1_buf,
            rnlp.cc2_buf,
        )
    end
    return j
end

function NLPModels.jac_lin_coord!(
    rnlp::NaturalResidualRelaxation,
    x::AbstractVector,
    jac::AbstractVector,
)
    return jac_lin_coord!(rnlp.mpcc, x, jac)
end

function NLPModels.jac_nln_coord!(
    rnlp::NaturalResidualRelaxation,
    x::AbstractVector,
    jac::AbstractVector,
)
    # TODO(@anton) might be useful to special case the devision operation at exactly 0
    #              or in some region around 0.
    @views begin
        jac_coord!(rnlp.mpcc, x, j[1:get_nln_nnzj(rnlp.mpcc)])
        comp_res_left!(rnlp.mpcc, x, rnlp.cc1_buf)
        comp_res_right!(rnlp.mpcc, x, rnlp.cc2_buf)
        map!(
            (a, b) -> 1 - (a-b)/(sqrt((a-b)^2 + 4*rnlp.σ[])),
            jac[(get_nln_nnzj(rnlp.mpcc)+1):(get_nln_nnzj(rnlp.mpcc)+get_ncc(rnlp.mpcc))],
            rnlp.cc1_buf,
            rnlp.cc2_buf,
        )
        map!(
            (a, b) -> 1 + (a-b)/(sqrt((a-b)^2 + 4*rnlp.σ[])),
            jac[(get_nln_nnzj(rnlp.mpcc)+get_ncc(
                rnlp.mpcc,
            )+1):(get_nln_nnzj(rnlp.mpcc)+2*get_ncc(rnlp.mpcc))],
            rnlp.cc1_buf,
            rnlp.cc2_buf,
        )
    end
    return jac
end

function NLPModels.jprod_lin!(
    rnlp::NaturalResidualRelaxation,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jv[1:rnlp.meta.nlin] = jac_lin(rnlp, x) * v
    return Jv
end

function NLPModels.jprod_nln!(
    rnlp::NaturalResidualRelaxation,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jv[1:rnlp.meta.nnln] = jac_nln(rnlp, x) * v
    return Jv
end

function NLPModels.jtprod_lin!(
    rnlp::NaturalResidualRelaxation,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jtv[1:rnlp.meta.nvar] = jac_lin(rnlp, x)' * v
    return Jtv
end

function NLPModels.jtprod_nln!(
    rnlp::NaturalResidualRelaxation,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jtv[1:rnlp.meta.nvar] = jac_nln(rnlp, x)' * v
    return Jtv
end

function NLPModels.hess_structure!(
    rnlp::NaturalResidualRelaxation,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    @views hess_structure!(
        rnlp.mpcc,
        rows[1:get_nnzh(rnlp.mpcc)],
        cols[1:get_nnzh(rnlp.mpcc)],
    )
    # TODO(@anton) it seems hard to vectorize in one operation this because there is no efficient unzip in Base:
    #              See https://github.com/JuliaLang/julia/issues/13942 for details
    nnzh = get_nnzh(rnlp.mpcc)
    ncc = get_ncc(rnlp.mpcc)
    # Off diagonal terms
    for i in 1:ncc
        cols[i+nnzh], rows[i+nnzh] =
            minmax(get_ind_cc1(rnlp.mpcc)[i], get_ind_cc2(rnlp.mpcc)[i])
    end
    # Diagonal terms
    for i in 1:ncc
        cols[i+nnzh+ncc], rows[i+nnzh+ncc] =
            get_ind_cc1(rnlp.mpcc)[i], get_ind_cc1(rnlp.mpcc)[i]
    end
    for i in 1:ncc
        cols[i+nnzh+2*ncc], rows[i+nnzh+2*ncc] =
            get_ind_cc2(rnlp.mpcc)[i], get_ind_cc2(rnlp.mpcc)[i]
    end
    return rows, cols
end
function NLPModels.hess_coord!(
    rnlp::NaturalResidualRelaxation{T, VT},
    x::AbstractVector{T},
    y::AbstractVector{T},
    H::AbstractVector{T};
    obj_weight::Real=one(T),
) where {T, VT}
    @views hess_coord!(
        rnlp.mpcc,
        x,
        y[1:get_ncon(rnlp.mpcc)],
        H[1:get_nnzh(rnlp.mpcc)];
        obj_weight=obj_weight,
    )
    # TODO(@anton) deduplicate by maybe having one more buffer for a^2 + b^2
    nnzh = get_nnzh(rnlp.mpcc)
    ncc = get_ncc(rnlp.mpcc)
    ncon = get_ncon(rnlp.mpcc)
    comp_res_left!(rnlp.mpcc, x, rnlp.cc1_buf)
    comp_res_right!(rnlp.mpcc, x, rnlp.cc2_buf)
    # xy
    for i in 1:ncc
        H[i+nnzh] =
            y[i+ncon]*(
                4*rnlp.σ[]
            )/((rnlp.cc1_buf[i] - rnlp.cc2_buf[i])^2 + 4*rnlp.σ[])^(3/2)
    end
    # xx
    for i in 1:ncc
        H[i+nnzh+ncc] =
            -y[i+ncon]*(
                4*rnlp.σ[]
            )/((rnlp.cc1_buf[i] - rnlp.cc2_buf[i])^2 + 4*rnlp.σ[])^(3/2)
    end
    # yy
    for i in 1:ncc
        H[i+nnzh+2*ncc] =
            -y[i+ncon]*(
                4*rnlp.σ[]
            )/((rnlp.cc1_buf[i] - rnlp.cc2_buf[i])^2 + 4*rnlp.σ[])^(3/2)
    end
    return H
end

function NLPModels.hprod!(
    rnlp::NaturalResidualRelaxation{T, VT},
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
