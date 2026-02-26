######################### Scholtes Relaxation #########################
struct ScholtesRelaxation{T, VT, MT <: AbstractMPCCModel{T, VT}} <:
       AbstractMPCCRelaxation{T, VT}
    mpcc::MT
    meta::NLPModels.NLPModelMeta{T, VT}
    σ::VT
    σopt::VT
    δ1::VT
    δ1opt::VT
    δ2::VT
    δ2opt::VT
    counters::NLPModels.Counters
end

function ScholtesRelaxation(mpcc::AbstractMPCCModel{T, VT}) where {T, VT}
    if !is_vertical(mpcc)
        # TODO(@anton) Perhaps we should do this automatically in the future or we can support non-vertical form scholtes
        #              though this makes the callbacks a bit more complicated
        error(
            "Scholtes Relaxation currently expects a vertical form MPCC, use vertical_form(mpcc) to convert it.",
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
    nnzh = get_nnzh(mpcc) + get_ncc(mpcc)
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
    σ = similar(get_x0(mpcc), get_ncc(mpcc))
    σ .= 0
    σopt = similar(get_x0(mpcc), get_ncc(mpcc))
    σopt .= 0
    δ1 = similar(get_x0(mpcc), get_ncc(mpcc))
    δ1 .= 0
    δ1opt = similar(get_x0(mpcc), get_ncc(mpcc))
    δ1opt .= 0
    δ2 = similar(get_x0(mpcc), get_ncc(mpcc))
    δ2 .= 0
    δ2opt = similar(get_x0(mpcc), get_ncc(mpcc))
    δ2opt .= 0
    return ScholtesRelaxation(mpcc, meta, σ, σopt, δ1, δ1opt, δ2, δ2opt, mpcc.counters)
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
    mpcc_ncon = get_ncon(rnlp.mpcc)
    if get_ncon(rnlp.mpcc.nlp) > 0
        cons!(rnlp.mpcc, x, view(cx, 1:mpcc_ncon))
    end
    cx[(mpcc_ncon+1):(get_ncon(rnlp))] =
        (comp_left(rnlp.mpcc, x) .- lcomp_left(rnlp.mpcc)) .*
        (comp_right(rnlp.mpcc, x) .- lcomp_right(rnlp.mpcc)) .- rnlp.σ
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
    mpcc_nnln = get_nnln(rnlp.mpcc)
    # This if statement is necessary as it seems that without it c!(cx,x) does not exist in a possible underlying ADNLPModel
    if get_ncon(rnlp.mpcc.nlp) > 0
        cons_nln!(rnlp.mpcc, x, view(cx, 1:mpcc_nnln))
    end
    # TODO(@anton) figure out if the intermediate outputs cause allocations
    cx[(mpcc_nnln+1):(get_nnln(rnlp))] .=
        (comp_left(rnlp.mpcc, x) .- lcomp_left(rnlp.mpcc)) .*
        (comp_right(rnlp.mpcc, x) .- lcomp_right(rnlp.mpcc)) .- rnlp.σ
    return cx
end

function NLPModels.jac_structure!(
    rnlp::ScholtesRelaxation,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    @views jac_structure!(
        rnlp.mpcc,
        rows[1:get_nnzj(rnlp.mpcc)],
        cols[1:get_nnzj(rnlp.mpcc)],
    ) # get including complementarities

    nnzj = get_nnzj(rnlp.mpcc)
    ncon = get_ncon(rnlp.mpcc)
    ncc = get_ncc(rnlp.mpcc)
    @allowscalar begin # TODO(@anton) think about your life choices
        @views begin
            copyto!(rows[(nnzj+1):(nnzj+ncc)], (ncon+1):(ncon+ncc))
            copyto!(cols[(nnzj+1):(nnzj+ncc)], get_ind_cc1(rnlp.mpcc))
            copyto!(rows[(nnzj+ncc+1):(nnzj+2*ncc)], (ncon+1):(ncon+ncc))
            copyto!(cols[(nnzj+ncc+1):(nnzj+2*ncc)], get_ind_cc2(rnlp.mpcc))
        end
    end
    # for i in 1:get_ncc(rnlp.mpcc)
    #     rows[i+get_nnzj(rnlp.mpcc)] = i + get_ncon(rnlp.mpcc)
    #     cols[i+get_nnzj(rnlp.mpcc)] = get_ind_cc1(rnlp.mpcc)[i]
    # end
    # for i in 1:get_ncc(rnlp.mpcc)
    #     rows[i+get_nnzj(rnlp.mpcc)+get_ncc(rnlp.mpcc)] = i + get_ncon(rnlp.mpcc)
    #     cols[i+get_nnzj(rnlp.mpcc)+get_ncc(rnlp.mpcc)] = get_ind_cc2(rnlp.mpcc)[i]
    # end

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
    rnlp::ScholtesRelaxation,
    x::AbstractVector,
    j::AbstractVector,
)
    jac_coord!(rnlp.mpcc, x, @view(j[1:get_nnzj(rnlp.mpcc)]))

    comp_res_right!(
        rnlp.mpcc,
        x,
        @view(j[(get_nnzj(rnlp.mpcc)+1):(get_nnzj(rnlp.mpcc)+get_ncc(rnlp.mpcc))])
    )
    comp_res_left!(
        rnlp.mpcc,
        x,
        @view(
            j[(get_nnzj(rnlp.mpcc)+get_ncc(
                rnlp.mpcc,
            )+1):(get_nnzj(rnlp.mpcc)+2*get_ncc(rnlp.mpcc))]
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
    jac_nln_coord!(rnlp.mpcc, x, @view(jac[1:get_nln_nnzj(rnlp.mpcc)]))

    comp_res_right!(
        rnlp.mpcc,
        x,
        @view(
            jac[(get_nln_nnzj(rnlp.mpcc)+1):(get_nln_nnzj(rnlp.mpcc)+get_ncc(rnlp.mpcc))]
        )
    )
    comp_res_left!(
        rnlp.mpcc,
        x,
        @view(
            jac[(get_nln_nnzj(rnlp.mpcc)+get_ncc(
                rnlp.mpcc,
            )+1):(get_nln_nnzj(rnlp.mpcc)+2*get_ncc(rnlp.mpcc))]
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
    Jv[1:get_nlin(rnlp)] = jac_lin(rnlp, x) * v
    return Jv
end

function NLPModels.jprod_nln!(
    rnlp::ScholtesRelaxation,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jv[1:get_nnln(rnlp)] = jac_nln(rnlp, x) * v
    return Jv
end

function NLPModels.jtprod_lin!(
    rnlp::ScholtesRelaxation,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jtv[1:get_nvar(rnlp)] = jac_lin(rnlp, x)' * v
    return Jtv
end

function NLPModels.jtprod_nln!(
    rnlp::ScholtesRelaxation,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jtv[1:get_nvar(rnlp)] = jac_nln(rnlp, x)' * v
    return Jtv
end

function NLPModels.hess_structure!(
    rnlp::ScholtesRelaxation,
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
    # TODO(@anton) think about your life choices again
    @allowscalar begin
        for i in 1:get_ncc(rnlp.mpcc)
            cols[i+get_nnzh(rnlp.mpcc)], rows[i+get_nnzh(rnlp.mpcc)] =
                minmax(get_ind_cc1(rnlp.mpcc)[i], get_ind_cc2(rnlp.mpcc)[i])
        end
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
        y[1:get_ncon(rnlp.mpcc)],
        H[1:get_nnzh(rnlp.mpcc)];
        obj_weight=obj_weight,
    )
    ncc = get_ncc(rnlp.mpcc)
    ncon = get_ncon(rnlp.mpcc)
    nnzh = get_nnzh(rnlp.mpcc)
    @views copyto!(H[(nnzh+1):(nnzh+ncc)], y[(ncon+1):(ncon+ncc)])
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
    @views hprod!(rnlp.mpcc, x, y[1:get_ncon(rnlp.mpcc)], v, Hv; obj_weight=obj_weight)
    for i in 1:get_ncc(rnlp.mpcc)
        Hv[get_ind_cc1(rnlp.mpcc)[i]] +=
            v[get_ind_cc2(rnlp.mpcc)[i]]*y[i+get_ncon(rnlp.mpcc)]
        Hv[get_ind_cc2(rnlp.mpcc)[i]] +=
            v[get_ind_cc1(rnlp.mpcc)[i]]*y[i+get_ncon(rnlp.mpcc)]
    end
    return Hv
end

function get_relaxation(rnlp::ScholtesRelaxation)
    return rnlp.σ
end

function set_relaxation(rnlp::ScholtesRelaxation{T}, σ::T) where {T}
    rnlp.σ .= σ
    return nothing
end

function initialize_relaxation(rnlp::ScholtesRelaxation{T}, σ::T, δ::T) where {T}
    rnlp.σ .= σ
    rnlp.σopt .= σ
    rnlp.δ1 .= δ
    rnlp.δ1opt .= δ
    rnlp.δ2 .= δ
    rnlp.δ2opt .= δ
    return nothing
end

function get_log_relaxation(rnlp::ScholtesRelaxation{T}) where {T}
    return log(
        10,
        mapreduce(
            (a, b, c) -> min(a, b == 0.0 ? Inf : b, c == 0.0 ? Inf : c),
            max,
            rnlp.σ,
            rnlp.δ1,
            rnlp.δ2,
        ),
    )
end
