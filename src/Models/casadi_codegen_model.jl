######################### Scholtes Relaxation #########################
struct CasadiCodegenModel{T, VT} <: AbstractMPCCRelaxation{T, VT}
    lib::Any # Library
    f_sym::Symbol
    nabla_f_sym::Symbol
    g_sym::Symbol
    nabla_g_sym::Symbol
    meta::NLPModels.NLPModelMeta{T, VT}
end

function CasadiCodegenModel(lib::String) where {T, VT}
   
    return CasadiCodegenModel(mpcc, meta, σ, σopt, δ1, δ1opt, δ2, δ2opt)
end

# Counters should be forwarded
function Base.getproperty(nlp::CasadiCodegenModel, sym::Symbol)
    if sym ∈ [:counters]
        getproperty(nlp.mpcc.nlp, sym)
    else
        getfield(nlp, sym)
    end
end

######################### NLPModels Callbacks #########################
NLPModels.obj(nlp::CasadiCodegenModel, x::AbstractVector) = NLPModels.obj(nlp.mpcc, x)

function NLPModels.grad!(nlp::CasadiCodegenModel, x::AbstractVector, gx::AbstractVector)
    return NLPModels.grad!(nlp.mpcc, x, gx)
end

function NLPModels.objgrad!(nlp::CasadiCodegenModel, x::AbstractVector, g::AbstractVector)
    return NLPModels.objgrad!(nlp.mpcc, x, g)
end

function NLPModels.cons!(nlp::CasadiCodegenModel, x::AbstractVector, cx::AbstractVector)
    mpcc_ncon = nlp.mpcc.meta.ncon
    if get_ncon(nlp.mpcc.nlp) > 0
        cons!(nlp.mpcc, x, view(cx, 1:mpcc_ncon))
    end
    cx[(mpcc_ncon+1):(nlp.meta.ncon)] =
        (comp_left(nlp.mpcc, x) .- lcomp_left(nlp.mpcc)) .*
        (comp_right(nlp.mpcc, x) .- lcomp_right(nlp.mpcc)) .- nlp.σ
    return cx
end

function NLPModels.cons_lin!(
    nlp::CasadiCodegenModel,
    x::AbstractVector,
    cx::AbstractVector,
)
    if get_ncon(nlp.mpcc.nlp) > 0
        return cons_lin!(nlp.mpcc, x, cx)
    else
        return cx
    end
end

function NLPModels.cons_nln!(
    nlp::CasadiCodegenModel,
    x::AbstractVector,
    cx::AbstractVector,
)
    mpcc_nnln = nlp.mpcc.meta.nnln
    # This if statement is necessary as it seems that without it c!(cx,x) does not exist in a possible underlying ADNLPModel
    if get_ncon(nlp.mpcc.nlp) > 0
        cons_nln!(nlp.mpcc, x, view(cx, 1:mpcc_nnln))
    end
    # TODO(@anton) figure out if the intermediate outputs cause allocations
    cx[(mpcc_nnln+1):(nlp.meta.nnln)] .=
        (comp_left(nlp.mpcc, x) .- lcomp_left(nlp.mpcc)) .*
        (comp_right(nlp.mpcc, x) .- lcomp_right(nlp.mpcc)) .- nlp.σ
    return cx
end

function NLPModels.jac_structure!(
    nlp::CasadiCodegenModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    @views jac_structure!(
        nlp.mpcc,
        rows[1:nlp.mpcc.meta.nnzj],
        cols[1:nlp.mpcc.meta.nnzj],
    ) # get including complementarities

    for i in 1:nlp.mpcc.meta.ncc
        rows[i+nlp.mpcc.meta.nnzj] = i + nlp.mpcc.meta.ncon
        cols[i+nlp.mpcc.meta.nnzj] = nlp.mpcc.meta.ind_cc1[i]
    end
    for i in 1:nlp.mpcc.meta.ncc
        rows[i+nlp.mpcc.meta.nnzj+nlp.mpcc.meta.ncc] = i + nlp.mpcc.meta.ncon
        cols[i+nlp.mpcc.meta.nnzj+nlp.mpcc.meta.ncc] = nlp.mpcc.meta.ind_cc2[i]
    end

    return rows, cols
end

function NLPModels.jac_lin_structure!(
    nlp::CasadiCodegenModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    jac_lin_structure!(nlp.mpcc, rows, cols) # get including complementarities
    return rows, cols
end

function NLPModels.jac_nln_structure!(
    nlp::CasadiCodegenModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    @views jac_nln_structure!(
        nlp.mpcc,
        rows[1:nlp.mpcc.meta.nln_nnzj],
        cols[1:nlp.mpcc.meta.nln_nnzj],
    ) # get including complementarities

    for i in 1:nlp.mpcc.meta.ncc
        rows[i+nlp.mpcc.meta.nln_nnzj] = i + nlp.mpcc.meta.nnln
        cols[i+nlp.mpcc.meta.nln_nnzj] = nlp.mpcc.meta.ind_cc1[i]
    end
    for i in 1:nlp.mpcc.meta.ncc
        rows[i+nlp.mpcc.meta.nln_nnzj+nlp.mpcc.meta.ncc] = i + nlp.mpcc.meta.nnln
        cols[i+nlp.mpcc.meta.nln_nnzj+nlp.mpcc.meta.ncc] = nlp.mpcc.meta.ind_cc2[i]
    end

    return rows, cols
end

function NLPModels.jac_coord!(
    nlp::CasadiCodegenModel,
    x::AbstractVector,
    j::AbstractVector,
)
    jac_coord!(nlp.mpcc, x, @view(j[1:nlp.mpcc.meta.nnzj]))

    comp_res_right!(
        nlp.mpcc,
        x,
        @view(j[(nlp.mpcc.meta.nnzj+1):(nlp.mpcc.meta.nnzj+nlp.mpcc.meta.ncc)])
    )
    comp_res_left!(
        nlp.mpcc,
        x,
        @view(
            j[(nlp.mpcc.meta.nnzj+nlp.mpcc.meta.ncc+1):(nlp.mpcc.meta.nnzj+2*nlp.mpcc.meta.ncc)]
        )
    )
    return j
end

function NLPModels.jac_lin_coord!(
    nlp::CasadiCodegenModel,
    x::AbstractVector,
    jac::AbstractVector,
)
    return jac_lin_coord!(nlp.mpcc, x, jac)
end

function NLPModels.jac_nln_coord!(
    nlp::CasadiCodegenModel,
    x::AbstractVector,
    jac::AbstractVector,
)
    jac_nln_coord!(nlp.mpcc, x, @view(jac[1:nlp.mpcc.meta.nln_nnzj]))

    comp_res_right!(
        nlp.mpcc,
        x,
        @view(
            jac[(nlp.mpcc.meta.nln_nnzj+1):(nlp.mpcc.meta.nln_nnzj+nlp.mpcc.meta.ncc)]
        )
    )
    comp_res_left!(
        nlp.mpcc,
        x,
        @view(
            jac[(nlp.mpcc.meta.nln_nnzj+nlp.mpcc.meta.ncc+1):(nlp.mpcc.meta.nln_nnzj+2*nlp.mpcc.meta.ncc)]
        )
    )
    return jac
end

function NLPModels.jprod_lin!(
    nlp::CasadiCodegenModel,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jv[1:nlp.meta.nlin] = jac_lin(nlp, x) * v
    return Jv
end

function NLPModels.jprod_nln!(
    nlp::CasadiCodegenModel,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jv[1:nlp.meta.nnln] = jac_nln(nlp, x) * v
    return Jv
end

function NLPModels.jtprod_lin!(
    nlp::CasadiCodegenModel,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jtv[1:nlp.meta.nvar] = jac_lin(nlp, x)' * v
    return Jtv
end

function NLPModels.jtprod_nln!(
    nlp::CasadiCodegenModel,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    # TODO(@anton) do this in a smarter way
    Jtv[1:nlp.meta.nvar] = jac_nln(nlp, x)' * v
    return Jtv
end

function NLPModels.hess_structure!(
    nlp::CasadiCodegenModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    @views hess_structure!(
        nlp.mpcc,
        rows[1:nlp.mpcc.meta.nnzh],
        cols[1:nlp.mpcc.meta.nnzh],
    )
    # TODO(@anton) it seems hard to vectorize in one operation this because there is no efficient unzip in Base:
    #              See https://github.com/JuliaLang/julia/issues/13942 for details
    for i in 1:nlp.mpcc.meta.ncc
        cols[i+nlp.mpcc.meta.nnzh], rows[i+nlp.mpcc.meta.nnzh] =
            minmax(nlp.mpcc.meta.ind_cc1[i], nlp.mpcc.meta.ind_cc2[i])
    end
    return rows, cols
end
function NLPModels.hess_coord!(
    nlp::CasadiCodegenModel{T, VT},
    x::AbstractVector{T},
    y::AbstractVector{T},
    H::AbstractVector{T};
    obj_weight::Real=one(T),
) where {T, VT}
    @views hess_coord!(
        nlp.mpcc,
        x,
        y[1:nlp.mpcc.meta.ncon],
        H[1:nlp.mpcc.meta.nnzh];
        obj_weight=obj_weight,
    )
    for i in 1:nlp.mpcc.meta.ncc
        H[i+nlp.mpcc.meta.nnzh] = y[i+nlp.mpcc.meta.ncon]
    end
    return H
end

function NLPModels.hprod!(
    nlp::CasadiCodegenModel{T, VT},
    x::AbstractVector{T},
    y::AbstractVector{T},
    v::AbstractVector{T},
    Hv::AbstractVector;
    obj_weight::Real=one(T),
) where {T, VT}
    @views hprod!(nlp.mpcc, x, y[1:nlp.mpcc.meta.ncon], v, Hv; obj_weight=obj_weight)
    for i in 1:nlp.mpcc.meta.ncc
        Hv[nlp.mpcc.meta.ind_cc1[i]] +=
            v[nlp.mpcc.meta.ind_cc2[i]]*y[i+nlp.mpcc.meta.ncon]
        Hv[nlp.mpcc.meta.ind_cc2[i]] +=
            v[nlp.mpcc.meta.ind_cc1[i]]*y[i+nlp.mpcc.meta.ncon]
    end
    return Hv
end

function get_relaxation(nlp::CasadiCodegenModel)
    return nlp.σ
end

function set_relaxation(nlp::CasadiCodegenModel{T}, σ::T) where {T}
    nlp.σ .= σ
    return nothing
end

function initialize_relaxation(nlp::CasadiCodegenModel{T}, σ::T, δ::T) where {T}
    nlp.σ .= σ
    nlp.σopt .= σ
    nlp.δ1 .= δ
    nlp.δ1opt .= δ
    nlp.δ2 .= δ
    nlp.δ2opt .= δ
    return nothing
end

function get_log_relaxation(nlp::CasadiCodegenModel{T}) where {T}
    return log(
        10,
        mapreduce(
            (a, b, c) -> min(a, b == 0.0 ? Inf : b, c == 0.0 ? Inf : c),
            max,
            nlp.σ,
            nlp.δ1,
            nlp.δ2,
        ),
    )
end
