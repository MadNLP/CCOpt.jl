# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Source: https://github.com/JuliaSmoothOptimizers/NLPModelsModifiers.jl/blob/40655dd6fc140da39da4cf630062d7b0ebf54c05/test/nlp/slack-model.jl
"""
    SimpleNLPModel <: AbstractNLPModel

Simple model for testing purposes.
Modified problem 14 in the Hock-Schittkowski suite

     min   (x₁ - 2)² + (x₂ - 1)²
    s.to   x₁ - 2x₂ + 1 = 0
           -x₁² / 4 - x₂² + 1 ≥ 0
           0 ≤ x ≤ 1

x₀ = [2.0, 2.0].
"""
mutable struct SimpleNLPModel{T, S, M <: AbstractNLPModelMeta{T, S}} <:
               AbstractNLPModel{T, S}
    meta::M
    counters::Counters
end

mutable struct SimpleNLPMeta{T, S} <: AbstractNLPModelMeta{T, S}
    nvar::Int
    x0::S
    lvar::S
    uvar::S

    ifix::Vector{Int}
    ilow::Vector{Int}
    iupp::Vector{Int}
    irng::Vector{Int}
    ifree::Vector{Int}
    iinf::Vector{Int}

    nlvb::Int
    nlvo::Int
    nlvc::Int

    ncon::Int
    y0::S
    lcon::S
    ucon::S

    jfix::Vector{Int}
    jlow::Vector{Int}
    jupp::Vector{Int}
    jrng::Vector{Int}
    jfree::Vector{Int}
    jinf::Vector{Int}

    nnzo::Int
    nnzj::Int
    lin_nnzj::Int
    nln_nnzj::Int
    nnzh::Int

    nlin::Int
    nnln::Int

    lin::Vector{Int}
    nln::Vector{Int}

    minimize::Bool
    islp::Bool
    name::String
    constraint_bounds_analysis::Bool
    variable_bounds_analysis::Bool
    sparse_jacobian::Bool
    sparse_hessian::Bool
    grad_available::Bool
    jac_available::Bool
    hess_available::Bool
    jprod_available::Bool
    jtprod_available::Bool
    hprod_available::Bool
    function SimpleNLPMeta{T, S}(
        nvar::Int;
        x0::S=fill!(S(undef, nvar), zero(T)),
        lvar::S=fill!(S(undef, nvar), T(-Inf)),
        uvar::S=fill!(S(undef, nvar), T(Inf)),
        nlvb=nvar,
        nlvo=nvar,
        nlvc=nvar,
        ncon=2,
        y0::S=fill!(S(undef, ncon), zero(T)),
        lcon::S=fill!(S(undef, ncon), T(-Inf)),
        ucon::S=fill!(S(undef, ncon), T(Inf)),
        nnzo=nvar,
        nnzj=nvar * ncon,
        lin_nnzj=0,
        nln_nnzj=nvar * ncon,
        nnzh=nvar * (nvar + 1) / 2,
        lin=Int[],
        minimize=true,
        islp=false,
        name="Generic",
        constraint_bound_analysis=false,
        variable_bound_analysis=false,
        sparse_jacobian=true,
        sparse_hessian=true,
        grad_available::Bool=true,
        jac_available::Bool=(ncon > 0),
        hess_available::Bool=true,
        jprod_available::Bool=(ncon > 0),
        jtprod_available::Bool=(ncon > 0),
        hprod_available::Bool=true,
    ) where {T, S}
        if (nvar < 1) || (ncon < 0)
            error("Nonsensical dimensions")
        end

        @lencheck nvar x0 lvar uvar
        @lencheck ncon y0 lcon ucon
        @rangecheck 1 ncon lin

        ifix = findall(lvar .== uvar)
        ilow = findall((lvar .> T(-Inf)) .& (uvar .== T(Inf)))
        iupp = findall((lvar .== T(-Inf)) .& (uvar .< T(Inf)))
        irng = findall((lvar .> T(-Inf)) .& (uvar .< T(Inf)) .& (lvar .< uvar))
        ifree = findall((lvar .== T(-Inf)) .& (uvar .== T(Inf)))
        iinf = findall(lvar .> uvar)

        jfix = findall(lcon .== ucon)
        jlow = findall((lcon .> T(-Inf)) .& (ucon .== T(Inf)))
        jupp = findall((lcon .== T(-Inf)) .& (ucon .< T(Inf)))
        jrng = findall((lcon .> T(-Inf)) .& (ucon .< T(Inf)) .& (lcon .< ucon))
        jfree = findall((lcon .== T(-Inf)) .& (ucon .== T(Inf)))
        jinf = findall(lcon .> ucon)

        nnzj = max(0, nnzj)
        nnzh = max(0, nnzh)

        nln = setdiff(1:ncon, lin)
        nlin = length(lin)
        nnln = length(nln)

        return new{T, S}(
            nvar,
            x0,
            lvar,
            uvar,
            ifix,
            ilow,
            iupp,
            irng,
            ifree,
            iinf,
            nlvb,
            nlvo,
            nlvc,
            ncon,
            y0,
            lcon,
            ucon,
            jfix,
            jlow,
            jupp,
            jrng,
            jfree,
            jinf,
            nnzo,
            nnzj,
            lin_nnzj,
            nln_nnzj,
            nnzh,
            nlin,
            nnln,
            lin,
            nln,
            minimize,
            islp,
            name,
            constraint_bound_analysis,
            variable_bound_analysis,
            sparse_jacobian,
            sparse_hessian,
            grad_available,
            jac_available,
            hess_available,
            jprod_available,
            jtprod_available,
            hprod_available,
        )
    end
end
NLPModels.equality_constrained(meta::SimpleNLPMeta) = length(meta.jfix) == meta.ncon > 0
NLPModels.unconstrained(meta::SimpleNLPMeta) = meta.ncon == 0 && !has_bounds(meta)

function SimpleNLPModel(::Type{T}, ::Type{NLPModelMeta}) where {T}
    meta = NLPModelMeta{T, Vector{T}}(
        2,
        nnzh=2,
        ncon=2,
        lvar=zeros(T, 2),
        uvar=ones(T, 2),
        x0=T[2; 2],
        lcon=T[0; 0],
        ucon=T[0; Inf],
        name="Simple NLP Model",
        lin_nnzj=2,
        nln_nnzj=2,
        lin=[1],
    )
    return SimpleNLPModel(meta, Counters())
end
function SimpleNLPModel(::Type{T}, ::Type{SimpleNLPMeta}) where {T}
    meta = SimpleNLPMeta{T, Vector{T}}(
        2,
        nnzh=2,
        ncon=2,
        lvar=zeros(T, 2),
        uvar=ones(T, 2),
        x0=T[2; 2],
        lcon=T[0; 0],
        ucon=T[0; Inf],
        name="Simple NLP Model",
        lin_nnzj=2,
        nln_nnzj=2,
        lin=[1],
    )
    return SimpleNLPModel(meta, Counters())
end
SimpleNLPModel() = SimpleNLPModel(Float64, NLPModelMeta)

function NLPModels.obj(nlp::SimpleNLPModel, x::AbstractVector)
    @lencheck 2 x
    increment!(nlp, :neval_obj)
    return (x[1] - 2)^2 + (x[2] - 1)^2
end

function NLPModels.grad!(nlp::SimpleNLPModel, x::AbstractVector, gx::AbstractVector)
    @lencheck 2 x gx
    increment!(nlp, :neval_grad)
    gx .= [2 * (x[1] - 2); 2 * (x[2] - 1)]
    return gx
end

function NLPModels.hess_structure!(
    nlp::SimpleNLPModel,
    rows::AbstractVector{Int},
    cols::AbstractVector{Int},
)
    @lencheck 2 rows cols
    rows[1] = 1
    rows[2] = 2
    cols[1] = 1
    cols[2] = 2
    return rows, cols
end

function NLPModels.hess_coord!(
    nlp::SimpleNLPModel,
    x::AbstractVector{T},
    y::AbstractVector{T},
    vals::AbstractVector{T};
    obj_weight=one(T),
) where {T}
    @lencheck 2 x y vals
    increment!(nlp, :neval_hess)
    vals .= 2obj_weight
    vals[1] -= y[2] / 2
    vals[2] -= 2y[2]
    return vals
end

function NLPModels.hprod!(
    nlp::SimpleNLPModel,
    x::AbstractVector{T},
    y::AbstractVector{T},
    v::AbstractVector{T},
    Hv::AbstractVector{T};
    obj_weight=one(T),
) where {T}
    @lencheck 2 x y v Hv
    increment!(nlp, :neval_hprod)
    Hv .= 2obj_weight * v
    Hv[1] -= y[2] * v[1] / 2
    Hv[2] -= 2y[2] * v[2]
    return Hv
end

function NLPModels.cons_lin!(nlp::SimpleNLPModel, x::AbstractVector, cx::AbstractVector)
    @lencheck 2 x
    @lencheck 1 cx
    increment!(nlp, :neval_cons)
    cx .= [x[1] - 2 * x[2] + 1]
    return cx
end

function NLPModels.cons_nln!(nlp::SimpleNLPModel, x::AbstractVector, cx::AbstractVector)
    @lencheck 2 x
    @lencheck 1 cx
    increment!(nlp, :neval_cons)
    cx .= [-x[1]^2 / 4 - x[2]^2 + 1]
    return cx
end

function NLPModels.jac_lin_structure!(
    nlp::SimpleNLPModel,
    rows::AbstractVector{Int},
    cols::AbstractVector{Int},
)
    @lencheck 2 rows cols
    rows .= [1, 1]
    cols .= [1, 2]
    return rows, cols
end

function NLPModels.jac_nln_structure!(
    nlp::SimpleNLPModel,
    rows::AbstractVector{Int},
    cols::AbstractVector{Int},
)
    @lencheck 2 rows cols
    rows .= [1, 1]
    cols .= [1, 2]
    return rows, cols
end

function NLPModels.jac_lin_coord!(
    nlp::SimpleNLPModel,
    x::AbstractVector,
    vals::AbstractVector,
)
    @lencheck 2 x
    @lencheck 2 vals
    increment!(nlp, :neval_jac)
    vals .= [1, -2]
    return vals
end

function NLPModels.jac_nln_coord!(
    nlp::SimpleNLPModel,
    x::AbstractVector,
    vals::AbstractVector,
)
    @lencheck 2 x
    @lencheck 2 vals
    increment!(nlp, :neval_jac)
    vals .= [-x[1] / 2, -2 * x[2]]
    return vals
end

function NLPModels.jprod_lin!(
    nlp::SimpleNLPModel,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    @lencheck 2 x v
    @lencheck 1 Jv
    increment!(nlp, :neval_jprod)
    Jv .= [v[1] - 2 * v[2]]
    return Jv
end

function NLPModels.jprod_nln!(
    nlp::SimpleNLPModel,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    @lencheck 2 x v
    @lencheck 1 Jv
    increment!(nlp, :neval_jprod)
    Jv .= [-x[1] * v[1] / 2 - 2 * x[2] * v[2]]
    return Jv
end

function NLPModels.jtprod_lin!(
    nlp::SimpleNLPModel,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    @lencheck 2 x Jtv
    @lencheck 1 v
    increment!(nlp, :neval_jtprod)
    Jtv .= [v[1]; -2 * v[1]]
    return Jtv
end

function NLPModels.jtprod_nln!(
    nlp::SimpleNLPModel,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    @lencheck 2 x Jtv
    @lencheck 1 v
    increment!(nlp, :neval_jtprod)
    Jtv .= [-x[1] * v[1] / 2; -2 * x[2] * v[1]]
    return Jtv
end

function NLPModels.ghjvprod!(
    nlp::SimpleNLPModel,
    x::AbstractVector{T},
    g::AbstractVector{T},
    v::AbstractVector{T},
    gHv::AbstractVector{T},
) where {T}
    @lencheck get_nvar(nlp) x g v
    @lencheck get_ncon(nlp) gHv
    increment!(nlp, :neval_hprod)
    gHv .= [T(0); -g[1] * v[1] / 2 - 2 * g[2] * v[2]]
    return gHv
end

function NLPModels.jth_hess_coord!(
    nlp::SimpleNLPModel,
    x::AbstractVector,
    j::Integer,
    vals::AbstractVector{T},
) where {T}
    @lencheck 2 x
    @rangecheck 1 2 j
    @lencheck 2 vals
    if j == 1
        vals .= zero(T)
    else
        vals[1] = -T(1 / 2)
        vals[2] = -T(2)
    end
    return vals
end

function NLPModels.jth_hprod!(
    nlp::SimpleNLPModel,
    x::AbstractVector,
    v::AbstractVector,
    j::Integer,
    hv::AbstractVector{T},
) where {T}
    @lencheck 2 x v hv
    @rangecheck 1 2 j
    if j == 1
        hv .= zero(T)
    else
        hv[1] = -v[1] / 2
        hv[2] = -2 * v[2]
    end
    return hv
end
