abstract type AbstractMPCCRelaxation{T, VT} <: NLPModels.AbstractNLPModel{T, VT} end

function get_relaxation(rnlp::M) where {M <: AbstractMPCCRelaxation}
    return rnlp.σ[]
end

function set_relaxation(rnlp::M, σ::T) where {T, M <: AbstractMPCCRelaxation{T}}
    rnlp.σ[] = σ
    return nothing
end

function get_log_relaxation(rnlp::M) where {T, M <: AbstractMPCCRelaxation{T}}
    return log(10, rnlp.σ[])
end

function initialize_relaxation(
    rnlp::M,
    σ::T,
    δ::T,
) where {T, M <: AbstractMPCCRelaxation{T}}
    rnlp.σ[] = σ
    return nothing
end
