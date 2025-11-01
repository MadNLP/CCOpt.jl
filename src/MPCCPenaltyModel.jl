abstract type AbstractMPCCPenaltyModel{T, VT} <: NLPModels.AbstractNLPModel{T, VT} end

function get_penalty(rnlp::M) where {M <: AbstractMPCCPenaltyModel}
    return rnlp.ρ[]
end

function set_penalty(rnlp::M, ρ::T) where {T, M <: AbstractMPCCPenaltyModel{T}}
    rnlp.ρ[] = ρ
    return nothing
end

function get_log_penalty(rnlp::M) where {T, M <: AbstractMPCCPenaltyModel{T}}
    return log(10, rnlp.ρ[])
end

