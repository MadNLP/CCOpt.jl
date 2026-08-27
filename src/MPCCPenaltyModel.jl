abstract type AbstractMPCCPenaltyModel{T,VT} <: NLPModels.AbstractNLPModel{T,VT} end

function get_penalty(pnlp::M) where {M<:AbstractMPCCPenaltyModel}
    return pnlp.ρ[]
end

function set_penalty(pnlp::M, ρ::T) where {T,M<:AbstractMPCCPenaltyModel{T}}
    pnlp.ρ[] = ρ
    return nothing
end

function get_log_penalty(pnlp::M) where {T,M<:AbstractMPCCPenaltyModel{T}}
    return log(10, get_penalty(pnlp))
end
