# TODO(@anton) Should this have an abstract type. I actually don't think we need to, this is a "concrete" type
struct VerticalMPCCMeta{T,S}
    # index sets
    indGlift::Vector{Int}
    indHlift::Vector{Int}
    indOrig::Vector{Int}

    # lengths
    nGlift::Int
    nHlift::Int
end
