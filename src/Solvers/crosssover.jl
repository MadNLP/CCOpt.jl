function solve_lpcc(
    lpcc::AbstractMPCCModel{T,VT},
    x::VT,
    ::Union{MadNLPCSolver,ExactPenaltySolver};
    kwargs...
)
    
end

function crossover(
    mpcc::AbstractMPCCModel{T, VT},
    x0::VT
) where {T, VT}
    # TODO(@anton) check
    
end
