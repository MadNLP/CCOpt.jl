# A vector that takes a set of indices and only allows manipulation of that subset of indices
# It pretends to be the original size but returns zero(T) when nonexistant indices are gotten and is a noop when nonexistant indices are assigned
# TODO: limit type T to be numeric
# TODO: This probably does not work well with @view

struct MappedVector{T, VT <: AbstractVector{T}} <: AbstractVector{T}
    par_vec::Base.RefValue{VT}
    ind_set::AbstractDict{Int, Int}
    len::Int
end

function MappedVector(
    par_vec::AbstractVector{T},
    inds::AbstractVector{Int},
    n::Int,
) where {T}
    # TODO(@anton) ideally this does not allocate any new memory.
    return MappedVector(Ref(par_vec), Dict([(ind, i) for (i, ind) in enumerate(inds)]), n)
end

function Base.size(A::MappedVector{T, VT}) where {T, VT}
    return (A.len,)
end

function Base.length(A::MappedVector{T, VT}) where {T, VT}
    return A.len
end

function Base.axes(A::MappedVector{T, VT}) where {T, VT}
    return (Base.OneTo(A.len),)
end

function Base.axes(A::MappedVector{T, VT}, i::Int) where {T, VT}
    if i==1
        return Base.OneTo(A.len)
    else
        return Base.OneTo(1)
    end
end

Base.ndims(A::MappedVector{T, VT}) where {T, VT} = 1

#function checkindex(A)

function Base.getindex(A::MappedVector{T, VT}, i::Int) where {T, VT}
    if haskey(A.ind_set, i)
        return A.par_vec[][A.ind_set[i]]
    else
        return zero(T)
    end
end

function Base.getindex(A::MappedVector{T, VT}, i::AbstractRange{Int}) where {T, VT}
    return [haskey(A.ind_set, ii) ? A.par_vec[][A.ind_set[ii]] : zero(T) for ii in i]
end

function Base.getindex(A::MappedVector{T, VT}, i::AbstractVector{Int}) where {T, VT}
    return [haskey(A.ind_set, ii) ? A.par_vec[][A.ind_set[ii]] : zero(T) for ii in i]
end

function Base.setindex!(A::MappedVector{T, VT}, x::T, i::Int) where {T, VT}
    if haskey(A.ind_set, i)
        par_vec = A.par_vec[][A.ind_set[i]] = x
    end
end

function Base.setindex!(
    A::MappedVector{T, VT},
    x::AbstractVector{T},
    i::IndexCartesian,
) where {T, VT}
    # TODO(@anton) This may cause bugs in the future but for now assume that we are always a VECTOR
    #              Should be a safe assumption but who knows what crazy things an NLPModel may do
    lin_i = LinearIndices(i) # TODO Does this copy a bunch?
    show(lin_i)
    return A.par_vec[][[A.ind_set(ii) for ii in lin_i if haskey(A.ind_set, ii)]] = x[[ii for ii in lin_i if haskey(A.indset, ii)]]
end

function Base.setindex!(
    A::MappedVector{T, VT},
    x::AbstractVector{T},
    i::AbstractRange{Int},
) where {T, VT}
    # TODO(@anton) check how efficient this is
    return A.par_vec[][[A.ind_set(ii) for ii in i if haskey(A.ind_set, ii)]] = x[[ii for ii in i if haskey(A.indset, ii)]]
end

function Base.setindex!(
    A::MappedVector{T, VT},
    x::AbstractVector{T},
    i::AbstractVector{Int},
) where {T, VT}
    return A.par_vec[][[A.ind_set(ii) for ii in i if haskey(A.ind_set, ii)]] = x[[ii for ii in i if haskey(A.indset, ii)]]
end

function Base.nextind(A::MappedVector{T, VT}, i::Integer) where {T, VT} end
