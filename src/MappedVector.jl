# A vector that takes a set of indices and only allows manipulation of that subset of indices
# It pretends to be the original size but returns zero(T) when nonexistant indices are gotten and is a noop when nonexistant indices are assigned
# TODO: limit type T to be numeric
# TODO: This probably does not work well with @view

struct MappedVector{T, VT} <: AbstractVector{T}
    par_vec::Base.RefValue{VT}
    ind_set::Set{Int}
end

function Base.size(A::MappedVector{T, VT}) where {T, VT}
    return size(A.par_vec[])
end

function Base.length(A::MappedVector{T, VT}) where {T, VT}
    return length(A.par_vec[])
end


function Base.getindex(A::MappedVector{T,VT}, i::Int) where {T,VT}
    if i ∈ A.ind_set
        return A.par_vec[][i]
    else
        return zero(T)
    end
end

function Base.getindex(A::MappedVector{T,VT}, i::AbstractRange{Int}) where {T,VT}
    out = A.par_vec[][i]
    out[i ∉ A.ind_set] = zero(T)
    return out
end

function Base.getindex(A::MappedVector{T,VT}, i::AbstractVector{Int}) where {T,VT}
    out = A.par_vec[][i]
    out[i ∉ A.ind_set] = zero(T)
    return out
end

function Base.setindex!(A::MappedVector{T,VT}, x::T, i::Int) where {T,VT}
    if i ∈ A.ind_set
        par_vec = A.par_vec[][i] = x
    end
end

function Base.setindex!(A::MappedVector{T,VT}, x::AbstractVector{T}, i::AbstractRange{Int}) where {T,VT}
    # TODO(@anton) check how efficient this is.
    A.par_vec[][filter((x) -> x ∈ A.ind_set,i)] = x[filter((x) -> x ∈ A.ind_set,i)]
end

function Base.setindex!(A::MappedVector{T,VT}, x::AbstractVector{T}, i::AbstractVector{Int}) where {T,VT}
    A.par_vec[][filter((x) -> x ∈ A.ind_set,i)] = x[filter((x) -> x ∈ A.ind_set,i)]
end

function Base.nextind(A::MappedVector{T,VT}, i::Integer) where {T,VT}

end
