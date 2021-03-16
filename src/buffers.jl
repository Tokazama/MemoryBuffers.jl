
abstract type AbstractMemoryBuffer{T} <: AbstractVector{T} end

abstract type AbstractMutableBuffer{T} <: AbstractMemoryBuffer{T} end

abstract type AbstractImmutableBuffer{T} <: AbstractMemoryBuffer{T} end

mutable struct StaticBuffer{T,N} <: AbstractMutableBuffer{T}
    data::NTuple{N,T}
end

struct MutableBuffer{T} <: AbstractMutableBuffer{T}
    data::StaticBuffer{T}
    length::Int

    MutableBuffer{T}(n::Int) where {T} = new{T}(StaticBuffer{T,n}(), n)
end

struct StaticImmutableBuffer{T,N} <: AbstractImmutableBuffer{T}
    data::NTuple{N,T}
end

struct ImmutableBuffer{T} <: AbstractImmutableBuffer{T}
    data::StaticImmutableBuffer{T}
    length::Int

    ImmutableBuffer(data::StaticImmutableBuffer{T}, length::Int) where {T} = new{T}(data, length)
end

@inline function Base.unsafe_convert(::Type{Ptr{T}}, d::StaticBuffer) where {T}
    return Base.unsafe_convert(Ptr{T}, Base.pointer_from_objref(d.data.data))
end
@inline function Base.unsafe_convert(::Type{Ptr{T}}, d::MutableBuffer) where {T}
    return Base.unsafe_convert(Ptr{T}, Base.pointer_from_objref(d.data.data))
end

@propagate_inbounds function Base.getindex(m::AbstractMutableBuffer{T}, i::Int) where {T}
    @boundscheck checkbounds(m, i)
    GC.@preserve m x = unsafe_load(pointer(m), i)
    return x
end
@propagate_inbounds function Base.getindex(m::StaticImmutableBuffer{T}, i::Int) where {T}
    @boundscheck checkbounds(m, i)
    return @inbounds(getfield(m.data, i))
end
@propagate_inbounds function Base.getindex(m::ImmutableBuffer{T}, i::Int) where {T}
    @boundscheck checkbounds(m, i)
    return @inbounds(getfield(m.data.data, i))
end
@propagate_inbounds function Base.setindex!(m::AbstractMutableBuffer{T}, x, i::Int) where {T}
    @boundscheck checkbounds(m, i)
    GC.@preserve m unsafe_store!(pointer(m), convert(T, x), i)
end

ArrayInterface.known_length(::Type{StaticImmutableBuffer{T,N}}) where {T,N} = N
ArrayInterface.known_length(::Type{StaticBuffer{T,N}}) where {T,N} = N
ArrayInterface.size(m::MutableBuffer) = (static_length(m),)

""" grow_beg! """
function grow_beg!(x, n::Integer)
    n < 0 && throw(ArgumentError("new length must be ≥ 0"))
    return unsafe_grow_beg!(x, n)
end
unsafe_grow_beg!(x::Vector, n) = Base._growbeg!(x, n)
unsafe_grow_beg!(x::AbstractVector, n) = unsafe_grow_beg!(parent(x), n)

###
### grow_end!
###
""" grow_end! """
function grow_end!(x, n::Integer)
    n < 0 && throw(ArgumentError("new length must be ≥ 0"))
    unsafe_grow_end!(x, n)
    return x
end
unsafe_grow_end!(x::Vector, n) = Base._growend!(x, n)
unsafe_grow_end!(x::AbstractVector, n) = unsafe_grow_end!(parent(x), n)

""" grow_to! """
function grow_to!(x, n::Integer)
    len = length(x)
    if len < n
        unsafe_grow_to!(x, n)
        return x
    elseif len == n
        return x
    else
        throw(ArgumentError("new length must be ≥ than length of collection, got length $(length(x))."))
    end
end
unsafe_grow_to!(x, n::Integer) = unsafe_grow_end!(x, n - length(x))

""" shrink_beg! """
function shrink_beg!(x, n::Integer)
    n > length(x) && throw(ArgumentError("new length cannot be < 0"))
    return unsafe_shrink_beg!(x, n)
end
unsafe_shrink_beg!(x::Vector, n) = Base._deletebeg!(x, n)
unsafe_shrink_beg!(x::AbstractVector, n) = unsafe_shrink_beg!(parent(x), n)

""" shrink_end! """
function shrink_end!(x, n::Integer)
    n < 0 && throw(ArgumentError("new length must be ≥ 0"))
    return unsafe_shrink_end!(x, n)
end
unsafe_shrink_end!(x::Vector, n) = Base._deleteend!(x, n)
unsafe_shrink_end!(x::AbstractVector, n) = unsafe_shrink_end!(parent(x), n)

""" shrink_to! """
function shrink_to!(x, n::Integer)
    len = length(x)
    if len > n
        return unsafe_shrink_to!(x, n)
    elseif len == n
        return x
    else
        throw(ArgumentError("new length must be ≤ than length of collection, got length $(length(x))."))
    end
end

unsafe_shrink_to!(x, n) = unsafe_shrink_end!(x, length(x) - n)

