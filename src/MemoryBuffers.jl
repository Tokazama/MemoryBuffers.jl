module MemoryBuffers

using ArrayInterface
using ArrayInterface: AbstractDevice, CPUPointer, CPUTuple, static_length, StaticInt
using Base.Broadcast: Broadcasted
using Base: tail, @propagate_inbounds
using LinearAlgebra

export allocate, dereference, reallocate

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

"""
    MemoryLayout

Supertype for specifying the layout of new instances of arrays.
"""
abstract type MemoryLayout{T} end

struct DynamicBufferLayout{T} <: MemoryLayout{T}
    length::Int
end

struct BufferLayout{T} <: MemoryLayout{T}
    length::Int
end

struct StaticBufferLayout{T,N} <: MemoryLayout{T} end

abstract type AbstractArrayLayout{T,N} <: MemoryLayout{T} end

struct ArrayLayout{T,N,M<:MemoryLayout{T},Axes<:Tuple{Vararg{Any,N}}} <: AbstractArrayLayout{T,N}
    memory::M
    axes::Axes
end
Base.axes(x::ArrayLayout) = x.axes
memory_layout(x::ArrayLayout) = x.memory

struct DiagonalLayout{T,V<:ArrayLayout{T,1}} <: AbstractArrayLayout{T,2}
    layout::V
end
function Base.axes(x::DiagonalLayout)
    axis = first(axes(x.layout))
    return (axis, axis)
end
memory_layout(x::DiagonalLayout) = memory_layout(x.layout)


MemoryLayout{T}(n::Int) where {T} = BufferLayout{T}(n)
MemoryLayout{T}(::StaticInt{N}) where {T,N} = StaticBufferLayout{T,N}()
function MemoryLayout{T}(x::Tuple) where {T}
    return ArrayLayout{T}(MemoryLayout{T}(prod(map(static_length, x)), x))
end

""" layout """
layout(x::AbstractArray{T}) where {T} = MemoryLayout{T}(axes(x))
layout(x::Diagonal) = DiagonalLayout(layout(x.diag))

""" allocate """
function allocate(::Union{CPUPointer,CPUTuple}, x::DynamicBufferLayout{T}) where {T}
    return Vector{T}(undef, length(x))
end
function allocate(::Union{CPUPointer,CPUTuple}, x::StaticBufferLayout{T,L}) where {T,L}
    return StaticBuffer{T,L}()
end
function allocate(::Union{CPUPointer,CPUTuple}, x::BufferLayout{T}) where {T}
    return MutableBuffer{T}(length(x))
end

# FIXME - what should structured_buffer produce
function allocate(d::AbstractDevice, x::ArrayLayout)
    return structured_buffer(allocate(memory_layout(x)), axes(x))
end

""" dereference """
dereference(::CPUTuple, x::StaticBuffer) = x
function dereference(::CPUTuple, x::MutableBuffer)
    return ImmutableBuffer(materialize(CPUTuple(), x.data), x.length)
end
dereference(::CPUPointer, x::StaticBuffer) = x
dereference(::CPUPointer, x::MutableBuffer) = x

# splits (buffer, pointer) up and assigns them symbols say they can be passed to gc_preserve
@generated function preserve(f, args::Tuple{Vararg{Tuple{Any,Any},N}}) where {N}
    blk = Expr(:block)
    buffers = Vecotr{Symbol}(undef, N)
    ptrs = Vecotr{Symbol}(undef, N)
    for i in 1:N
        bufsym = buffers[i] = gensym(:b, i)
        ptrsym = pointers[i] = gensym(:p, i)
        ptrcall = :(@inbounds(args[$i]))
        push!(blk.args, Expr(:(=), Expr(:tuple, bufsym, ptrsym), ptrcall))
    end
    call = append!(Expr(:call, :f), ptrs)
    push!(blk.args, esc(Expr(:gc_preserve, call, buffers...)))
    return blk
end

# FIXME this needs to be figured out with structured_buffer
buffer_pointer(x) = x, pointer(x)

###
### broadcast interface
###
@inline function instantiate(bc::Broadcasted)
    return materialize!(allocate(device(bc), combine_layouts(bc.args...)), bc)
end

# this method needs to change to facilitate iteration or `f` needs to be initialized to be
# in charge of iterating for this to broadcast
function materialize!(dst, bc::Broadcasted)
    preserve(bc.f, buffer_pointer(dst, bc.args...))
    return dereference(dst)
end

@inline combine_layouts(A, B...) = broadcast_layout(layout(A), combine_layouts(B...))
@inline combine_layouts(A, B) = broadcast_layout(layout(A), layout(B))
combine_layout(A) = broadcast_layout(layout(A))


broadcast_layout(x) = x
broadcast_layout(x, y, zs...) = broadcast_layout(broadcast_layout(x, y), zs...)
function broadcast_layout(x::DiagonalLayout, y::DiagonalLayout)
    return DiagonalLayout(broadcast_layout(x.layout, y.layout))
end
function broadcast_layout(x::AbstractArrayLayout, y::AbstractArrayLayout)
    return ArrayLayout(
        combine_layout(memory_layout(x), memory_layout(y)),
        _bcs(axes(x), axes(y))
    )
end

_bcs(::Tuple{}, ::Tuple{}) = ()
_bcs(::Tuple{}, y::Tuple) = (first(y), _bcs((), tail(y))...)
_bcs(shape::Tuple, ::Tuple{}) = (shape[1], _bcs(tail(shape), ())...)
function _bcs(x::Tuple, y::Tuple)
    return (ArrayInterface.broadcast_axis(first(x), first(y)), _bcs(tail(x), tail(y))...)
end

###
### indexing interface
###
function unsafe_get_collection(A::AbstractArray{T}, inds) where {T}
    axs = to_axes(A, inds)
    dest = allocate(MemoryLayout{T}(axs))
    if map(Base.unsafe_length, axes(dest)) == map(Base.unsafe_length, axs)
        preserve(buffer_pointer(dest, A)) do d, a
            _unsafe_getindex!(d, a, inds...)
        end
    else
        Base.throw_checksize_error(dest, axs)
    end
    return dereference(dest)
end

end
