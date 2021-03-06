
"""
    AbstractLayout

Supertype for specifying the layout of new instances of arrays.
"""
abstract type AbstractLayout{T} end

""" MemoryLayout """
abstract type MemoryLayout{T} <: AbstractLayout{T} end

struct DynamicMemory{T} <: MemoryLayout{T}
    length::Int
end

struct FixedMemory{T} <: MemoryLayout{T}
    length::Int
end

struct StaticMemory{T,N} <: MemoryLayout{T} end

MemoryLayout{T}(length::Int) where {T} = FixedMemory{T}(length)
MemoryLayout{T}(::StaticInt{N}) where {T,N} = StaticMemory{T,N}()

""" AbstractArrayLayout """
abstract type AbstractArrayLayout{T,N,L} <: AbstractLayout{T} end

struct StrideLayout{T,N,L<:MemoryLayout{T},B,C,S,I<:Tuple{Vararg{Any,N}}} <: AbstractArrayLayout{T,N,L}
    layout::L
    contiguous_batch_size::B
    contiguous_axis::C
    strides::S
    indices::I

    function StrideLayout(x::AbstractArray{T,N}) where {T,N}
        inds = map(ArrayInterface.indices, axes(x))
        m = MemoryLayout{T}(prod(map(static_length, inds)))
        b = ArrayInterface.contiguous_batch_size(x)
        c = ArrayInterface.contiguous_axis(x)
        s = ArrayInterface.strides(x)
        return new{T,N,typeof(m),typeof(b),typeof(c),typeof(s),typeof(inds)}(
            m, b, c, s, inds
        )
    end
end

struct ArrayLayout{T,N,L<:AbstractLayout{T},I<:Tuple{Vararg{Any,N}}} <: AbstractArrayLayout{T,N,L}
    layout::L
    indices::I
end

abstract type SparseLayout{T,N,L} <: AbstractArrayLayout{T,N,L} end

struct DiagonalLayout{T,L<:ArrayLayout{T,1}} <: SparseLayout{T,2,L}
    layout::L
end

""" layout """
function layout(x::AbstractArray{T,N}) where {T,N}
    if ArrayInterface.defines_strides(x)
        return StrideLayout(x)
    else
        return ArrayLayout(x)
    end
    inds = map(ArrayInterface.indices, axes(x))
    return ArrayLayout(MemoryLayout{T}(prod(map(static_length, inds))), inds)
end
function layout(x::AbstractVector{T}) where {T}
    if ArrayInterface.can_change_size(x)
        return DynamicMemory(length(x))
    else
        return MemoryLayout{T}(length(x))
    end
end
layout(x::Diagonal) = DiagonalLayout(layout(x.diag))

""" layout """
layout(x::DynamicMemory, ::Type{T}) where {T} = DynamicMemory{T}(x.length)
layout(x::FixedMemory, ::Type{T}) where {T} = FixedMemory{T}(x.length)
layout(x::StaticMemory{T1,N}, ::Type{T2}) where {T1,T2,N} = StaticMemory{T2,N}()
layout(x::ArrayLayout, ::Type{T}) where {T} = ArrayLayout(layout(x.layout, T), x.axes)
layout(x::DiagonalLayout, ::Type{T}) where {T} = DiagonalLayout(layout(x.layout, T))
layout(x::AbstractArray, ::Type{T}) where {T} = layout(layout(x), T)

""" combine_layouts """
@inline function combine_layouts(bc::AbstractBroadcasted)
    typed_layout = Fix2(layout, Base.Broadcast.combine_eltypes(bc.f, bc.args))
    return broadcast_layout(map(typed_layout, bc.args)...)
end

# This is the where new subtypes of `AbstractLayout` need to specify how they combine with
# eachother
""" broadcast_layout(x::AbstractLayout, y::AbstractLayout) """
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


# TODO
""" vcat_layouts """

""" hcat_layouts """

""" cat_layouts """


