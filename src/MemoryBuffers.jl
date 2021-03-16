module MemoryBuffers

using ArrayInterface
using ArrayInterface: AbstractDevice, CPUPointer, CPUTuple, static_length, StaticInt
using Base.Broadcast: Broadcasted
using Base: tail, @propagate_inbounds
using LinearAlgebra

include("layouts.jl")
include("buffers.jl")

""" allocate """
function allocate(::Union{CPUPointer,CPUTuple}, x::DynamicMemory{T}) where {T}
    return Vector{T}(undef, length(x))
end
function allocate(::Union{CPUPointer,CPUTuple}, x::StaticMemory{T,L}) where {T,L}
    return StaticBuffer{T,L}()
end
function allocate(::Union{CPUPointer,CPUTuple}, x::FixedMemory{T}) where {T}
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
    return materialize!(allocate(device(bc), combine_layouts(bc)), bc)
end

# this method needs to change to facilitate iteration or `f` needs to be initialized to be
# in charge of iterating for this to broadcast
function materialize!(dst, bc::Broadcasted)
    preserve(bc.f, buffer_pointer(dst, bc.args...))
    return dereference(dst)
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
