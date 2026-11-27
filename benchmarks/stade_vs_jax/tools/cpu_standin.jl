# Stand-in for CUDA.jl on a machine without a GPU. cpu_standin.py splices this file into a generated runner script
# in place of the GPU code, so the whole runner (phases, JSON, memory bookkeeping) runs on the CPU.
# CuArray makes a registered copy. The "device memory in use" is the sum of the registered arrays that are still alive.
# Test use only: it proves the logic of the runner, not the speed or the memory of a GPU.
module CUDA
export CuArray
const FAKE = true
const REG = WeakRef[]
reg!(a) = (push!(REG, WeakRef(a)); a)
function register_all!(r)
    r === nothing && return r
    for x in (r isa Tuple ? r : (r,)); x isa AbstractArray && reg!(x); end
    return r
end
CuArray(x::AbstractArray) = reg!(copy(x))
zeros(::Type{T}, dims...) where {T} = reg!(Base.zeros(T, dims...))
synchronize() = nothing
module Profile; start() = nothing; stop() = nothing; end   # the profiler range of the prof phases
reclaim() = (filter!(w -> w.value !== nothing, REG); nothing)
used_bytes() = sum((sizeof(w.value) for w in REG if w.value !== nothing); init = 0)
total_memory() = 16 * 10^9
free_memory() = total_memory() - used_bytes()
used_memory() = used_bytes()
driver_version() = v"12.4.0"
runtime_version() = v"12.9.0"
device() = 0
name(_) = "CPU stand-in"
allowscalar(_) = nothing
macro elapsed(ex); :(Base.@elapsed $(esc(ex))); end
macro allocated(ex); :(Base.@allocated $(esc(ex))); end
end

# the device entry points of a kernel are the CPU functions; the stacks are registered as device arrays
function standin_alias!(name)
    cpu = getfield(Main, Symbol(name)); cpub = getfield(Main, Symbol(name * "_b")); cpui = getfield(Main, Symbol("initstacks_" * name * "_b"))
    Core.eval(Main, :(const $(Symbol(name * "_cuda")) = $cpu))
    Core.eval(Main, :(const $(Symbol(name * "_b_cuda")) = $cpub))
    an = Base.method_argnames(first(methods(cpui)))[2:end]
    fn = Symbol("initstacks_" * name * "_b_cuda")
    Core.eval(Main, :(function $fn($(an...)); CUDA.register_all!($cpui($(an...))); end))
    return nothing
end
