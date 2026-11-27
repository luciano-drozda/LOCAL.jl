"""Builds the experiment of job ATOM1: a copy of a generated STADE runner script (config C) in which the warp-level atomic add
`__stade_wa_add!(arr, idx, val::Float64)` reduces over the half-warp (16 lanes) when the 16 lanes add to one address, and then does one atomic add.
Usage: python3 patch_halfwarp.py IN.jl OUT_BASE.jl OUT_X.jl
  OUT_BASE.jl  the same script with the size bench_1e5 verified against the CPU (so that the new path is checked where nm = 16)
  OUT_X.jl     OUT_BASE.jl with the patched helper
The lane ids assume 1-D blocks (as the helper of STADE does). A half-warp with lanes that are not all active, or with different addresses, uses the plain atomic add.
It never falls through to a full-warp operation after a half-warp operation: the mask of a full-warp operation could name lanes that left."""
import sys
src = open(sys.argv[1]).read()
old = 'const SIZES = [("tiny", true), ("bench_1e5", false), ("bench_1e6", false)]'
# a script with another size list (or one already patched) is left as it is: only `tiny` is then verified against the CPU
base = src.replace(old, old.replace('("bench_1e5", false)', '("bench_1e5", true)')) if old in src else src
a = base.index('@inline function __stade_wa_add!(arr, idx::Integer, val::Float64)')
b = base.index('@inline function __stade_wa_add!(arr, idx::Integer, val::Float32)')
new = '''@inline function __stade_wa_lane_g(m::UInt32, u::UInt64, src::Int32)
        lo = shfl_sync(m, UInt32(u & 0x00000000ffffffff), src)
        hi = shfl_sync(m, UInt32(u >> 32), src)
        return UInt64(hi) << 32 | UInt64(lo)
    end

@inline function __stade_wa_add!(arr, idx::Integer, val::Float64)
        m = active_mask()
        hw = ((threadIdx()).x - 1) & 16
        gm = hw == 0 ? UInt32(0x0000ffff) : UInt32(0xffff0000)
        if (m & gm) == gm
            i = Int64(idx)
            ref = reinterpret(Int64, __stade_wa_lane_g(gm, reinterpret(UInt64, i), Int32(hw + 1)))
            if vote_all_sync(gm, i == ref)
                v = val
                v += reinterpret(Float64, __stade_wa_xor(gm, reinterpret(UInt64, v), Int32(8)))
                v += reinterpret(Float64, __stade_wa_xor(gm, reinterpret(UInt64, v), Int32(4)))
                v += reinterpret(Float64, __stade_wa_xor(gm, reinterpret(UInt64, v), Int32(2)))
                v += reinterpret(Float64, __stade_wa_xor(gm, reinterpret(UInt64, v), Int32(1)))
                if ((threadIdx()).x - 1) & 15 == 0
                    CUDA.@atomic arr[idx] += v
                end
                return nothing
            end
        end
        CUDA.@atomic arr[idx] += val
        return nothing
    end

'''
open(sys.argv[2], "w").write(base)
open(sys.argv[3], "w").write(base[:a] + new + base[b:])
