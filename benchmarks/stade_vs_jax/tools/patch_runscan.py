"""Builds the experiment of job RS1: a copy of a generated STADE runner script (config C or G) in which the warp-level atomic add
`__stade_wa_add!(arr, idx, val::Float64)` sums over each run of adjacent lanes that add to one address (a segmented scan with shfl_up), and the last lane of the run does one atomic add.
Usage: python3 patch_runscan.py IN.jl OUT.jl
A warp with a lane that is not active uses the plain atomic add. The test is the same as the one of the helper of STADE (active_mask == full), so it is the same for all lanes of the warp
and every shuffle and vote is reached by all 32 lanes. A warp in which no run has two lanes uses the plain atomic add (the vote is uniform too).
The lane ids assume 1-D blocks, as the helper of STADE does. The size bench_1e5 is verified against the CPU, as in patch_halfwarp.py."""
import sys
src = open(sys.argv[1]).read()
old = 'const SIZES = [("tiny", true), ("bench_1e5", false), ("bench_1e6", false)]'
if old in src:
    src = src.replace(old, old.replace('("bench_1e5", false)', '("bench_1e5", true)'))
a = src.index('@inline function __stade_wa_add!(arr, idx::Integer, val::Float64)')
b = src.index('@inline function __stade_wa_add!(arr, idx::Integer, val::Float32)')
new = '''@inline function __stade_wa_up(m::UInt32, u::UInt64, k::Int32)
        lo = shfl_up_sync(m, UInt32(u & 0x00000000ffffffff), k)
        hi = shfl_up_sync(m, UInt32(u >> 32), k)
        return (UInt64(hi) << 32) | UInt64(lo)
    end

@inline function __stade_wa_add!(arr, idx::Integer, val::Float64)
        m = active_mask()
        if m == 0xffffffff
            i = Int64(idx)
            lane = UInt32((threadIdx().x - 1) & 31)
            prev = reinterpret(Int64, __stade_wa_up(m, reinterpret(UInt64, i), Int32(1)))
            hb = vote_ballot_sync(m, lane == 0x00000000 || i != prev)
            if hb != 0xffffffff
                start = 31 - leading_zeros(hb & (0xffffffff >> (31 - lane)))
                v = val
                for d in (Int32(1), Int32(2), Int32(4), Int32(8), Int32(16))
                    n = reinterpret(Float64, __stade_wa_up(m, reinterpret(UInt64, v), d))
                    if Int32(lane) - d >= Int32(start)
                        v += n
                    end
                end
                if lane == 0x0000001f || ((hb >> (lane + 0x00000001)) & 0x00000001) != 0x00000000
                    CUDA.@atomic arr[idx] += v
                end
                return nothing
            end
        end
        CUDA.@atomic arr[idx] += val
        return nothing
    end

'''
open(sys.argv[2], "w").write(src[:a] + new + src[b:])
