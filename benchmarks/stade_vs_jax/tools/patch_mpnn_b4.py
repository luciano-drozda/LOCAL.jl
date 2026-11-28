"""Builds the experiment variants of job MPNN2 from scripts of job MPNN1 (the mpnn_gather adjoint, STADE v0.6.39).
Usage: python3 patch_mpnn_b4.py IN.jl OUT.jl VARIANT
  IN is the script C (variant CA) or the script Z0 of patch_mpnn_split.py (the other variants). Only the kernel `cuda_kernel_mpnn_gather_b_4_p!` (the reverse edge loop) is changed, except in ZF.
  CA   exact. The adjoint adstb[idx] of the thread is kept in a register in the edge loop (read before the loop, written after it) instead of a read and a write of the global array in every iteration.
       The array is not aliased by the other arrays of the loop, and idx belongs to the thread. The result is CORRECT up to the order of the sum.
  ZB   (on Z0) the scatter into asrcb is removed. WRONG result, timing only.
  ZC   (on Z0) the four adds into efb are removed. WRONG result, timing only.
  ZD   (on Z0) the gather asrc[(j - 1) * nm + k] reads asrc[idx] (a coalesced read instead of a random row). WRONG result, timing only.
  ZE   (on Z0) the body of the edge loop keeps only the read of src[e]: the floor of the kernel without the loop work. WRONG result, timing only.
  ZF   (on Z0) the block-final atomic adds of all the kernels (`CUDA.@atomic w[__q] += __sh_w[__q]`) become plain adds. WRONG result, timing only.
The size bench_1e5 is verified against the CPU, as in patch_halfwarp.py."""
import re, sys
src = open(sys.argv[1]).read(); var = sys.argv[3]
assert var in ("CA", "ZB", "ZC", "ZD", "ZE", "ZF"), var
old = 'const SIZES = [("tiny", true), ("bench_1e5", false), ("bench_1e6", false)]'
if old in src:
    src = src.replace(old, old.replace('("bench_1e5", false)', '("bench_1e5", true)'))
name = "function cuda_kernel_mpnn_gather_b_4_p!"
a = src.index(name); b = src.index("\nfunction ", a + 10)
body = src[a:b]; n = {}
if var == "CA":
    o1 = "                    adstb[idx] = adstb[idx] + __cse_19\n"
    assert body.count(o1) == 1
    body = body.replace(o1, "                    __ac_adstb = __ac_adstb + __cse_19\n")
    o2 = "                for e = hi:-1:lo\n"
    assert body.count(o2) == 1
    body = body.replace(o2, "                __ac_adstb = adstb[idx]\n" + o2)
    # the end of the edge loop: the first `end` at the indent of the `for` line
    i = body.index(o2); j = body.index("\n                end\n", i) + len("\n                end\n")
    body = body[:j] + "                adstb[idx] = __ac_adstb\n" + body[j:]
elif var == "ZB":
    lines = [l for l in body.split("\n") if not l.strip().startswith("asrcb[(j - 1) * nm + k] +=")]
    n["removed"] = body.count("\n") - len(lines) + 1; body = "\n".join(lines)
elif var == "ZC":
    lines = [l for l in body.split("\n") if not l.strip().startswith("efb[")]
    n["removed"] = body.count("\n") - len(lines) + 1; body = "\n".join(lines)
elif var == "ZD":
    n["gather"] = body.count("asrc[(j - 1) * nm + k]"); body = body.replace("asrc[(j - 1) * nm + k]", "asrc[idx]")
elif var == "ZE":
    o2 = "                for e = hi:-1:lo\n"
    i = body.index(o2) + len(o2); j = body.index("\n                end\n", i)
    body = body[:i] + "                    j = src[e]\n                    __ac1_bmsgb = __ac1_bmsgb + j\n                    __ha1_bmsgb = 1" + body[j:]
src = src[:a] + body + src[b:]
if var == "ZF":
    pat = re.compile(r"^(\s*)CUDA\.@atomic (\w+)\[__q\] \+= (__sh_\w+)\[__q\]$", re.M)
    src, c = pat.subn(lambda m: f"{m.group(1)}{m.group(2)}[__q] += {m.group(3)}[__q]", src); n["flush"] = c
open(sys.argv[2], "w").write(src)
print(var, n)
