"""Builds the experiment variants of job MPNN1 from a generated runner script of the mpnn_gather adjoint (STADE v0.6.39, config C).
Usage: python3 patch_mpnn_split.py IN.jl OUT.jl VARIANT
  VARIANT   G0   every global add through the warp helper (`__stade_wa_runs!`, `__stade_wa_add!`) becomes the plain add `a[i] += v`. WRONG result (races), timing only.
            W0   every atomic add into a shared-memory array (`CUDA.@atomic __sh_NAME[i] += v`) becomes the plain add. WRONG result, timing only.
            Z0   G0 and W0 together. The block-final atomic adds into the global arrays stay. WRONG result, timing only.
            X16  the shared-memory atomic adds of the kernels b_3_p and b_5_p add the value of the lane l + 16 first (shfl_xor 16) and only the lanes 0 to 15 of a warp
                 do the atomic add. This is exact for nm = nf = 16 and a warp whose 32 lanes are active (the test is active_mask(), as in the helper of STADE),
                 because lane l and lane l + 16 then add to one address. A warp with an inactive lane does the plain atomic add. The result is CORRECT up to the order of the sum.
The kernels b_4_p (the adds sit behind a lane-dependent branch) and every other kernel are not changed by X16. The size bench_1e5 is verified against the CPU, as in patch_halfwarp.py."""
import re, sys
src = open(sys.argv[1]).read(); var = sys.argv[3]
assert var in ("G0", "W0", "Z0", "X16"), var
old = 'const SIZES = [("tiny", true), ("bench_1e5", false), ("bench_1e6", false)]'
if old in src:
    src = src.replace(old, old.replace('("bench_1e5", false)', '("bench_1e5", true)'))
n = {}

def plain_global(s):
    out = []; i = 0; cnt = 0
    for key in ("__stade_wa_runs!(", "__stade_wa_add!("):
        out = []; i = 0
        while True:
            j = s.find(key, i)
            if j < 0: out.append(s[i:]); break
            k = j + len(key); depth = 1; args = []; start = k; m = k
            while depth:
                c = s[m]
                if c == "(": depth += 1
                elif c == ")":
                    depth -= 1
                    if depth == 0: args.append(s[start:m].strip()); break
                elif c == "," and depth == 1: args.append(s[start:m].strip()); start = m + 1
                m += 1
            ls = s.rfind("\n", 0, j) + 1
            if len(args) == 3 and s[ls:j].strip() == "":        # a call statement, not a definition
                out.append(s[i:j]); out.append(f"{args[0]}[{args[1]}] += {args[2]}"); cnt += 1
            else:
                out.append(s[i:m + 1])
            i = m + 1
        s = "".join(out)
    n["global"] = cnt
    return s

SH = re.compile(r"^(\s*)CUDA\.@atomic (__sh_\w+)\[(.*)\] \+= (.*)$", re.M)
if var in ("G0", "Z0"): src = plain_global(src)
if var in ("W0", "Z0"):
    src, c = SH.subn(lambda m: f"{m.group(1)}{m.group(2)}[{m.group(3)}] += {m.group(4)}", src); n["shared"] = c
if var == "X16":
    helper = '''@inline function __x16(v::Float64)
        u = reinterpret(UInt64, v)
        lo = shfl_xor_sync(0xffffffff, UInt32(u & 0x00000000ffffffff), Int32(16))
        hi = shfl_xor_sync(0xffffffff, UInt32(u >> 32), Int32(16))
        return v + reinterpret(Float64, (UInt64(hi) << 32) | UInt64(lo))
    end

'''
    a = src.index("function cuda_kernel_mpnn_gather_b_3_p!")
    src = src[:a] + helper + src[a:]
    cnt = 0
    for name in ("cuda_kernel_mpnn_gather_b_3_p!", "cuda_kernel_mpnn_gather_b_5_p!"):
        a = src.index("function " + name); b = src.index("\nfunction ", a + 10) if "\nfunction " in src[a + 10:] else len(src)
        body = src[a:b]
        def rep(m):
            global cnt
            cnt += 1
            ind, arr, idx, val = m.groups()
            return (f"{ind}if active_mask() == 0xffffffff\n{ind}    __xv = __x16({val})\n{ind}    if ((threadIdx()).x - 1) & 16 == 0\n"
                    f"{ind}        CUDA.@atomic {arr}[{idx}] += __xv\n{ind}    end\n{ind}else\n{ind}    CUDA.@atomic {arr}[{idx}] += {val}\n{ind}end")
        body2 = SH.sub(rep, body)
        src = src[:a] + body2 + src[b:]
    n["x16"] = cnt
open(sys.argv[2], "w").write(src)
print(var, n)
