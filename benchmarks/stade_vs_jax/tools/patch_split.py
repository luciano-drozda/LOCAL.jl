"""Builds the timing-only variants of job ATOM3 from the script X of job ATOM1 (the half-warp reduction, made by patch_halfwarp.py).
Usage: python3 patch_split.py IN_X.jl OUT.jl GROUPS
  GROUPS is a comma list of the call sites of `__stade_wa_add!(array, index, value)` that become the plain add `array[index] += value`:
    shared   the arrays of shared memory (names that start with __sh_)
    scatter  the scatter into asrcb (random addresses)
    reduced  the adds into aggb, hb and efb (the global adds that the half-warp reduction already cuts to a sixteenth)
The result of the adjoint is WRONG (lost updates, races). Use the script only to time the kernels without these adds.
The block-final atomic adds (`CUDA.@atomic` outside the helper) and the adds into other arrays stay."""
import sys
GROUPS = {"shared": lambda a: a.startswith("__sh_"), "scatter": lambda a: a == "asrcb", "reduced": lambda a: a in ("aggb", "hb", "efb")}
src = open(sys.argv[1]).read(); want = [g for g in sys.argv[3].split(",") if g]
for g in want: assert g in GROUPS, g
key = "__stade_wa_add!("
out = []; i = 0; n = {g: 0 for g in want}
while True:
    j = src.find(key, i)
    if j < 0: out.append(src[i:]); break
    k = j + len(key); depth = 1; args = []; start = k; m = k
    while depth:                       # split at the commas of depth 1 up to the closing parenthesis
        c = src[m]
        if c == "(": depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0: args.append(src[start:m].strip()); break
        elif c == "," and depth == 1: args.append(src[start:m].strip()); start = m + 1
        m += 1
    line_start = src.rfind("\n", 0, j) + 1
    is_call = len(args) == 3 and src[line_start:j].strip() == ""       # a call statement (the definitions have `function` before it)
    g = next((g for g in want if is_call and GROUPS[g](args[0])), None)
    if g:
        out.append(src[i:j]); out.append(f"{args[0]}[{args[1]}] += {args[2]}"); n[g] += 1
    else:
        out.append(src[i:m + 1])
    i = m + 1
open(sys.argv[2], "w").write("".join(out))
print(n)
