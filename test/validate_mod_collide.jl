# ==================== validate_mod_collide.jl =======================
# Checks the GPU split gate against a write index that two iterations can share (v0.5.56). CPU only, text of the generated CUDA and JACC code.
# An array written through an index built from the loop variable by `mod` alone (`u[mod(3t, n) + 1]`) has colliding iterations once the trip count
# passes the modulus. Equal read and write indices do not help: the loop must not become a parallel kernel, except an accumulate-only array, which
# an atomic add serves. A lone `mod` over a loop that visits at most as many consecutive values as the modulus (`for i = 1:n` and `mod(i, n)`, the periodic roll,
# and its reversed sweep in the adjoint) is one-to-one and stays parallel. A `div` beside the `mod` (the digits of a fused nest), a dimension that is the loop variable, and an index read from an
# array (a scatter-accumulate) keep their parallel kernel.
#   1. REFUSED:  an in-place update, and a plain write, through a lone `mod` index: no parallel kernel holds the `mod`, on both backends.
#   2. KEPT:     the accumulation, the fused nest, the loop-variable dimension, the gather: a parallel kernel holds the write.
#   3. SABOTAGE: the guard disabled makes case 1 fail.
#
#   julia validate_mod_collide.jl [--sabotage]

isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
using Logging

# (name, source, expected[, parallel kernels wanted[, expected when checkpointed]]): :refused or :kept
function mcl_cases()
    return Any[
        ("mc_rmw", """
function mc_rmw(u, a, out, n, nt)
    for t = 1:nt
        j = mod(3 * t, n) + 1
        u[j] = u[j] * a + sin(u[j])
    end
    s = 0.0
    for i = 1:n
        s = s + u[i] * u[i]
    end
    out[1] = s
    return nothing
end""", :refused),
        ("mc_branch", """
function mc_branch(u, a, out, n, nt, lim)
    for t = 1:nt
        if t <= lim
            j = mod(3 * t, n) + 1
            u[j] = u[j] * a + sin(u[j])
        end
    end
    s = 0.0
    for i = 1:n
        s = s + u[i] * u[i]
    end
    out[1] = s
    return nothing
end""", :refused),
        ("mc_roll", """
function mc_roll(y, x, out, n)
    for i = 1:n
        y[mod(i, n) + 1] = x[i] * x[i]
    end
    s = 0.0
    for k = 1:n
        s = s + y[k] * y[k]
    end
    out[1] = s
    return nothing
end""", :kept, 2),
        ("mc_roll_shift", """
function mc_roll_shift(y, x, out, n)
    for i = 1:n
        j = mod(i - 1, n) + 1
        y[j] = y[j] * x[i] + sin(y[j])
    end
    s = 0.0
    for k = 1:n
        s = s + y[k] * y[k]
    end
    out[1] = s
    return nothing
end""", :kept, 2, :refused),
        ("mc_roll_long", """
function mc_roll_long(y, x, out, n, nt)
    for i = 1:nt
        y[mod(i, n) + 1] = x[i] * x[i]
    end
    s = 0.0
    for k = 1:n
        s = s + y[k] * y[k]
    end
    out[1] = s
    return nothing
end""", :refused),
        ("mc_roll_lit_ok", """
function mc_roll_lit_ok(y, x, out, n)
    for i = 1:8
        y[mod(i, 10) + 1] = x[i] * x[i]
    end
    s = 0.0
    for k = 1:n
        s = s + y[k] * y[k]
    end
    out[1] = s
    return nothing
end""", :kept, 2),
        ("mc_roll_lit_long", """
function mc_roll_lit_long(y, x, out, n)
    for i = 1:12
        y[mod(i, 10) + 1] = x[i] * x[i]
    end
    s = 0.0
    for k = 1:n
        s = s + y[k] * y[k]
    end
    out[1] = s
    return nothing
end""", :refused),
        ("mc_roll_stride", """
function mc_roll_stride(y, x, out, n)
    for i = 1:n
        y[mod(2 * i, n) + 1] = x[i] * x[i]
    end
    s = 0.0
    for k = 1:n
        s = s + y[k] * y[k]
    end
    out[1] = s
    return nothing
end""", :refused),
        ("mc_plain", """
function mc_plain(u, x, out, n, nt)
    for t = 1:nt
        j = mod(t, n) + 1
        u[j] = x[t] * x[t]
    end
    s = 0.0
    for i = 1:n
        s = s + u[i] * u[i]
    end
    out[1] = s
    return nothing
end""", :refused),
        ("mc_accum", """
function mc_accum(u, x, out, n, nt)
    for t = 1:nt
        j = mod(t, n) + 1
        u[j] = u[j] + x[t] * x[t]
    end
    s = 0.0
    for i = 1:n
        s = s + u[i] * u[i]
    end
    out[1] = s
    return nothing
end""", :kept),
        ("mc_fused", """
function mc_fused(y, x, a, out, n, m)
    for idx = 1:n * m
        i = div(idx - 1, m) + 1
        j = mod(idx - 1, m) + 1
        y[i, j] = y[i, j] * a + sin(x[i, j])
    end
    s = 0.0
    for k = 1:n
        s = s + y[k, 1] * y[k, 1]
    end
    out[1] = s
    return nothing
end""", :kept),
        ("mc_dim", """
function mc_dim(u, a, out, n, nt)
    for t = 1:nt
        u[mod(3 * t, n) + 1, t] = u[mod(3 * t, n) + 1, t] * a + sin(u[mod(3 * t, n) + 1, t])
    end
    s = 0.0
    for i = 1:n
        s = s + u[i, 1] * u[i, 1]
    end
    out[1] = s
    return nothing
end""", :kept),
        ("mc_gather", """
function mc_gather(u, idx, x, out, n, nt)
    for t = 1:nt
        j = idx[t]
        u[j] = u[j] + x[t] * x[t]
    end
    s = 0.0
    for i = 1:n
        s = s + u[i] * u[i]
    end
    out[1] = s
    return nothing
end""", :kept),
    ]
end

# text of every parallel kernel of the plan, one string per kernel (the serial and glue kernels are not parallel)
function mcl_parallel_kernels(plan, prefix::String)
    out = String[]
    for k in plan.kernels
        k isa Expr || continue
        nm = string(k.args[1].args[1])
        startswith(nm, prefix) && push!(out, string(k))
    end
    return out
end

function validate_mod_collide(; verbose::Bool = true)
    checks = 0; bad = 0
    say(ok, msg) = (checks += 1; ok || (bad += 1; verbose && println("NOT OK ", msg)))
    for case in mcl_cases()
        (name, src, expect0) = case[1:3]; kmin = length(case) >= 4 ? case[4] : 1
      for ckpt in (false, true)
        expect = ckpt && length(case) >= 5 ? case[5] : expect0   # a strip-mined segment has a run-time length: no proof, stays sequential
        r = with_logger(NullLogger()) do
            ckpt ? STADE.agen_ckpt_adjoint(Meta.parse(src); min_trip = 0) : STADE.stade_adjoint(Meta.parse(src); keep_push_pop = false)
        end
        r === nothing && continue   # no region to strip-mine (the fused nest, the gather)
        cu = STADE.stade_gpu(r.adjoint, STADE.cgen_backend_cuda(); inbounds = false, dead_scalars = false, split_reduce = 0, contraction = 0)
        ja = STADE.stade_jacc(r.adjoint; inbounds = false, dead_scalars = false, split_reduce = 0, warp_atomics = false, jacc_contraction = 0)
        for (label, plan, prefix) in (("cuda", cu, "cuda_kernel_"), ("jacc", ja, "jacc_kernel_"))
            label = ckpt ? label * " checkpointed" : label
            ks = mcl_parallel_kernels(plan, prefix)
            nk = count(k -> occursin("mod(", k) || occursin("idx[", k), ks)
            if expect === :refused
                say(!any(k -> occursin("mod(", k), ks), "$name $label: a parallel kernel holds the colliding `mod` index")
            else
                say(nk >= kmin, "$name $label: $nk parallel kernel(s) hold the write, $kmin expected (the guard refused a loop it should keep)")
            end
        end
      end
    end
    verbose && println(bad == 0 ? "validate_mod_collide: $checks checks passed" : "validate_mod_collide: $bad of $checks checks FAILED")
    return bad
end

# a guard disabled, or the one-to-one proof made always true: the refused cases must fail
function validate_mod_collide_sabotage(; verbose::Bool = true)
    here = abspath(@__FILE__)
    src = read(joinpath(@__DIR__, "..", "src", "STADE.jl"), String)
    bad = 0
    sabs = Any[("the guard disabled", "        collide = cgen_writes_collide_by_mod(widxs, merge(summary.defs, cgen_nested_scalar_defs(body)), loopvar, bounds)\n", "        collide = false\n"),
               ("the proof always true", "    bounds === nothing && return false\n    while e isa Expr && e.head == :call && e.args[1] in (:+, :-) && length(e.args) == 3\n", "    return true\n    while e isa Expr && e.head == :call && e.args[1] in (:+, :-) && length(e.args) == 3\n"),
               ("the trip count not checked", "        return m >= 1 && abs(hi - lo) + 1 <= m\n", "        return true\n"),
               ("the modulus not compared", "        return lo isa Integer && lo >= 1 && hi == m\n", "        return lo isa Integer && lo >= 1\n")]
    for (label, anchor, repl) in sabs
        count(anchor, src) == 1 || (verbose && println("sabotage anchor not found: ", label); bad += 1; continue)
        dir = mktempdir()
        mkpath(joinpath(dir, "src")); mkpath(joinpath(dir, "test"))
        write(joinpath(dir, "src", "STADE.jl"), replace(src, anchor => repl))
        cp(here, joinpath(dir, "test", "validate_mod_collide.jl"))
        out = read(ignorestatus(`$(Base.julia_cmd()) --startup-file=no $(joinpath(dir, "test", "validate_mod_collide.jl"))`), String)
        failed = occursin(r"FAILED", out)
        verbose && println(failed ? "sabotage caught: $label fails the validator" : "SABOTAGE NOT CAUGHT: $label")
        failed || (bad += 1)
    end
    return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    rc = "--sabotage" in ARGS ? validate_mod_collide_sabotage() : validate_mod_collide()
    exit(rc == 0 ? 0 : 1)
end
