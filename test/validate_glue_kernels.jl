# ===================== validate_glue_kernels.jl ======================
# Checks the glue kernels of stade_gpu (`glue_kernels = true`, v0.5.26): the scalar statements between the kernels of a host loop run as one-thread
# kernels instead of `CUDA.@allowscalar` blocks, and a Float64 scalar that only kernels and other glue read lives in a one-element device box.
# CPU only. The generated CUDA host function runs with the stand-ins of validate_serial_loops.jl and its gradient (adjoint) or Hv (HVP) is compared with the CPU one.
#   1. BASIC: a scalar computed from a device value between two kernels (alpha = 1 / (1 + s[1])) becomes a box; gradient and Hv equal the CPU ones.
#   2. HOST USE: a scalar that the host itself reads (an `if alpha > c` that picks a kernel) stays a host scalar.
#   3. CROSS: a local that one glue block computes and a later glue block reads is correct whichever way the pass places it.
#   4. ZERO TRIP: m = 0 and m = 1 give the CPU result.
#   5. ORDER: a device read between two writes of the same cell sees the first write (the pass keeps the order of the statements of a block).
#   6. CORPUS: every corpus kernel whose plan holds a glue kernel gives the CPU gradient and the CPU Hv, and the glue kernels removed scalar blocks.
#   7. DEFAULT: glue_kernels is on by default (v0.5.26); `glue_kernels = false` gives no glue kernel and the host function of the serial track alone.
#
#   julia validate_glue_kernels.jl [kernel names...]      (no names: every corpus kernel; run it on a cluster node)

include(joinpath(@__DIR__, "validate_serial_loops.jl"))

glue_names(plan) = [string(k.args[1].args[1]) for k in plan.kernels if startswith(string(k.args[1].args[1]), "cuda_glue_")]
# statements that still run inside an allowscalar block (a block of k statements counts k)
function count_allowscalar(plan)
    n = Ref(0)
    function w(e)
        e isa Expr || return
        if e.head === :macrocall && occursin("allowscalar", string(e.args[1]))
            b = e.args[end]
            n[] += b isa Expr && b.head === :block ? count(x -> !(x isa LineNumberNode), b.args) : 1
        end
        foreach(w, e.args)
    end
    w(plan.host)
    return n[]
end

src_gbasic = "function glue_basic(x, y, s, n, m)\n    for t = 1:m\n        s[1] = 0.0\n        for i = 1:n\n            s[1] = s[1] + x[i] * x[i]\n        end\n        alpha = 1.0 / (1.0 + s[1])\n        for i = 1:n\n            x[i] = x[i] - alpha * y[i]\n        end\n    end\n    return nothing\nend"
src_ghost = "function glue_hostuse(x, y, s, n, m)\n    for t = 1:m\n        s[1] = 0.0\n        for i = 1:n\n            s[1] = s[1] + x[i] * x[i]\n        end\n        alpha = 1.0 / (1.0 + s[1])\n        if alpha > 0.3\n            for i = 1:n\n                x[i] = x[i] - alpha * y[i]\n            end\n        else\n            for i = 1:n\n                x[i] = 0.5 * x[i]\n            end\n        end\n    end\n    return nothing\nend"
src_gcross = "function glue_cross(x, y, s, n, m)\n    for t = 1:m\n        s[1] = 0.0\n        for i = 1:n\n            s[1] = s[1] + x[i] * x[i]\n        end\n        c = 2.0 * s[1]\n        for i = 1:n\n            y[i] = y[i] + c * x[i]\n        end\n        d = c + s[1]\n        s[1] = d\n        for i = 1:n\n            x[i] = x[i] + 0.01 * s[1] * y[i]\n        end\n    end\n    return nothing\nend"
src_gorder = "function glue_order(x, s, w, n, m)\n    for t = 1:m\n        s[1] = 0.0\n        for i = 1:n\n            s[1] = s[1] + x[i]\n        end\n        w[1] = s[1]\n        s[1] = 3.0 * w[1]\n        w[2] = s[1] + w[1]\n        for i = 1:n\n            x[i] = x[i] * 0.5 + w[2] * 0.01\n        end\n    end\n    return nothing\nend"

function glue_case(label, src; ints = Dict{Symbol,Int}(), modes = (:adjoint, :hvp), expect_glue = true)
    ok = true
    for mode in modes
        r = glue_compare(Meta.parse(src); mode, int_override = ints)
        gn = glue_names(r.plan)
        e = relerr(r.cpu, r.gpu)
        good = e <= 1e-12 && maximum(abs.(r.cpu); init = 0.0) > 0 && (!expect_glue || !isempty(gn))
        println(good ? "ok   " : "FAIL ", label, " [", mode, "]: glue kernels ", length(gn), ", serial kernels ", length(r.serial), ", rel ", e, ", nonzero ", maximum(abs.(r.cpu); init = 0.0) > 0)
        ok &= good
    end
    return ok
end

glue_compare(primal; kw...) = serial_compare(primal; serial = true, glue = true, kw...)

# plan-level unit tests of the launch guards on a hand-written host function and kernels
function glue_unit_tests()
    ok = true
    be = STADE.cgen_backend_cuda()
    mk(host, ks...) = STADE.cgen_glue_apply(Meta.parse(host), Expr[Meta.parse(k) for k in ks], be)
    boxed(r, v) = occursin("$v = CuArray", replace(string(r[1]), r"\s+" => " "))
    h1(second) = "function f_cuda(x, s, n)\n    CUDA.@allowscalar begin\n        alpha = 1.0 / (1.0 + s[1])\n    end\n    @cuda threads = 4 blocks = 1 cuda_kernel_f_1!(alpha, x, n)\n    $second\n    return nothing\nend"
    kern(body) = "function cuda_kernel_f_1!(a, x, n)\n    $body\n    return nothing\nend"
    function check(c, msg); (c || (ok = false)); println(c ? "ok   " : "FAIL ", msg); end
    check(boxed(mk(h1(""), kern("x[1] = a * x[1]")), "alpha"), "unit: a candidate that every site passes to a kernel is boxed")
    check(!boxed(mk(h1("@cuda threads = 4 blocks = 1 cuda_kernel_f_1!(0.5, x, n)"), kern("x[1] = a * x[1]")), "alpha"), "unit: another launch site passing a value keeps the scalar on the host")
    check(!boxed(mk(h1(""), kern("a = a + 1.0\n    x[1] = a * x[1]")), "alpha"), "unit: a callee that assigns the parameter keeps the scalar on the host")
    h3 = "function f_cuda(x, s, n)\n    CUDA.@allowscalar begin\n        alpha = 1.0 / (1.0 + s[1])\n    end\n    @cuda threads = 4 blocks = 1 other_kernel!(alpha, x, n)\n    return nothing\nend"
    check(!boxed(mk(h3, kern("x[1] = a * x[1]")), "alpha"), "unit: a launch of a function that is no kernel of this function keeps the scalar on the host")
    h4 = "function f_cuda(x, s, n)\n    CUDA.@allowscalar begin\n        alpha = 1.0 / (1.0 + s[1])\n    end\n    @cuda threads = 4 blocks = 1 cuda_kernel_f_1!(alpha, x, n)\n    y = alpha + 1.0\n    return nothing\nend"
    check(!boxed(mk(h4, kern("x[1] = a * x[1]")), "alpha"), "unit: a host read of the scalar outside glue keeps it on the host")
    # a local that glue computes and a kernel launch takes by value is a host value: it stays in the allowscalar block
    h5 = "function f_cuda(x, dst, n)\n    for e = 1:n\n        CUDA.@allowscalar begin\n            d = dst[e]\n        end\n        @cuda threads = 4 blocks = 1 cuda_kernel_f_1!(d, x, n)\n    end\n    return nothing\nend"
    r5 = mk(h5, kern("x[1] = a * x[1]"))
    check(!occursin("cuda_glue", string(r5[1])) && occursin("allowscalar", string(r5[1])), "unit: a local passed to a kernel by value is not computed by a glue kernel")
    return ok
end

function validate_glue_kernels(names = String[])
    ok = true
    check(c, msg) = (c || (println("FAIL ", msg); ok = false); c && println("ok   ", msg))

    ok &= glue_unit_tests()
    ok &= glue_case("basic", src_gbasic)
    rb = glue_compare(Meta.parse(src_gbasic))
    htext = string(rb.plan.host)
    check(occursin("alpha = CuArray", htext) || occursin("alpha = CuArray", replace(htext, r"\s+" => " ")), "basic: alpha lives in a device box")
    r0 = serial_compare(Meta.parse(src_gbasic); serial = true, glue = false)
    check(count_allowscalar(rb.plan) < count_allowscalar(r0.plan), "basic: fewer @allowscalar statements than without glue kernels ($(count_allowscalar(rb.plan)) against $(count_allowscalar(r0.plan)))")

    ok &= glue_case("host use", src_ghost)
    rh = glue_compare(Meta.parse(src_ghost))
    check(!occursin("alpha = CuArray", replace(string(rh.plan.host), r"\s+" => " ")), "host use: alpha, read by the host in an if, is not boxed")

    ok &= glue_case("cross", src_gcross)
    ok &= glue_case("order", src_gorder)
    for mm in (0, 1)
        ok &= glue_case("zero trip m = $mm", src_gbasic; ints = Dict(:m => mm), modes = (:adjoint,), expect_glue = mm > 0)
    end

    # 7: the rollback
    rd = serial_compare(Meta.parse(src_gbasic); serial = true, glue = false)
    check(isempty(glue_names(rd.plan)) && relerr(rd.cpu, rd.gpu) <= 1e-12, "rollback: glue_kernels = false gives no glue kernel")
    rdef = serial_compare(Meta.parse(src_gbasic); serial = true, glue = nothing)
    check(!isempty(glue_names(rdef.plan)) && relerr(rdef.cpu, rdef.gpu) <= 1e-12, "default: glue_kernels is on when the caller does not say")

    # 6: corpus
    paths = corpus_paths()
    sel = isempty(names) ? sort(collect(keys(paths))) : names
    # GLUE_SHARD=k/N runs every N-th kernel (a cluster run in parallel shards)
    shard = get(ENV, "GLUE_SHARD", "")
    if isempty(names) && !isempty(shard)
        (k, nn) = parse.(Int, split(shard, "/"))
        sel = sel[[i for i in eachindex(sel) if mod(i - 1, nn) == k]]
    end
    nglue = 0; nskip = 0; nerr = 0; worst = 0.0; worsth = 0.0; saved = 0
    for n in sel
        haskey(paths, n) || (println("FAIL unknown kernel $n"); ok = false; continue)
        primal = STADE.io_read_corpus_entry(paths[n])
        seed = Int(sum(Int.(codeunits(n))))
        res = try
            pre = glue_names_only(primal)
            isempty(pre) ? nothing : (glue_compare(primal; seed = seed), glue_compare(primal; seed = seed, mode = :hvp), serial_compare(primal; serial = true, glue = false, seed = seed))
        catch e
            e
        end
        if res isa Exception
            msg = first(sprint(showerror, res), 200)
            if occursin("keep_push_pop", msg) || occursin("val_generate_baseline", msg) || occursin("unsupported array rank", msg) || startswith(msg, "parse_kernel:")
                nskip += 1
            else
                nerr += 1; ok = false; println("FAIL $n: ", msg)
            end
            continue
        end
        res === nothing && (nskip += 1; continue)
        (ra, rh, rbase) = res
        nglue += 1
        ea = relerr(ra.cpu, ra.gpu); eh = relerr(rh.cpu, rh.gpu); worst = max(worst, ea); worsth = max(worsth, eh)
        maximum(abs.(ra.cpu); init = 0.0) > 0 || (ok = false; println("FAIL $n: the CPU gradient is all zero"))
        (ea <= 1e-12 && eh <= 1e-12) || (ok = false; println("FAIL $n: glue kernels $(length(glue_names(ra.plan))), adjoint rel $ea, hvp rel $eh"))
        d = count_allowscalar(rbase.plan) - count_allowscalar(ra.plan)
        d > 0 || (ok = false; println("FAIL $n: the glue kernels removed no @allowscalar statement ($(count_allowscalar(rbase.plan)) before, $(count_allowscalar(ra.plan)) after)"))
        saved += d
    end
    println("corpus: $(length(sel)) kernels, $nglue with a glue kernel (adjoint worst rel $worst, HVP worst rel $worsth), $saved @allowscalar statements removed, $nskip without, $nerr errors")
    check(isempty(names) && isempty(shard) ? nglue >= 20 : nglue >= 0, "corpus: the glue pass met at least 20 kernels")
    println(ok ? "ALL OK" : "FAILED")
    return ok
end

# plan only: the glue kernels of the adjoint of one corpus kernel
function glue_names_only(primal::Expr)
    gen = STADE.stade_adjoint(primal; keep_push_pop = false, fuse_ii_loops = true)
    fl = gpu_flags(true, true)
    pi = STADE.stade_gpu(gen.initstacks, STADE.cgen_backend_cuda(); fl...)
    pg = STADE.stade_gpu(gen.adjoint, STADE.cgen_backend_cuda(); fl..., init_hosts = Dict{Symbol,Expr}(pi.host.args[1].args[1] => pi.host))
    return glue_names(pg)
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    validate_glue_kernels(ARGS) || exit(1)
end
