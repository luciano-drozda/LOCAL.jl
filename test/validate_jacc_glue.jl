# ===================== validate_jacc_glue.jl ======================
# Checks the glue kernels of stade_jacc (`glue_kernels = true`, v0.5.26): the scalar statements between the kernels of a host loop run as
# one-thread JACC kernels (`jacc_glue_*!(__jacc_i, ...)`, launched `JACC.@parallel_for range = 1`) instead of `CUDA.@allowscalar` blocks.
# CPU only. The generated JACC host function runs with a stand-in JACC (a serial `for` loop per launch) and its gradient (adjoint) or Hv (HVP)
# is compared with the CPU one. The hardware passes (warp atomics, contraction, split reductions, block reductions) are off.
#   1. the hand cases of validate_glue_kernels.jl (basic, host use, cross, order, zero trip), glue on against the CPU result
#   2. the boxes are `JACC.array([0.0])`, the launches `JACC.@parallel_for range = 1`, and fewer @allowscalar statements remain
#   3. the rollback: `glue_kernels = false` (the default of stade_jacc) gives no glue kernel
#   4. CORPUS: every corpus kernel whose JACC plan holds a glue kernel gives the CPU gradient and Hv
#
#   julia validate_jacc_glue.jl [kernel names...]      (no names: every corpus kernel; GLUE_SHARD=k/N runs every N-th kernel)

include(joinpath(@__DIR__, "validate_glue_kernels.jl"))


# a kernel whose HVP launches take more than jgen_max_splat_args() = 32 free variables, so the JACC launch packs them in one tuple (`__jargs_*`), and whose
# loops accumulate into scalars that the planner boxes (alpha, beta and their adjoints). The tuple must hold the boxes: a box read as a value (`alphab[1]`)
# makes the kernel atomic-add into a Float64 (v0.5.26: InvalidIRError on a device, a MethodError on the stand-in).
src_jpack = raw"""
# sv_pcg2d(b, dg, x, r, z, p, Ap, rz, s, t, nx, ny, maxit, tol2)
#
# Conjugate gradient with a diagonal preconditioner on the 2D five-point Laplacian
# (nx by ny grid flattened row by row, zero border), at most maxit iterations.
#
# b: right-hand side of length nx * ny, zero on the border
# dg: diagonal of the preconditioner, length nx * ny, nonzero
# x: solution of length nx * ny, starts at zero
# r: residual work array of length nx * ny
# z: preconditioned residual work array of length nx * ny
# p: search direction work array of length nx * ny
# Ap: matrix-vector product work array of length nx * ny
# rz: length-1 array holding r.z
# s: length-1 array holding p.Ap
# t: length-1 array holding the new r.z
# nx: number of rows
# ny: number of columns
# maxit: iteration cap
# tol2: tolerance on r.z
function sv_pcg2d(b, dg, x, r, z, p, Ap, rz, s, t, nx, ny, maxit, tol2)
    ni = nx - 2
    nj = ny - 2
    rz[1] = 0.0
    for c = 1:nx * ny
        r[c] = b[c]
        z[c] = b[c] / dg[c]
        p[c] = z[c]
        x[c] = 0.0
        Ap[c] = 0.0
        rz[1] += b[c] * z[c]
    end
    for k = 1:maxit
        if rz[1] > tol2
            # matrix-vector product on the interior
            for idx = 1:ni * nj
                i = div(idx - 1, nj) + 2
                j = mod(idx - 1, nj) + 2
                c = (i - 1) * ny + j
                Ap[c] = 4.0 * p[c] - p[c - 1] - p[c + 1] - p[c - ny] - p[c + ny]
            end
            s[1] = 0.0
            for c = 1:nx * ny
                s[1] += p[c] * Ap[c]
            end
            alpha = rz[1] / s[1]
            t[1] = 0.0
            for c = 1:nx * ny
                x[c] += alpha * p[c]
                r[c] -= alpha * Ap[c]
                z[c] = r[c] / dg[c]
                t[1] += r[c] * z[c]
            end
            beta = t[1] / rz[1]
            for c = 1:nx * ny
                p[c] = z[c] + beta * p[c]
            end
            rz[1] = t[1]
        end
    end
    return nothing
end
"""

module JStub
    module JACC
        array(x) = copy(x)
        to_host(x) = copy(x)
        zeros(T, d...) = Base.zeros(T, d...)
        # `JACC.@parallel_for range = R [sync = false] f(args...)`: f(i, args...) for i in 1:R, in order
        macro parallel_for(a...)
            call = a[end]
            rng = nothing
            for x in a[1:end-1]
                x isa Expr && x.head === :(=) && x.args[1] === :range && (rng = x.args[2])
            end
            rng === nothing && error("stand-in parallel_for: no range")
            return esc(:(for __ji in 1:($rng); $(call.args[1])(__ji, $(call.args[2:end]...)); end))
        end
        # the large-n branch is behind a threshold the test never reaches
        macro parallel_reduce(a...) return :(error("stand-in JACC: parallel_reduce was reached")) end
    end
    module CUDA
        macro allowscalar(e) esc(e) end
    end
    module Atomix
        macro atomic(e) esc(e) end
    end
end

jacc_glue_names(plan) = [string(k.args[1].args[1]) for k in plan.kernels if startswith(string(k.args[1].args[1]), "jacc_glue_")]
jacc_flags(glue) = (warp_atomics = false, split_reduce = 0, jacc_contraction = 0, reduction_threshold = typemax(Int) ÷ 2, glue_kernels = glue)

# gradient of the CPU adjoint and of the stand-in JACC host function for one kernel and one draw. Returns (cpu, gpu, plan)
# the argument tuples of the packed launches (`__jargs_* = (...)`) in a host expression
function jacc_packed_tuples(e)
    out = Expr[]
    walk(x) = x isa Expr ? (x.head === :(=) && x.args[1] isa Symbol && startswith(String(x.args[1]), "__jargs_") && x.args[2] isa Expr && x.args[2].head === :tuple && push!(out, x.args[2]);
                             foreach(walk, x.args)) : nothing
    walk(e)
    return out
end

function jacc_compare(primal::Expr; glue::Bool = true, seed::Int = 1, int_lo::Int = 3, int_hi::Int = 5, int_override::Dict{Symbol,Int} = Dict{Symbol,Int}(), mode::Symbol = :adjoint)
    kernel = STADE.parse_kernel(primal)
    Random.seed!(seed)
    baseline = STADE.val_generate_baseline(kernel, primal; int_lo = int_lo, int_hi = int_hi, grow_max = 512)
    int_args = merge(baseline.int_args, int_override)
    gen = mode == :hvp ? STADE.stade_hvp(primal; keep_push_pop = false, fuse_ii_loops = true) : STADE.stade_adjoint(primal; keep_push_pop = false, fuse_ii_loops = true)
    cpu_expr = mode == :hvp ? gen.hvp : gen.adjoint
    stack_arg_names = Symbol[a for a in gen.initstacks.args[1].args[2:end]]
    sd = STADE.val_random_values_like(kernel, baseline.values)
    td = mode == :hvp ? STADE.val_random_values_like(kernel, baseline.values) : nothing
    tsd = mode == :hvp ? STADE.val_random_values_like(kernel, baseline.values) : nothing
    call(af, inf) = mode == :hvp ? STADE.val_call_hv(af, inf, kernel, int_args, baseline.values, sd, td, tsd; stack_arg_names) :
                                   STADE.val_call_adjoint(af, inf, kernel, int_args, baseline.values, sd; stack_arg_names)
    cm = Module()
    Core.eval(cm, :(using LinearAlgebra))
    Core.eval(cm, cpu_expr); Core.eval(cm, gen.initstacks)
    cpu = call(getfield(cm, cpu_expr.args[1].args[1]), getfield(cm, gen.initstacks.args[1].args[1]))
    fl = jacc_flags(glue)
    plan_init = STADE.stade_jacc(gen.initstacks; fl...)
    plan_gen = STADE.stade_jacc(cpu_expr; fl..., init_hosts = Dict{Symbol,Expr}(plan_init.host.args[1].args[1] => plan_init.host))
    for h in vcat(plan_init.helpers, plan_gen.helpers); Core.eval(JStub, h); end
    for k in plan_init.kernels; Core.eval(JStub, k); end
    Core.eval(JStub, plan_init.host)
    for k in plan_gen.kernels; Core.eval(JStub, k); end
    Core.eval(JStub, plan_gen.host)
    gpu = call(getfield(JStub, plan_gen.host.args[1].args[1]), getfield(JStub, plan_init.host.args[1].args[1]))
    return (cpu = cpu, gpu = gpu, plan = plan_gen)
end

function jacc_glue_case(label, src; ints = Dict{Symbol,Int}(), modes = (:adjoint, :hvp), expect_glue = true)
    ok = true
    for mode in modes
        r = jacc_compare(Meta.parse(src); mode, int_override = ints)
        gn = jacc_glue_names(r.plan)
        e = relerr(r.cpu, r.gpu)
        nz = maximum(abs.(r.cpu); init = 0.0) > 0
        good = e <= 1e-12 && nz && (!expect_glue || !isempty(gn))
        println(good ? "ok   " : "FAIL ", label, " [", mode, "]: glue kernels ", length(gn), ", rel ", e, ", nonzero ", nz)
        ok &= good
    end
    return ok
end

# plan only: the glue kernels of the JACC adjoint of one corpus kernel
function jacc_glue_names_only(primal::Expr)
    gen = STADE.stade_adjoint(primal; keep_push_pop = false, fuse_ii_loops = true)
    fl = jacc_flags(true)
    pi = STADE.stade_jacc(gen.initstacks; fl...)
    pg = STADE.stade_jacc(gen.adjoint; fl..., init_hosts = Dict{Symbol,Expr}(pi.host.args[1].args[1] => pi.host))
    return jacc_glue_names(pg)
end


# plan-level unit tests of the launch guards on a hand-written JACC host function and kernels (the kernel takes the thread index first)
function jacc_unit_tests()
    ok = true
    be = STADE.jgen_glue_backend()
    mk(host, ks...) = STADE.cgen_glue_apply(Meta.parse(host), Expr[Meta.parse(k) for k in ks], be; jacc = true)
    boxed(r, v) = occursin("$v = JACC.array", replace(string(r[1]), r"\s+" => " "))
    launch(args) = "JACC.@parallel_for range = 4 jacc_kernel_f_1!($args)"
    h1(second) = "function f_jacc(x, s, n)\n    CUDA.@allowscalar begin\n        alpha = 1.0 / (1.0 + s[1])\n    end\n    $(launch("alpha, x, n"))\n    $second\n    return nothing\nend"
    kern(body) = "function jacc_kernel_f_1!(__jacc_i, a, x, n)\n    $body\n    return nothing\nend"
    function check(c, msg); (c || (ok = false)); println(c ? "ok   " : "FAIL ", msg); end
    r = mk(h1(""), kern("x[1] = a * x[1]"))
    check(boxed(r, "alpha"), "jacc unit: a candidate that every site passes to a kernel is boxed")
    check(occursin("a[1]", string(r[2][1])), "jacc unit: the kernel reads the box through its own parameter (a[1])")
    check(!boxed(mk(h1(launch("0.5, x, n")), kern("x[1] = a * x[1]")), "alpha"), "jacc unit: another launch site passing a value keeps the scalar on the host")
    check(!boxed(mk(h1(""), kern("a = a + 1.0\n    x[1] = a * x[1]")), "alpha"), "jacc unit: a callee that assigns the parameter keeps the scalar on the host")
    h3 = "function f_jacc(x, s, n)\n    CUDA.@allowscalar begin\n        alpha = 1.0 / (1.0 + s[1])\n    end\n    JACC.@parallel_for range = 4 other_kernel!(alpha, x, n)\n    return nothing\nend"
    check(!boxed(mk(h3, kern("x[1] = a * x[1]")), "alpha"), "jacc unit: a launch of a function that is no kernel of this function keeps the scalar on the host")
    h4 = "function f_jacc(x, s, n)\n    CUDA.@allowscalar begin\n        alpha = 1.0 / (1.0 + s[1])\n    end\n    $(launch("alpha, x, n"))\n    y = alpha + 1.0\n    return nothing\nend"
    check(!boxed(mk(h4, kern("x[1] = a * x[1]")), "alpha"), "jacc unit: a host read of the scalar outside glue keeps it on the host")
    h5 = "function f_jacc(x, dst, n)\n    for e = 1:n\n        CUDA.@allowscalar begin\n            d = dst[e]\n        end\n        $(launch("d, x, n"))\n    end\n    return nothing\nend"
    r5 = mk(h5, kern("x[1] = a * x[1]"))
    check(!occursin("jacc_glue", string(r5[1])) && occursin("allowscalar", string(r5[1])), "jacc unit: a local passed to a kernel by value is not computed by a glue kernel")
    # the launch passes (x, alpha, n) to the kernel (__jacc_i, x, a, n): the candidate is the second launch argument and the third kernel parameter
    h6 = "function f_jacc(x, s, n)\n    CUDA.@allowscalar begin\n        alpha = 1.0 / (1.0 + s[1])\n    end\n    $(launch("x, alpha, n"))\n    return nothing\nend"
    k6 = "function jacc_kernel_f_1!(__jacc_i, x, a, n)\n    x[1] = a * x[1]\n    return nothing\nend"
    k6b = "function jacc_kernel_f_1!(__jacc_i, x, a, n)\n    a = 2.0\n    x[1] = a * x[1]\n    return nothing\nend"
    r6 = mk(h6, k6)
    check(boxed(r6, "alpha") && occursin("a[1]", string(r6[2][1])), "jacc unit: the position of a candidate is counted after the index parameter")
    check(!boxed(mk(h6, k6b), "alpha"), "jacc unit: a callee that assigns the parameter at that position keeps the scalar on the host")
    return ok
end

function validate_jacc_glue(names = String[])
    ok_unit = jacc_unit_tests()
    ok = ok_unit
    check(c, msg) = (c || (println("FAIL ", msg); ok = false); c && println("ok   ", msg))
    flat(x) = replace(string(x), r"\s+" => " ")

    ok &= jacc_glue_case("basic", src_gbasic)
    rb = jacc_compare(Meta.parse(src_gbasic))
    r0 = jacc_compare(Meta.parse(src_gbasic); glue = false)
    check(occursin("alpha = JACC.array([0.0])", flat(rb.plan.host)), "basic: alpha lives in a JACC device box")
    check(occursin("JACC.@parallel_for range = 1 jacc_glue_", flat(rb.plan.host)), "basic: the glue kernel launches with range = 1")
    check(count_allowscalar(rb.plan) < count_allowscalar(r0.plan), "basic: fewer @allowscalar statements than without glue kernels ($(count_allowscalar(rb.plan)) against $(count_allowscalar(r0.plan)))")
    # every glue kernel takes the thread index first
    check(all(k -> k.args[1].args[2] === :__jacc_i, [k for k in rb.plan.kernels if startswith(string(k.args[1].args[1]), "jacc_glue_")]), "basic: every glue kernel takes __jacc_i first")

    ok &= jacc_glue_case("host use", src_ghost)
    rh = jacc_compare(Meta.parse(src_ghost))
    check(!occursin("alpha = JACC.array", flat(rh.plan.host)), "host use: alpha, read by the host in an if, is not boxed")
    ok &= jacc_glue_case("cross", src_gcross)
    ok &= jacc_glue_case("order", src_gorder)
    for mm in (0, 1)
        ok &= jacc_glue_case("zero trip m = $mm", src_gbasic; ints = Dict(:m => mm), modes = (:adjoint,), expect_glue = mm > 0)
    end


    # packed launch with boxed scalars (HVP of a diagonally preconditioned CG: 35 free variables in one kernel), glue off and on
    for gl in (false, true)
        r = jacc_compare(Meta.parse(src_jpack); mode = :hvp, glue = gl)
        e = relerr(r.cpu, r.gpu)
        packed = !isempty(jacc_packed_tuples(r.plan.host))
        tups = jacc_packed_tuples(r.plan.host)
        ref_in_tuple = any(t -> any(x -> x isa Expr && x.head === :ref && length(x.args) == 2 && x.args[2] == 1, t.args), tups)
        check(packed, "packed launch (glue $(gl)): the plan has a packed argument tuple")
        check(!ref_in_tuple, "packed launch (glue $(gl)): no boxed scalar is read as a value inside a packed tuple")
        check(e <= 1e-12, "packed launch (glue $(gl)): the HVP equals the CPU one (rel $(e))")
    end

    # rollback and default
    check(isempty(jacc_glue_names(r0.plan)) && relerr(r0.cpu, r0.gpu) <= 1e-12, "rollback: glue_kernels = false gives no glue kernel")
    gen = STADE.stade_adjoint(Meta.parse(src_gbasic); keep_push_pop = false, fuse_ii_loops = true)
    pdef = STADE.stade_jacc(gen.adjoint; warp_atomics = false, split_reduce = 0, jacc_contraction = 0)
    check(isempty(jacc_glue_names(pdef)), "default: stade_jacc has no glue kernel until the caller asks")

    # corpus
    paths = corpus_paths()
    sel = isempty(names) ? sort(collect(keys(paths))) : names
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
            pre = jacc_glue_names_only(primal)
            isempty(pre) ? nothing : (jacc_compare(primal; seed = seed), jacc_compare(primal; seed = seed, mode = :hvp), jacc_compare(primal; glue = false, seed = seed))
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
        (ra, rh2, rbase) = res
        nglue += 1
        ea = relerr(ra.cpu, ra.gpu); eh = relerr(rh2.cpu, rh2.gpu); worst = max(worst, ea); worsth = max(worsth, eh)
        maximum(abs.(ra.cpu); init = 0.0) > 0 || (ok = false; println("FAIL $n: the CPU gradient is all zero"))
        (ea <= 1e-12 && eh <= 1e-12) || (ok = false; println("FAIL $n: glue kernels $(length(jacc_glue_names(ra.plan))), adjoint rel $ea, hvp rel $eh"))
        d = count_allowscalar(rbase.plan) - count_allowscalar(ra.plan)
        d > 0 || (ok = false; println("FAIL $n: the glue kernels removed no @allowscalar statement ($(count_allowscalar(rbase.plan)) before, $(count_allowscalar(ra.plan)) after)"))
        saved += d
    end
    println("corpus: $(length(sel)) kernels, $nglue with a JACC glue kernel (adjoint worst rel $worst, HVP worst rel $worsth), $saved @allowscalar statements removed, $nskip without, $nerr errors")
    check(isempty(names) && isempty(shard) ? nglue >= 20 : nglue >= 0, "corpus: the glue pass met at least 20 kernels")
    println(ok ? "ALL OK" : "FAILED")
    return ok
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    validate_jacc_glue(ARGS) || exit(1)
end
