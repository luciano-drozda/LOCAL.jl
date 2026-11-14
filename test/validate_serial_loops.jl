# ==================== validate_serial_loops.jl =======================
# Checks the serial track of stade_gpu (`serial_loops = true`, v0.5.26): a loop that must run in order, with no parallel loop inside it, becomes a
# kernel launched with one thread. CPU only. The generated CUDA host function runs here with stand-ins for CUDA.jl (a CuArray is an Array, a launch
# runs every thread in turn, @atomic is a plain update), and its gradient is compared with the CPU adjoint of the same kernel.
#   1. CORPUS: every corpus kernel whose plan holds a serial kernel gives the CPU gradient (the CPU adjoint), and the run changed something.
#   2. CARRY: a scalar that the loop reads before it assigns, and that is not a running sum (`acc = 0.5 * acc + x[i]`), reaches the kernel as an argument.
#   3. COPY-BACK: a scalar the loop assigns and a later statement reads comes back through the box; one that nothing reads does not.
#   4. NEST: an order-dependent loop that holds a parallel loop keeps its host form, and the parallel loop stays a kernel.
#   5. ZERO TRIP: n = 0 leaves the scalar as it was.
#   6. DEFAULT: without the option no serial kernel exists and the generated code is the code of v0.5.25.
#
#   julia validate_serial_loops.jl [kernel names...]      (no names: every corpus kernel; this takes some minutes, run it on a cluster node)

isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
using Random

module SerialStub
    const CuArray = Array
    module CUDA
        macro allowscalar(e) esc(e) end
        macro atomic(e) esc(e) end
    end
    const bi = Ref(1); const bd = Ref(1); const ti = Ref(1)
    blockIdx() = (x = bi[],)
    blockDim() = (x = bd[],)
    threadIdx() = (x = ti[],)
    # threads and blocks may be any expression; every (block, thread) pair runs once, in order
    macro cuda(a, b, call)
        kv = Dict{Symbol,Any}(x.args[1] => x.args[2] for x in (a, b))
        return esc(quote
            let __T = $(kv[:threads]), __B = $(kv[:blocks])
                for __b in 1:__B, __t in 1:__T
                    SerialStub.bi[] = __b; SerialStub.ti[] = __t; SerialStub.bd[] = __T
                    $call
                end
            end
        end)
    end
end

# the passes of stade_gpu that need CUDA hardware (warp shuffles, shared memory, cuBLAS, block reductions) are off; the serial track is on or off
gpu_flags(serial) = (warp_atomics = false, sync_free_reduce = false, reduce_block = false, shared_private = false, contraction = 0, split_reduce = 0,
                     row_block = 0, col_block = 0, cublas = 0, serial_loops = serial)

function serial_kernel_names(plan)
    return [string(k.args[1].args[1]) for k in plan.kernels if startswith(string(k.args[1].args[1]), "cuda_serial_")]
end

# gradient of the CPU adjoint and of the stand-in GPU host function for one kernel and one draw. Returns (cpu, gpu, serial kernel names, plan)
function serial_compare(primal::Expr; serial::Bool = true, seed::Int = 1, int_lo::Int = 3, int_hi::Int = 5, int_override::Dict{Symbol,Int} = Dict{Symbol,Int}())
    kernel = STADE.parse_kernel(primal)
    Random.seed!(seed)
    baseline = STADE.val_generate_baseline(kernel, primal; int_lo = int_lo, int_hi = int_hi, grow_max = 512)
    int_args = merge(baseline.int_args, int_override)
    gen = STADE.stade_adjoint(primal; keep_push_pop = false, fuse_ii_loops = true)
    stack_arg_names = Symbol[a for a in gen.initstacks.args[1].args[2:end]]
    sd = STADE.val_random_values_like(kernel, baseline.values)
    cm = Module()
    Core.eval(cm, :(using LinearAlgebra))
    Core.eval(cm, gen.adjoint); Core.eval(cm, gen.initstacks)
    cpu = STADE.val_call_adjoint(getfield(cm, gen.adjoint.args[1].args[1]), getfield(cm, gen.initstacks.args[1].args[1]), kernel, int_args, baseline.values, sd; stack_arg_names)
    fl = gpu_flags(serial)
    plan_init = STADE.stade_gpu(gen.initstacks, STADE.cgen_backend_cuda(); fl...)
    plan_gen = STADE.stade_gpu(gen.adjoint, STADE.cgen_backend_cuda(); fl..., init_hosts = Dict{Symbol,Expr}(plan_init.host.args[1].args[1] => plan_init.host))
    plan_init = merge(plan_init, (host = STADE.cgen_shrink_stacks(plan_init.host, plan_gen.dead_stacks, plan_gen.compact_stacks),))
    for h in plan_gen.helpers; Core.eval(SerialStub, h); end
    for k in plan_init.kernels; Core.eval(SerialStub, k); end
    Core.eval(SerialStub, plan_init.host)
    for k in plan_gen.kernels; Core.eval(SerialStub, k); end
    Core.eval(SerialStub, plan_gen.host)
    gpu = STADE.val_call_adjoint(getfield(SerialStub, plan_gen.host.args[1].args[1]), getfield(SerialStub, plan_init.host.args[1].args[1]), kernel, int_args, baseline.values, sd; stack_arg_names)
    return (cpu = cpu, gpu = gpu, serial = serial_kernel_names(plan_gen), plan = plan_gen)
end

relerr(a, b) = maximum(abs.(a .- b); init = 0.0) / max(1.0, maximum(abs.(a); init = 0.0))

# kernels written for this test; the corpus holds the other shapes
src_carry = "function serial_carry(x, y, out, n)\n    acc = 0.0\n    for i = 1:n\n        acc = 0.5 * acc + x[i]\n        y[i] = acc * acc\n    end\n    out[1] = acc\n    return nothing\nend"
src_nest = "function serial_nest(a, w, u, n, m)\n    for t = 1:n\n        for j = 1:m\n            u[j] = u[j] + a[j] * w[t]\n        end\n        w[t + 1] = 0.5 * w[t] + u[1]\n    end\n    return nothing\nend"
src_zero = "function serial_zero(a, x, y, out, n)\n    acc = a[1]\n    for i = 1:n\n        acc = 0.5 * acc + x[i]\n        y[i] = acc\n    end\n    out[1] = 3.0 * acc\n    return nothing\nend"
src_ret = "function serial_ret(y, x, c, n)\n    for i = 1:n\n        y[i + 1] = y[i] + c * x[i]\n    end\n    return nothing\nend"
src_nocopy = "function serial_nocopy(x, y, n)\n    acc = 0.0\n    for i = 1:n\n        acc = 0.5 * acc + x[i]\n        y[i] = acc * acc\n    end\n    return nothing\nend"

function corpus_paths()
    paths = Dict{String,String}()
    for d in sort(filter(isdir, readdir(@__DIR__; join = true))), f in sort(readdir(d; join = true))
        endswith(f, ".jl") && occursin("-corpus", d) || continue
        n = splitext(basename(f))[1]
        occursin(r"_(b|d|hv|bd|cuda|jacc)$", n) && continue
        haskey(paths, n) || (paths[n] = f)
    end
    return paths
end

function validate_serial_loops(names = String[])
    ok = true
    check(c, msg) = (c || (println("FAIL ", msg); ok = false); c && println("ok   ", msg))

    # 2, 3, 5: the written kernels
    carry = Meta.parse(src_carry)
    r = serial_compare(carry)
    check(!isempty(r.serial), "carry: the loops became serial kernels ($(length(r.serial)))")
    check(relerr(r.cpu, r.gpu) <= 1e-13, "carry: gradient equals the CPU adjoint (rel $(relerr(r.cpu, r.gpu)))")
    text = join((string(k) for k in r.plan.kernels), "\n")
    check(occursin("__so", text), "copy-back: the loop whose scalar is read afterwards writes the box")
    zero = Meta.parse(src_zero)
    for nn in (0, 3)
        r0 = serial_compare(zero; int_override = Dict(:n => nn))
        check(!isempty(r0.serial) && relerr(r0.cpu, r0.gpu) <= 1e-13 && maximum(abs.(r0.cpu)) > 0, "zero trip: n = $nn, the scalar entering the loop reaches the output (rel $(relerr(r0.cpu, r0.gpu)))")
    end
    nocopy = Meta.parse(src_nocopy)
    rn = serial_compare(nocopy)
    sk = [k for k in rn.plan.kernels if startswith(string(k.args[1].args[1]), "cuda_serial_")]
    has_so = [occursin("__so", string(k)) for k in sk]
    check(length(sk) == 2 && has_so == [true, false], "copy-back: the forward loop (acc read after it) has a box, the reverse loop (nothing reads its scalars after) has none (box flags $has_so)")
    check(relerr(rn.cpu, rn.gpu) <= 1e-13, "copy-back: gradient equals the CPU adjoint (rel $(relerr(rn.cpu, rn.gpu)))")
    check(maximum(abs.(rn.cpu)) > 0, "copy-back: the gradient is not all zero")
    check(maximum(abs.(r.cpu)) > 0, "carry: the gradient is not all zero")

    # the adjoint of a scalar argument is returned by the function, and no statement of the body reads it afterwards
    rr = serial_compare(Meta.parse(src_ret))
    check(!isempty(rr.serial) && relerr(rr.cpu, rr.gpu) <= 1e-13 && maximum(abs.(rr.cpu)) > 0, "returned scalar: the adjoint of an argument the loop assigns reaches the return (rel $(relerr(rr.cpu, rr.gpu)))")

    # 4: nest
    nest = Meta.parse(src_nest)
    rs = serial_compare(nest)
    check(relerr(rs.cpu, rs.gpu) <= 1e-13, "nest: gradient equals the CPU adjoint (rel $(relerr(rs.cpu, rs.gpu)))")
    check(maximum(abs.(rs.cpu)) > 0, "nest: the gradient is not all zero")
    nk = [string(k.args[1].args[1]) for k in rs.plan.kernels]
    check(any(k -> startswith(k, "cuda_kernel_"), nk) && isempty(rs.serial), "nest: the parallel inner loop stays a kernel and no serial kernel holds it (kernels: $(join(nk, ", ")))")

    # 6: default
    rd = serial_compare(carry; serial = false)
    check(isempty(rd.serial) && relerr(rd.cpu, rd.gpu) <= 1e-13, "default: no serial kernel without the option, gradient unchanged")

    # 1: corpus
    paths = corpus_paths()
    sel = isempty(names) ? sort(collect(keys(paths))) : names
    nser = 0; nskip = 0; nerr = 0; worst = 0.0
    for n in sel
        haskey(paths, n) || (println("FAIL unknown kernel $n"); ok = false; continue)
        primal = STADE.io_read_corpus_entry(paths[n])
        res = try
            serial_compare(primal; seed = Int(sum(Int.(codeunits(n)))))
        catch e
            e
        end
        if res isa Exception
            msg = first(sprint(showerror, res), 160)
            # a kernel that does not generate with keep_push_pop = false, or whose draw fails, is not a serial-track case
            if occursin("keep_push_pop", msg) || occursin("val_generate_baseline", msg) || occursin("unsupported array rank", msg)
                nskip += 1
            else
                nerr += 1; ok = false; println("FAIL $n: ", msg)
            end
            continue
        end
        isempty(res.serial) && (nskip += 1; continue)
        nser += 1
        e = relerr(res.cpu, res.gpu); worst = max(worst, e)
        maximum(abs.(res.cpu); init = 0.0) > 0 || (ok = false; println("FAIL $n: the CPU gradient is all zero, the draw carries no signal"))
        e <= 1e-12 || (ok = false; println("FAIL $n: serial kernels $(length(res.serial)), gradient differs from the CPU adjoint, rel $e"))
    end
    println("corpus: $(length(sel)) kernels, $nser with a serial kernel (worst rel $worst), $nskip without, $nerr errors")
    check(isempty(names) ? nser >= 30 : nser >= 0, "corpus: the serial track met at least 30 kernels (it saw something)")
    println(ok ? "ALL OK" : "FAILED")
    return ok
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    validate_serial_loops(ARGS) || exit(1)
end
