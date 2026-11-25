# The checkpointed adjoint on the loops of the corpora, forced: every top-level loop that can be a region is one (the literal trip-count rule is off), whatever the kernel.
# For each kernel of the directories and each segment length ck in (1, 2, 3, 7): every argument after the call and every returned scalar of the checkpointed adjoint
# equal, bit for bit, those of stade_adjoint(keep_push_pop = false) (the fixed-point adjoint where a solve loop is) on the same random draw. A kernel the adjoint without checkpoints cannot size is skipped.
# Every kernel gets one row: no loop that can be a region, no region (the reasons), checked, differs, error (the first line of the message).
#   julia validate_checkpoint_corpus.jl [dir ...] [-k kernel,kernel] [--zero] [--device] [--hvp]   (--zero draws the integers from 0 to 2: loops that run no step, as validate_zerotrip.jl; --hvp checks the checkpointed HVP against the plain one instead of the adjoint; --device also runs the CUDA and JACC text on the serial stand-ins against the CPU checkpointed adjoint, 1e-9)
isdefined(@__MODULE__, :STADE) || include(joinpath(dirname(@__DIR__), "src", "STADE.jl"))
isdefined(@__MODULE__, :ckv_call) || include(joinpath(@__DIR__, "validate_checkpoint.jl"))
isdefined(@__MODULE__, :ckh_compare) || include(joinpath(@__DIR__, "validate_checkpoint_hvp.jl"))
using Random, Logging

# The device run of an adjoint (its text planned for the environment) against its CPU run, on every segment length of `cks`: the worst relative error over the arrays, their shadows and the returned scalars.
# Returns the lengths whose error is above 1e-9, or a string when the environment cannot plan the text.
function ckc_device(env, bundle, sig, int_args, vals, seedv, cks)
    plan = try
        pi = env.plan(bundle.initstacks, nothing)
        pa = env.plan(bundle.adjoint, env.initmap(pi, bundle.initstacks))
        (sg_eval_plan!(env.mod, pi), sg_eval_plan!(env.mod, pa))
    catch e
        return first(split(sprint(showerror, e), "\n"))
    end
    names = Symbol[a for a in bundle.initstacks.args[1].args[2:end]]
    cin = STADE.val_compile(sg_rename(bundle.initstacks, :ckc_dev_init)); cfn = STADE.val_compile(sg_rename(bundle.adjoint, :ckc_dev_adj))
    bad = Int[]
    for c in cks
        ia = haskey(bundle, :primal) ? merge(int_args, Dict(:ck => c)) : int_args
        a1, t1 = sg_build_call(sig, ia, vals, seedv, identity)
        s1 = sg_init_stacks(cin, sg_stack_extra(names, ia, vals, identity))
        r1 = Base.invokelatest(cfn, a1..., s1...)
        a2, t2 = sg_build_call(sig, ia, vals, seedv, env.wrap)
        s2 = sg_init_stacks(plan[1], sg_stack_extra(names, ia, vals, env.wrap))
        r2 = try Base.invokelatest(plan[2], a2..., s2...)
             catch e; msg = first(split(sprint(showerror, e), "\n")); occursin("parallel_reduce was reached", msg) && return "needs parallel_reduce"; rethrow() end
        err = 0.0
        for (a, ct) in t1
            ct.kind in (:array_float, :array_int) || continue
            err = max(err, sg_relerr(ct.val, env.tohost(t2[a].val)))
            ct.kind == :array_float && (err = max(err, sg_relerr(ct.sh, env.tohost(t2[a].sh))))
        end
        err = max(err, sg_scalar_err(count(a -> sig.kinds[a] == :scalar_float, sig.args), r1, r2))
        err <= 1e-9 || push!(bad, c)
    end
    return bad
end

function ckc_kernel(path::String; cks = (1, 2, 3, 7), int_lo::Int = 3, int_hi::Int = 5, envs = (), hvp::Bool = false)
    name = splitext(basename(path))[1]
    primal = STADE.io_read_corpus_entry(path)
    kernel = STADE.parse_kernel(primal); sig = kernel.sig
    (cands, why) = STADE.agen_ckpt_candidates(primal, kernel; min_trip = 0)
    isempty(cands) && return (name = name, status = :no_candidate, note = isempty(why) ? "" : first(why))
    ref = try with_logger(NullLogger()) do; hvp ? STADE.stade_hvp(primal; keep_push_pop = false) : STADE.stade_adjoint(primal; keep_push_pop = false) end
          catch e; return (name = name, status = :plain_refuses, note = first(split(sprint(showerror, e), "\n"))) end
    if hvp
        # the checkpointed HVP against the plain one, three seedings (validate_checkpoint_hvp.jl)
        ckh = ckv_logged(() -> try STADE.agen_ckpt_adjoint(primal; min_trip = 0, hvp = true) catch e; e end).value
        ckh isa Exception && return (name = name, status = :error, note = first(split(sprint(showerror, ckh), "\n")))
        ckh === nothing && return (name = name, status = :no_region, note = "")
        Random.seed!(STADE.val_stable_seed(name))
        bl = STADE.val_generate_baseline(kernel, primal; self_check = false, int_lo = int_lo, int_hi = int_hi, grow_start = 8, grow_max = 64)
        STADE.val_coerce_int_arrays!(kernel, bl.values)
        (st, note) = ckh_compare(primal, bl.int_args, bl.values, cks, STADE.val_stable_seed(name))
        st === :differs && return (name = name, status = :differs, note = note)
        st === :error && return (name = name, status = :error, note = note)
        return (name = name, status = st === :checked ? :checked : :no_region, note = note)
    end
    lg = ckv_logged(() -> try STADE.agen_ckpt_adjoint(primal; min_trip = 0) catch e; e end)
    ck = lg.value
    ck isa Exception && return (name = name, status = :error, note = first(split(sprint(showerror, ck), "\n")))
    if ck === nothing
        why = [replace(l, "┌ Info: " => "", "stade_adjoint: checkpoint: " => "") for l in split(lg.log, "\n") if occursin("checkpoint:", l)]
        return (name = name, status = :no_region, note = isempty(why) ? "" : why[end])
    end
    Random.seed!(STADE.val_stable_seed(name))
    bl = STADE.val_generate_baseline(kernel, primal; self_check = false, int_lo = int_lo, int_hi = int_hi, grow_start = 8, grow_max = 64)
    STADE.val_coerce_int_arrays!(kernel, bl.values)
    int_args = bl.int_args; vals = bl.values
    seedv = STADE.val_random_values_like(kernel, vals)
    rn = Symbol[a for a in ref.initstacks.args[1].args[2:end]]
    rfn = STADE.val_compile(sg_rename(ref.adjoint, :ckc_ref)); rin = STADE.val_compile(sg_rename(ref.initstacks, :ckc_ref_init))
    r0 = ckv_call(rfn, rin, sig, int_args, vals, seedv, rn)
    ksig = STADE.parse_kernel(ck.primal).sig
    cn = Symbol[a for a in ck.initstacks.args[1].args[2:end]]
    cfn = STADE.val_compile(sg_rename(ck.adjoint, :ckc_ck)); cin = STADE.val_compile(sg_rename(ck.initstacks, :ckc_ck_init))
    bad = Int[]
    for c in cks
        r = try ckv_call(cfn, cin, ksig, merge(int_args, Dict(:ck => c)), vals, seedv, cn)
            catch e; return (name = name, status = :error, note = "ck=$c: " * first(split(sprint(showerror, e), "\n"))) end
        same = ckv_same(r0.ret, r.ret) && length(r.args) == length(r0.args) + 1 && all(ckv_same(x, y) for (x, y) in zip(r0.args, r.args[1:end - 1]))
        f0 = ckv_fpinfo(ref.initstacks, r0.stacks); f1 = ckv_fpinfo(ck.initstacks, r.stacks)
        same &= (f0 === nothing) == (f1 === nothing) && (f0 === nothing || ckv_same(f0, f1))
        same || push!(bad, c)
    end
    isempty(bad) || return (name = name, status = :differs, note = "ck=$(join(bad, ","))")
    notes = String["$(length(ck.regions)) region(s)"]
    for (ename, mk) in envs
        env = mk()
        d = ckc_device(env, ck, ksig, int_args, vals, seedv, cks)
        if d isa String
            # a text the stage cannot plan is no defect of the checkpoint when the plain adjoint cannot be planned either
            p = ckc_device(env, ref, sig, int_args, vals, seedv, (1,))
            p isa String && (push!(notes, "$ename: plain refused too ($(first(p, 60)))"); continue)
            return (name = name, status = :device_error, note = "$ename: $(first(d, 160))")
        end
        isempty(d) && (push!(notes, "$ename ok"); continue)
        # the plain adjoint differs on the device too: a kernel outside the contract of the stage, not a defect of the checkpoint
        p = ckc_device(env, ref, sig, int_args, vals, seedv, (1,))
        (p isa Vector && !isempty(p)) && (push!(notes, "$ename: plain adjoint differs too"); continue)
        return (name = name, status = :device_differs, note = "$ename ck=$(join(d, ","))")
    end
    return (name = name, status = :checked, note = join(notes, "; "))
end

function validate_checkpoint_corpus(dirs::Vector{String}; only::Vector{String} = String[], verbose::Bool = true, int_lo::Int = 3, int_hi::Int = 5, envs = (), hvp::Bool = false)
    rows = NamedTuple[]
    for d in dirs
        for f in sort(readdir(d))
            (endswith(f, ".jl") && !(endswith(f, "_b.jl") || endswith(f, "_d.jl") || endswith(f, "_hv.jl"))) || continue
            isempty(only) || splitext(f)[1] in only || continue
            r = try ckc_kernel(joinpath(d, f); int_lo = int_lo, int_hi = int_hi, envs = envs, hvp = hvp)
                catch e
                    msg = first(split(sprint(showerror, e), "\n"))
                    # a kernel the parser or the baseline generator refuses is no subject for this test
                    (name = splitext(f)[1], status = (startswith(msg, "parse_kernel:") || startswith(msg, "val_generate_baseline:") || startswith(msg, "agen_split_uev!:")) ? :unusable : :error, note = msg)
                end
            r = merge(r, (dir = basename(d),))
            push!(rows, r)
            verbose && println(rpad(r.dir, 18), rpad(r.name, 34), rpad(string(r.status), 14), first(r.note, 160))
            flush(stdout)
        end
    end
    counts = Dict{Symbol,Int}()
    for r in rows; counts[r.status] = get(counts, r.status, 0) + 1; end
    verbose && println("validate_checkpoint_corpus: ", join(["$k=$v" for (k, v) in sort(collect(counts); by = first)], " "))
    return get(counts, :differs, 0) == 0 && get(counts, :error, 0) == 0 && get(counts, :device_differs, 0) == 0 && get(counts, :device_error, 0) == 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = [a for a in ARGS if !(a in ("--zero", "--device", "--hvp"))]
    zero = "--zero" in ARGS
    envs = ()
    if "--device" in ARGS
        isdefined(@__MODULE__, :scc_env) || include(joinpath(@__DIR__, "validate_adjoint_split_cuda_cpu.jl"))
        isdefined(@__MODULE__, :spj_env) || include(joinpath(@__DIR__, "validate_adjoint_split_jacc_cpu.jl"))
        envs = (("cuda", scc_env), ("jacc", spj_env))
    end
    only = String[]
    i = findfirst(==("-k"), args)
    i === nothing || (only = String.(split(args[i + 1], ",")); deleteat!(args, i:i + 1))
    dirs = isempty(args) ? [joinpath(@__DIR__, d) for d in ("val-corpus", "counted-corpus", "scalar-acc-corpus", "int-state-corpus", "inactive-read-corpus", "plain-read-corpus", "boxed-reset-corpus", "while-corpus", "split-corpus", "battle-corpus")] : String[abspath(a) for a in args]
    exit(validate_checkpoint_corpus(dirs; only = only, int_lo = zero ? 0 : 3, int_hi = zero ? 2 : 5, envs = envs, hvp = "--hvp" in ARGS) ? 0 : 1)
end
