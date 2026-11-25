# The device code (CUDA and JACC) of the checkpointed adjoint, run on the CPU with the serial stand-ins of validate_adjoint_split_cuda_cpu.jl and validate_adjoint_split_jacc_cpu.jl.
# For each shape of ckpt_shapes.jl with a region, each number of steps and each segment length, the device adjoint equals the CPU checkpointed adjoint (every shadow, every returned
# scalar, to 1e-12) and reports the same fp_info; the checkpoint stacks are the ones of the converted initstacks.
#   julia validate_checkpoint_gpu_cpu.jl [shape names]
isdefined(@__MODULE__, :STADE) || include(joinpath(dirname(@__DIR__), "src", "STADE.jl"))
isdefined(@__MODULE__, :ckv_call) || include(joinpath(@__DIR__, "validate_checkpoint.jl"))
isdefined(@__MODULE__, :scc_env) || include(joinpath(@__DIR__, "validate_adjoint_split_cuda_cpu.jl"))
isdefined(@__MODULE__, :spj_env) || include(joinpath(@__DIR__, "validate_adjoint_split_jacc_cpu.jl"))
using Random, Logging

# the device run of one shape with nt steps: (worst relative error against the CPU over the segment lengths, the lengths that differ, a message)
function ckg_run(env, name::String, primal::Expr, nt::Int; cks = (1, 2, 3), seed::Int = 20261122)
    kernel = STADE.parse_kernel(primal)
    r = with_logger(NullLogger()) do; STADE.agen_ckpt_adjoint(primal; min_trip = 0) end
    r === nothing && return (:no_region, 0.0, Int[], "")
    opts = ckv_options(name)
    pins = Dict{Symbol,Int}()
    for a in kernel.sig.args
        kernel.sig.kinds[a] == :scalar_int || continue
        pins[a] = a === :n ? 6 : a === :nt2 ? nt + 2 : nt
    end
    merge!(pins, opts.pins)
    Random.seed!(seed)
    bl = STADE.val_generate_baseline(kernel, primal; self_check = false, int_pin = copy(pins), int_lo = 3, int_hi = 5, grow_start = 4, grow_max = 64)
    STADE.val_coerce_int_arrays!(kernel, bl.values)
    vals = bl.values
    for (k, v) in opts.floats; haskey(vals, k) && (vals[k] = v); end
    seedv = STADE.val_random_values_like(kernel, vals)
    ksig = STADE.parse_kernel(r.primal).sig
    snames = Symbol[a for a in r.initstacks.args[1].args[2:end]]
    cpu_init = STADE.val_compile(sg_rename(r.initstacks, :ckg_init_cpu)); cpu_adj = STADE.val_compile(sg_rename(r.adjoint, :ckg_adj_cpu))
    plan_init = env.plan(r.initstacks, nothing)
    plan_adj = env.plan(r.adjoint, env.initmap(plan_init, r.initstacks))
    dev_init = sg_eval_plan!(env.mod, plan_init); dev_adj = sg_eval_plan!(env.mod, plan_adj)
    worst = 0.0; bad = Int[]; msg = ""
    for c in cks
        int_args = merge(bl.int_args, pins, Dict(:ck => c))
        a1, t1 = sg_build_call(ksig, int_args, vals, seedv, identity)
        s1 = sg_init_stacks(cpu_init, sg_stack_extra(snames, int_args, vals, identity))
        r1 = Base.invokelatest(cpu_adj, a1..., s1...)
        a2, t2 = sg_build_call(ksig, int_args, vals, seedv, env.wrap)
        s2 = sg_init_stacks(dev_init, sg_stack_extra(snames, int_args, vals, env.wrap))
        r2 = try Base.invokelatest(dev_adj, a2..., s2...) catch e; msg = first(sprint(showerror, e), 200); nothing end
        if r2 === nothing && !isempty(msg); push!(bad, c); continue; end
        err = 0.0
        for (a, ct) in t1
            ct.kind == :array_float || continue
            err = max(err, sg_relerr(ct.sh, env.tohost(t2[a].sh)))
        end
        err = max(err, sg_scalar_err(count(a -> ksig.kinds[a] == :scalar_float, ksig.args), r1, r2))
        worst = max(worst, err)
        err <= 1e-12 || push!(bad, c)
    end
    # the guard of the segment length survives the ingest: a length below 1 raises on the device host function too
    for c in (0, -3)
        int_args = merge(bl.int_args, pins, Dict(:ck => c))
        raised = try Base.invokelatest(dev_init, sg_stack_extra(snames, int_args, vals, env.wrap)...); false
                 catch e; occursin("at least 1", sprint(showerror, e)) end
        raised || (push!(bad, c); msg = "ck = $c did not raise")
    end
    return (:checked, worst, bad, msg)
end

function validate_checkpoint_gpu_cpu(; verbose = true, only = String[], nsteps = (0, 1, 2, 10))
    checks = 0; bad = 0
    say(ok, msg) = (checks += 1; ok || (bad += 1; verbose && println("NOT OK ", msg)))
    for (ename, mk) in (("cuda", scc_env), ("jacc", spj_env))
        env = mk()
        for (name, src, expect) in ckp_shapes()
            expect === :region || continue
            isempty(only) || name in only || continue
            primal = Meta.parse(src)
            for nt in nsteps
                res = try ckg_run(env, name, primal, nt)
                      catch e
                          msg = first(split(sprint(showerror, e), "\n"))
                          if ename == "jacc" && occursin("parallel_reduce was reached", msg)
                              verbose && println(rpad(name, 18), " ", ename, " nt=", nt, ": needs parallel_reduce (skipped)")
                              continue
                          end
                          say(false, "$ename $name nt=$nt: $msg"); continue
                      end
                (st, err, badk, msg) = res
                st === :no_region && continue
                verbose && println(rpad(name, 18), " ", ename, " nt=", lpad(nt, 2), ": err=", round(err, sigdigits = 3), isempty(badk) ? "" : " differs at ck=$(badk) $msg")
                say(isempty(badk), "$ename $name nt=$nt: device checkpointed adjoint against the CPU one differs at ck=$badk ($msg)")
            end
        end
    end
    verbose && println("validate_checkpoint_gpu_cpu: $(checks - bad)/$checks ok")
    return bad == 0
end

# sabotage: each stage that puts the guard back must be needed (without it a segment length below 1 reaches the device unchecked)
function validate_checkpoint_gpu_cpu_sabotage(; verbose = true)
    text = read(joinpath(dirname(@__DIR__), "src", "STADE.jl"), String)
    bad = 0
    cases = (("the CUDA host function loses the guard", "host = cgen_guard_prepend(cgen_fp_raise_append(host, backend.allowscalar_macro), expr)", "host = cgen_fp_raise_append(host, backend.allowscalar_macro)"),
             ("the JACC host function loses the guard", "host = cgen_guard_prepend(cgen_fp_raise_append(host, jgen_default_allowscalar_macro()), expr)", "host = cgen_fp_raise_append(host, jgen_default_allowscalar_macro())"))
    mktempdir() do tmp
        for (label, old, new) in cases
            length(findall(old, text)) == 1 || (bad += 1; verbose && println("FAIL sabotage \"$label\": anchor not found once"); continue)
            path = joinpath(tmp, "STADE_sab.jl"); write(path, replace(text, old => new; count = 1))
            script = """
                include($(repr(path)))
                cd($(repr(@__DIR__)))
                include($(repr(joinpath(@__DIR__, "validate_checkpoint_gpu_cpu.jl"))))
                println("RESULT ", validate_checkpoint_gpu_cpu(; verbose = false, only = ["ck_array"]) ? 0 : 1)
            """
            out = try read(`$(Base.julia_cmd()) --startup-file=no -e $script`, String) catch; "ERR" end
            m = match(r"RESULT (\d)", out)
            caught = out == "ERR" || m === nothing || m.captures[1] == "1"
            caught || (bad += 1)
            verbose && println(caught ? "ok     " : "NOT OK ", "sabotage \"$label\": ", caught ? "a check fails" : "NOT CAUGHT")
        end
    end
    return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    only = String[a for a in ARGS if !startswith(a, "--")]
    ok = validate_checkpoint_gpu_cpu(; only)
    if "--sabotage" in ARGS
        ok = validate_checkpoint_gpu_cpu_sabotage() == 0 && ok
    end
    exit(ok ? 0 : 1)
end
