# The device code (CUDA and JACC) of the fixed-point HVP (agen_hvp_fixedpoint), run on the CPU with the serial stand-ins of validate_adjoint_split_cuda_cpu.jl and
# validate_adjoint_split_jacc_cpu.jl. For each case of validate_fixedpoint.jl and each regime, the device HVP equals the CPU fixed-point HVP (every output, to 1e-9) and reports
# the same fp_info. In the cap regime both raise the cut-off message, in the zero and nocap regimes neither raises.
#   julia validate_fixedpoint_hvp_gpu_cpu.jl [kernel names]
isdefined(@__MODULE__, :STADE) || include(joinpath(dirname(@__DIR__), "src", "STADE.jl"))
isdefined(@__MODULE__, :fph_call) || include(joinpath(@__DIR__, "validate_fixedpoint_hvp.jl"))
isdefined(@__MODULE__, :scc_env) || include(joinpath(@__DIR__, "validate_adjoint_split_cuda_cpu.jl"))
isdefined(@__MODULE__, :spj_env) || include(joinpath(@__DIR__, "validate_adjoint_split_jacc_cpu.jl"))

# the device run of the case in one regime: (worst relative error against the CPU, cpu fp_info, device fp_info, raised on the CPU, raised on the device and its message)
function fphg_run(env, primal, pins, floats, gs, prep, dprep, regime::Symbol; seed::Int = 20261122)
    kernel = STADE.parse_kernel(primal); sig = kernel.sig
    Random.seed!(seed)
    bl = STADE.val_generate_baseline(kernel, primal; self_check = false, int_pin = copy(pins), int_lo = 3, int_hi = 5, grow_start = gs, grow_max = 64)
    STADE.val_coerce_int_arrays!(kernel, bl.values)
    int_args = merge(bl.int_args, pins); vals = bl.values
    regime === :cap && (int_args = merge(int_args, Dict(:maxit => 3)))
    regime === :nocap && (int_args = merge(int_args, Dict(:maxit => 0), haskey(int_args, :maxq) ? Dict(:maxq => 0) : Dict{Symbol,Int}()))
    for (k, v) in floats; vals[k] = v; end
    prep === nothing || prep(vals)
    regime === :zero && (vals[:tol] = 1e30)
    seedv = STADE.val_random_values_like(kernel, vals)
    dseed = STADE.val_random_values_like(kernel, vals)
    mask = STADE.val_zeros_like(kernel, vals); mask[:tol] = 1.0
    mi = findfirst(!=(0.0), STADE.val_flatten(kernel, mask))
    d = STADE.val_randn(length(STADE.val_flatten(kernel, vals))); d[mi] = 0.0
    D = STADE.val_unflatten(kernel, int_args, vals, d)
    dprep === nothing || dprep(D)
    fp = STADE.agen_hvp_fixedpoint(primal)
    snames = Symbol[a for a in fp.initstacks.args[1].args[2:end]]
    cpu_init = STADE.val_compile(sg_rename(fp.initstacks, :fphg_init_cpu)); cpu_hvp = STADE.val_compile(sg_rename(fp.hvp, :fphg_hvp_cpu))
    c = fph_call(cpu_hvp, cpu_init, kernel, int_args, vals, seedv, D, dseed, snames)
    # the device plan of initstacks and of the HVP
    plan_init = env.plan(fp.initstacks, nothing)
    plan_hvp = env.plan(fp.hvp, env.initmap(plan_init, fp.initstacks))
    dev_init = sg_eval_plan!(env.mod, plan_init); dev_hvp = sg_eval_plan!(env.mod, plan_hvp)
    d = fph_call(dev_hvp, dev_init, kernel, int_args, vals, seedv, D, dseed, snames; wrap = env.wrap, tohost = env.tohost)
    err = isempty(d.raised) && isempty(c.raised) ? fpt_rel(c.out, d.out) : 0.0
    return (err = err, cpu_info = fph_info(c), dev_info = fph_info(d; tohost = env.tohost), raised = c.raised, devraised = d.raised)
end

function validate_fixedpoint_hvp_gpu_cpu(; verbose = true, only = String[])
    checks = 0; bad = 0
    say(ok, msg) = (checks += 1; ok || (bad += 1; verbose && println("NOT OK ", msg)))
    for (ename, mk) in (("cuda", scc_env), ("jacc", spj_env))
        env = mk()
        # the shapes that stay on the host (tier-B tables with an inline index do not ingest) are not device cases, as in validate_fixedpoint_gpu_cpu.jl
        host_only = Set{String}("fps_$(b)_$(n)" for (b, n) in fpv_shape_cases_host_only())
        for (name, primal, pins, floats, gs, prep, dprep) in fpt_cases()
            isempty(only) || name in only || continue
            name in host_only && continue
            for regime in (:converged, :zero, :cap, :nocap)
                r = try fphg_run(env, primal, pins, floats, gs, prep, dprep, regime)
                    catch e
                        msg = first(split(sprint(showerror, e), "\n"))
                        if ename == "jacc" && occursin("parallel_reduce was reached", msg)
                            verbose && println(rpad(name, 26), " ", ename, " ", regime, ": needs parallel_reduce (skipped)")
                            continue
                        end
                        say(false, "$ename $name $regime: $msg"); continue
                    end
                verbose && println(rpad(name, 26), " ", ename, " ", regime, ": err=", round(r.err, sigdigits = 3), " info=", r.dev_info)
                if regime === :cap
                    say(occursin("stopped on its cap", r.raised) && occursin("stopped on its cap", r.devraised), "$ename $name cap: CPU raised ($(first(r.raised, 60))), device raised ($(first(r.devraised, 60)))")
                    say(r.dev_info[4] >= 1 && r.dev_info[5] == 2, "$ename $name cap: device fp_info $(r.dev_info)")
                else
                    say(isempty(r.raised) && isempty(r.devraised), "$ename $name $regime: raised on the CPU ($(r.raised)) or on the device ($(r.devraised))")
                    say(r.err <= 1e-9, "$ename $name $regime: device HVP against the CPU fixed-point HVP, err=$(r.err)")
                    say(r.cpu_info[[1, 2, 4, 5]] == r.dev_info[[1, 2, 4, 5]] && abs(r.cpu_info[3] - r.dev_info[3]) <= 1e-6 * max(r.cpu_info[3], 1e-30), "$ename $name $regime: fp_info CPU $(r.cpu_info) device $(r.dev_info)")
                end
            end
        end
    end
    verbose && println("validate_fixedpoint_hvp_gpu_cpu: $(checks - bad)/$checks ok")
    return bad == 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_fixedpoint_hvp_gpu_cpu(; only = String[a for a in ARGS if !startswith(a, "--")]) ? 0 : 1)
end
