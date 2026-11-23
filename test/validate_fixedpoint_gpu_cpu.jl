# Regression for v0.5.37: the device code (CUDA and JACC) of stade_adjoint_fixedpoint, run on the CPU with the serial stand-ins of
# validate_adjoint_split_cuda_cpu.jl and validate_adjoint_split_jacc_cpu.jl. For each kernel of validate_fixedpoint.jl and each regime, the device
# adjoint equals the CPU fixed-point adjoint (every shadow, every returned scalar) and reports the same fp_info.
#   The defect this guards: the GPU stage drops stores nothing in the function reads, and a fixed-point adjoint's `fp_info` is only written.
#   julia validate_fixedpoint_gpu_cpu.jl [kernel names] [--sabotage]
isdefined(@__MODULE__, :STADE) || include(joinpath(dirname(@__DIR__), "src", "STADE.jl"))
isdefined(@__MODULE__, :sab_read) || include(joinpath(@__DIR__, "sabotage_run.jl"))
isdefined(@__MODULE__, :fpv_cases) || include(joinpath(@__DIR__, "validate_fixedpoint.jl"))
isdefined(@__MODULE__, :scc_env) || include(joinpath(@__DIR__, "validate_adjoint_split_cuda_cpu.jl"))
isdefined(@__MODULE__, :spj_env) || include(joinpath(@__DIR__, "validate_adjoint_split_jacc_cpu.jl"))

function validate_fixedpoint_gpu_cpu(; verbose = true, only = String[], sabotage = false)
    checks = 0; bad = 0
    say(ok, msg) = (checks += 1; ok || (bad += 1; verbose && println("NOT OK ", msg)))
    for (ename, mk) in (("cuda", scc_env), ("jacc", spj_env))
        env = mk()
        for (dir, name, pins, floats, gs, prep) in fpv_cases()
            isempty(only) || name in only || continue
            for regime in (:converged, :zero, :cap)
                r = try fpv_device(dir, name, pins, floats, gs, prep, env, regime)
                    catch e
                        # the stand-in JACC has no block reduction: a loop long enough to launch one cannot run here (the V100 job covers it)
                        msg = first(split(sprint(showerror, e), "\n"))
                        if ename == "jacc" && occursin("parallel_reduce was reached", msg)
                            verbose && println(rpad(name, 22), " ", ename, " ", regime, ": needs parallel_reduce (skipped)")
                            continue
                        end
                        say(false, "$ename $name $regime: $msg"); continue
                    end
                verbose && println(rpad(name, 22), " ", ename, " ", regime, ": err=", round(r.err, sigdigits = 3), " info=", r.dev_info)
                say(r.err <= 1e-12 && !r.devraised && !r.raised, "$ename $name $regime: device vs CPU fixed-point adjoint err=$(r.err) (raised: $(r.raised)/$(r.devraised) $(r.devmsg))")
                say(r.dev_info == r.cpu_info || (r.dev_info[1:2] == r.cpu_info[1:2] && r.dev_info[4:5] == r.cpu_info[4:5] && isapprox(r.dev_info[3], r.cpu_info[3]; rtol = 1e-6)), "$ename $name $regime: fp_info device $(r.dev_info) vs CPU $(r.cpu_info)")
            end
        end
    end
    # a converged loop whose reverse cannot settle: the CPU adjoint and the device one raise, and fp_info[5] is set on both
    for (ename, mk) in (("cuda", scc_env), ("jacc", spj_env))
        isempty(only) || "fp_slow_linear" in only || continue
        r = try fpv_device("fp-corpus", "fp_slow_linear", Dict{Symbol,Int}(), Dict{Symbol,Float64}(), 4, nothing, mk(), :unsettled)
            catch e; say(false, "$ename fp_slow_linear unsettled: " * first(split(sprint(showerror, e), "\n"))); continue; end
        verbose && println(rpad("fp_slow_linear", 22), " ", ename, " unsettled: err=", round(r.err, sigdigits = 3), " raised=", r.raised, "/", r.devraised, " info=", r.dev_info)
        say(r.raised && r.devraised && occursin("did not settle", r.devmsg) && r.dev_info[4] == 0 && r.dev_info[5] == 1 && r.err <= 1e-12, "$ename fp_slow_linear unsettled: CPU raised ($(r.raised)), device raised ($(r.devraised): $(first(r.devmsg, 80))), fp_info[5] on the device ($(r.dev_info)), shadows agree ($(r.err))")
    end
    verbose && println("validate_fixedpoint_gpu_cpu: $(checks - bad)/$checks ok")
    if sabotage
        text = read(joinpath(dirname(@__DIR__), "src", "STADE.jl"), String)
        # No case for the CUDA dead-stack pass on a fixed-point adjoint: it deletes only the `*_stack` arrays an initstacks host allocates, fp_info is not one,
        # and the adjoint is one function holding both the record and the replay, so running the pass there changes nothing (an equivalent mutant, four kernels checked).
        cases = [
            ("the stage does not put the raise back", "    host = cgen_fp_raise_append(host, backend.allowscalar_macro)\n", "", ["fp_slow_linear"]),
            ("the JACC stage does not put the raise back", "    host = cgen_fp_raise_append(host, jgen_default_allowscalar_macro())\n", "", ["fp_slow_linear"]),
            ("the ingest keeps the raise of a fixed-point adjoint", " && !cgen_is_fp_raise(s)]", "]", ["fp_scalar_cos"]),
            ("the stage drops the fp_info stores (CUDA)", "dead_scalars && !(split_half in (:fwd, :fixedpoint)) && (expr = cgen_dead_pops(expr, init_hosts))", "dead_scalars && (expr = cgen_dead_pops(expr, init_hosts))", ["fp_scalar_cos"]),
            ("the fixed-point adjoint is not recognised", ":fp_info in expr.args[1].args[2:end] && return :fixedpoint", ":fp_infox in expr.args[1].args[2:end] && return :fixedpoint", ["fp_scalar_cos"]),
        ]
        mktempdir() do tmp
            for (label, old, new, kset) in cases
                isempty(get(ENV, "FPG_SAB_ONLY", "")) || any(t -> occursin(t, label), split(ENV["FPG_SAB_ONLY"], "|")) || continue
                n = length(findall(old, text))
                if n != 1
                    bad += 1; verbose && println("NOT OK sabotage \"$label\": anchor found $n times"); continue
                end
                get(ENV, "SAB_ANCHORS_ONLY", "") == "1" && continue
                path = joinpath(tmp, "STADE_sab.jl"); write(path, replace(text, old => new; count = 1))
                script = """
                    include($(repr(path)))
                    cd($(repr(@__DIR__)))
                    include($(repr(joinpath(@__DIR__, "validate_fixedpoint_gpu_cpu.jl"))))
                    println("RESULT ", validate_fixedpoint_gpu_cpu(; verbose = false, only = $(repr(kset))) ? 0 : 1)
                """
                out = sab_read(script)
                m = match(r"RESULT (\d)", out)
                caught = out == "ERR" || m === nothing || m.captures[1] == "1"
                caught || (bad += 1)
                verbose && println(caught ? "ok     " : "NOT OK ", "sabotage \"$label\": ", caught ? "a check fails" : "NOT CAUGHT")
            end
        end
    end
    return bad == 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_fixedpoint_gpu_cpu(; only = String[a for a in ARGS if !startswith(a, "--")], sabotage = "--sabotage" in ARGS) ? 0 : 1)
end
