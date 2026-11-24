# Tangent and HVP of kernels with a solve loop (a counted loop with a guard), in the converged regime.
# The tangent differentiates the iterations that ran, the adjoint and the HVP (stade_hvp, keep_push_pop = false) take the fixed-point path, so this suite checks
# that each is correct and that the tangent agrees with the fixed-point adjoint to the loop tolerance. It runs the six corpus kernels, the counted, host-box and inactive-read kernels
# of validate_fixedpoint.jl, and every generated shape of fp_shapes.jl. For each kernel and three random directions d (the tolerance is left alone):
#   tangent   J d against a central difference of the seeded primal, and <seed, J d> against <fixed-point adjoint gradient, d> (the dot-product identity, which holds
#             to the loop tolerance and not to rounding, because the tangent differentiates the iterations that ran)
#   HVP       H d (stade_hvp, keep_push_pop = false) against a central difference of the fixed-point adjoint gradient
#   vacuity   J d and H d are not zero for at least one direction (a check that compares zero with zero proves nothing)
# usage: julia validate_fixedpoint_tangent_hvp.jl [kernel ...] [--sabotage]
include(joinpath(@__DIR__, "validate_fixedpoint.jl"))
using LinearAlgebra

fpt_rel(a, b) = maximum(abs, a .- b; init = 0.0) / max(maximum(abs, a; init = 0.0), maximum(abs, b; init = 0.0), 1e-12)

# every case of validate_fixedpoint.jl as (name, primal Expr, pins, floats, array start size, preparation, projection of a direction or nothing)
function fpt_cases()
    out = Any[]
    for (dir, name, pins, floats, gs, prep) in fpv_cases()
        push!(out, (name, STADE.io_read_corpus_entry(joinpath(@__DIR__, dir, name * ".jl")), pins, floats, gs, prep, name == "ir_mg2d_solve_3" ? prep : nothing))
    end
    for (body, nest) in vcat(fpv_shape_cases(), fpv_shape_cases_host_only())
        sp = fpv_shape_params(body, nest)
        push!(out, ("fps_$(body)_$(nest)", sp.primal, sp.pins, sp.floats, sp.gs, sp.prep, nothing))
    end
    return out
end

# one kernel: the worst errors over three directions, and the sizes of J d and H d
function fpt_run(primal, pins, floats, gs, prep; dprep = nothing, seed::Int = 20261122, trials::Int = 3, eps_fd::Float64 = 1e-6, eps_hv::Float64 = 1e-5)
    kernel = STADE.parse_kernel(primal)
    Random.seed!(seed)
    bl = STADE.val_generate_baseline(kernel, primal; self_check = false, int_pin = copy(pins), int_lo = 3, int_hi = 5, grow_start = gs, grow_max = 64)
    STADE.val_coerce_int_arrays!(kernel, bl.values)
    int_args = merge(bl.int_args, pins); vals = bl.values
    for (k, v) in floats; vals[k] = v; end
    prep === nothing || prep(vals)
    seedv = STADE.val_random_values_like(kernel, vals); seed_flat = STADE.val_flatten(kernel, seedv)
    x0 = STADE.val_flatten(kernel, vals)
    unflat(x) = STADE.val_unflatten(kernel, int_args, vals, x)
    zeros_ = () -> STADE.val_zeros_like(kernel, vals)
    # the tolerance is not differentiated: its direction is zero
    mask = zeros_(); mask[:tol] = 1.0
    mi = findfirst(!=(0.0), STADE.val_flatten(kernel, mask))
    # primal, tangent, fixed-point adjoint, HVP
    obs = STADE.val_compile(STADE.val_primal_observing_expr(kernel, primal))
    f = x -> STADE.val_call_primal_observed(obs, kernel, int_args, unflat(x))
    tan = STADE.val_compile(fpv_logged(() -> STADE.stade_tangent(primal)).value)
    fp = STADE.agen_adjoint_fixedpoint(primal)
    names_fp = Symbol[a for a in fp.initstacks.args[1].args[2:end]]
    # the HVP's initstacks has the name of the fixed-point one: rename before compiling
    afn = STADE.val_compile(sg_rename(fp.adjoint, :fpt_adj)); ifn = STADE.val_compile(sg_rename(fp.initstacks, :fpt_init))
    grad = x -> STADE.val_call_adjoint(afn, ifn, kernel, int_args, unflat(x), seedv; stack_arg_names = names_fp)
    hv = fpv_logged(() -> STADE.stade_hvp(primal; keep_push_pop = false)).value
    hfn = STADE.val_compile(sg_rename(hv.hvp, :fpt_hvp)); hin = STADE.val_compile(sg_rename(hv.initstacks, :fpt_hvinit))
    hnames = Symbol[a for a in hv.initstacks.args[1].args[2:end]]
    jvp = d -> STADE.val_call_tangent(tan, kernel, int_args, unflat(x0), unflat(d))
    hvd = d -> STADE.val_call_hv(hfn, hin, kernel, int_args, unflat(x0), seedv, unflat(d), zeros_(); stack_arg_names = hnames)
    r = (tan_fd = 0.0, dot = 0.0, hvp_fd = 0.0, jsize = 0.0, hsize = 0.0)
    for _ in 1:trials
        d = STADE.val_randn(length(x0)); d[mi] = 0.0
        # a direction that keeps the structure the case needs (the zero ghost cells of the multigrid state), or the perturbed solve never converges
        dprep === nothing || (D = unflat(d); dprep(D); d = STADE.val_flatten(kernel, D); d[mi] = 0.0)
        d ./= sqrt(sum(d .^ 2))
        jd = jvp(d)
        fd = (f(x0 .+ eps_fd .* d) .- f(x0 .- eps_fd .* d)) ./ (2 * eps_fd)
        s1 = sum(seed_flat .* jd); s2 = sum(grad(x0) .* d)
        hd = hvd(d)
        hfd = (grad(x0 .+ eps_hv .* d) .- grad(x0 .- eps_hv .* d)) ./ (2 * eps_hv)
        hd[mi] = 0.0; hfd[mi] = 0.0
        r = (tan_fd = max(r.tan_fd, fpt_rel(jd, fd)), dot = max(r.dot, abs(s1 - s2) / max(abs(s1), abs(s2), 1e-12)),
             hvp_fd = max(r.hvp_fd, fpt_rel(hd, hfd)), jsize = max(r.jsize, maximum(abs, jd)), hsize = max(r.hsize, maximum(abs, hd)))
    end
    return r
end

function validate_fixedpoint_tangent_hvp(; verbose = true, only = String[], sabotage = false)
    checks = 0; bad = 0
    say(ok, msg) = (checks += 1; ok || (bad += 1; verbose && println("FAIL ", msg)))
    for (name, primal, pins, floats, gs, prep, dprep) in fpt_cases()
        isempty(only) || name in only || continue
        r = try fpt_run(primal, pins, floats, gs, prep; dprep = dprep) catch e; (error = first(split(sprint(showerror, e), "\n")),) end
        if hasproperty(r, :error)
            say(false, "$name: raised ($(r.error))"); continue
        end
        verbose && println(rpad(name, 26), " tan_fd=", round(r.tan_fd, sigdigits = 3), " dot=", round(r.dot, sigdigits = 3), " hvp_fd=", round(r.hvp_fd, sigdigits = 3),
                           " |Jd|=", round(r.jsize, sigdigits = 3), " |Hd|=", round(r.hsize, sigdigits = 3))
        say(r.jsize > 0, "$name: the tangent is zero in every direction (vacuous)")
        say(r.hsize > 0, "$name: the HVP is zero in every direction (vacuous)")
        say(r.tan_fd <= 1e-5, "$name: tangent against a central difference of the primal, err=$(r.tan_fd)")
        say(r.dot <= 1e-8, "$name: <seed, J d> against <fixed-point gradient, d>, err=$(r.dot)")
        say(r.hvp_fd <= 1e-5, "$name: HVP against a central difference of the fixed-point gradient, err=$(r.hvp_fd)")
    end
    verbose && println("validate_fixedpoint_tangent_hvp: $(checks - bad)/$checks ok")
    if sabotage
        text = read(joinpath(dirname(@__DIR__), "src", "STADE.jl"), String)
        cases = [
            ("the partial of cos has the wrong sign", "    return [der_neg(Expr(:call, :sin, args[1]))]\n", "    return [Expr(:call, :sin, args[1])]\n", ["fp_scalar_cos"]),
            ("the HVP leaves an assignment without its shadow", "        haskey(shadow_of, var) && push!(out, Expr(:(=), hvp_shadow_lvalue(lhs, shadow_of), hvp_tangent_expr(e.args[2], shadow_of)))\n        push!(out, e)\n        return out\n    elseif e.head == :for", "        push!(out, e)\n        return out\n    elseif e.head == :for", ["fp_scalar_cos", "fp_array_tanh"]),
            ("the fixed-point adjoint runs one sweep only", "                    if __fp_cn <= \$tolx * __fp_c0\n                        __fp_go = 0\n                    end\n                end\n            end\n            __fp_iters", "                    __fp_go = 0\n                end\n            end\n            __fp_iters", ["fp_scalar_cos"]),
        ]
        mktempdir() do tmp
            for (label, old, new, kset) in cases
                n = length(findall(old, text))
                if n != 1
                    bad += 1; verbose && println("FAIL sabotage \"$label\": anchor found $n times"); continue
                end
                path = joinpath(tmp, "STADE_sab.jl"); write(path, replace(text, old => new; count = 1))
                script = """
                    include($(repr(path)))
                    cd($(repr(@__DIR__)))
                    include($(repr(joinpath(@__DIR__, "validate_fixedpoint_tangent_hvp.jl"))))
                    println("RESULT ", validate_fixedpoint_tangent_hvp(; verbose = false, only = $(repr(kset))) ? 0 : 1)
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
    exit(validate_fixedpoint_tangent_hvp(; only = String[a for a in ARGS if !startswith(a, "--")], sabotage = "--sabotage" in ARGS) ? 0 : 1)
end
