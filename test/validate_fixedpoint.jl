# Regression for v0.5.37: agen_adjoint_fixedpoint, the fixed-point reverse of a counted loop with a guard (`for kk = 1:maxit`, one `if rn > tol`).
# For each kernel and three regimes:
#   converged  the loop stops on its tolerance: the fixed-point adjoint (every shadow and returned scalar) equals the full adjoint of the unrolled loop to
#              1e-9, a central difference of the primal (the tolerance held fixed) to 1e-5, the sweeps ran (at least one), neither flag is set, and the state
#              arrays it leaves hold the primal result (the unrolled reverse rolls them back to their entry values)
#   zero       the guard is closed before the first iteration: no sweep, the two adjoints equal
#   cap        maxit = 3: the forward flag is set, no sweep runs and the adjoint is the unrolled one (equal to stade_adjoint with keep_push_pop = false)
# and the unsettled case (fp_slow_linear: the loop converges, the reverse sweeps cannot settle: the adjoint raises and fp_info[5] is set), the refusal rule
# (a whitelist of the features the validated kernels show) and the refusals: two loops, a counter read in the body, an accumulation (no state adjoint consumed).
#   julia validate_fixedpoint.jl [kernel names] [--sabotage]
isdefined(@__MODULE__, :STADE) || include(joinpath(dirname(@__DIR__), "src", "STADE.jl"))
isdefined(@__MODULE__, :sab_read) || include(joinpath(@__DIR__, "sabotage_run.jl"))
using Random, LinearAlgebra
isdefined(@__MODULE__, :sg_build_call) || include(joinpath(@__DIR__, "validate_adjoint_split_gpu.jl"))

# the multigrid kernel keeps a zero ghost layer on every level; a random draw has none, and its residual then stagnates and never closes the guard
function fpv_zero_ghosts!(vals, sizes::Dict{Char,Int})
    for (a, v) in vals
        v isa AbstractMatrix || continue
        e = sizes[last(string(a))]
        v[1, :] .= 0.0; v[:, 1] .= 0.0; v[e, :] .= 0.0; v[:, e] .= 0.0
    end
    return nothing
end

# (dir, kernel, integer pins, float overrides, array start size, preparation of the drawn values or nothing)
fpv_cases() = Any[
    ("fp-corpus", "fp_newton_sqrt", Dict(:maxit => 80), Dict(:a => 3.0, :x0 => 1.5, :tol => 1e-30), 4, nothing),
    ("fp-corpus", "fp_scalar_cos", Dict(:maxit => 500), Dict(:c => 0.6, :tol => 1e-30), 4, nothing),
    ("fp-corpus", "fp_array_tanh", Dict(:n => 6, :maxit => 800), Dict(:c => 0.3, :tol => 1e-30), 8, nothing),
    ("fp-corpus", "fp_gauss_seidel", Dict(:n => 6, :maxit => 800), Dict(:tol => 1e-30), 8, nothing),
    ("fp-corpus", "fp_mixed_state", Dict(:n => 5, :maxit => 800), Dict(:a => 0.5, :tol => 1e-30), 6, nothing),
    ("counted-corpus", "phc_jacobi2d", Dict(:nint => 6, :maxit => 2000), Dict(:tol => 1e-28, :h2 => 0.02), 8, nothing),
    ("counted-corpus", "phc_gs1d", Dict(:n => 8, :maxit => 2000), Dict(:tol => 1e-28, :h2 => 0.05), 10, nothing),
    ("host-box-corpus", "hb_gs1d", Dict(:n => 8, :maxit => 2000), Dict(:tol => 1e-28, :h2 => 0.05), 10, nothing),
    ("inactive-read-corpus", "ir_mg2d_solve_3", Dict(:n1 => 7, :nu1 => 2, :nu2 => 2, :maxit => 200), Dict(:tol => 1e-24, :h1 => 0.125, :w => 0.8), 9, v -> fpv_zero_ghosts!(v, Dict('1' => 9, '2' => 5, '3' => 3))),
]

# central difference of the seeded primal in a random direction that leaves the tolerance alone, against the adjoint's gradient
function fpv_fd(kernel, primal, adj_out, int_args, vals, stack_names, tol_sym; trials::Int = 3, eps::Float64 = 1e-6)
    obs_fn = STADE.val_compile(STADE.val_primal_observing_expr(kernel, primal))
    adjoint_fn = STADE.val_compile(adj_out.adjoint); init_fn = STADE.val_compile(adj_out.initstacks)
    seed = STADE.val_random_values_like(kernel, vals); seed_flat = STADE.val_flatten(kernel, seed)
    x0 = STADE.val_flatten(kernel, vals)
    f = x -> sum(STADE.val_call_primal_observed(obs_fn, kernel, int_args, STADE.val_unflatten(kernel, int_args, vals, x)) .* seed_flat)
    g = STADE.val_call_adjoint(adjoint_fn, init_fn, kernel, int_args, deepcopy(vals), seed; stack_arg_names = stack_names)
    mask = STADE.val_zeros_like(kernel, vals); mask[tol_sym] = 1.0
    mi = findfirst(!=(0.0), STADE.val_flatten(kernel, mask))
    worst = 0.0
    for _ in 1:trials
        d = STADE.val_randn(length(x0)); d[mi] = 0.0; d ./= sqrt(sum(d .^ 2))
        fd = (f(x0 .+ eps .* d) - f(x0 .- eps .* d)) / (2 * eps)
        jv = sum(g .* d)
        worst = max(worst, abs(fd - jv) / max(abs(fd), abs(jv), 1e-12))
    end
    return worst
end

# one kernel in one regime: (err against the unrolled adjoint, fp_info, elements changed, FD error or NaN, the check's verdict string or "")
function fpv_run(dir, name, pins, floats, gs, prep, regime::Symbol; fd::Bool = false)
    primal = STADE.io_read_corpus_entry(joinpath(@__DIR__, dir, name * ".jl")); kernel = STADE.parse_kernel(primal)
    pins2 = copy(pins); floats2 = copy(floats)
    regime === :cap && (pins2[:maxit] = 3)
    Random.seed!(20261122)
    bl = STADE.val_generate_baseline(kernel, primal; self_check = false, int_pin = pins2, int_lo = 3, int_hi = 5, grow_start = gs, grow_max = 64)
    STADE.val_coerce_int_arrays!(kernel, bl.values)
    int_args = merge(bl.int_args, pins2); vals = bl.values
    for (k, v) in floats2; vals[k] = v; end
    prep === nothing || prep(vals)
    regime === :zero && (vals[:tol] = 1e30)
    seed = STADE.val_random_values_like(kernel, vals)
    full = STADE.stade_adjoint(primal; keep_push_pop = false)
    fp = STADE.agen_adjoint_fixedpoint(primal)
    fi = STADE.val_compile(full.initstacks); fg = STADE.val_compile(full.adjoint)
    pi_ = STADE.val_compile(sg_rename(fp.initstacks, :fpv_init)); pg = STADE.val_compile(sg_rename(fp.adjoint, :fpv_adj))
    names_full = Symbol[a for a in full.initstacks.args[1].args[2:end]]
    n_ret = count(a -> kernel.sig.kinds[a] == :scalar_float, kernel.sig.args)
    c1 = sg_cpu_run(kernel.sig, int_args, vals, seed, fi, fg, names_full)
    a2, t2 = sg_build_call(kernel.sig, int_args, vals, seed, identity)
    s2 = sg_init_stacks(pi_, sg_stack_extra(names_full, int_args, vals, identity))
    r2 = Base.invokelatest(pg, a2..., s2...)
    # the adjoints (every shadow, every returned scalar); the primal arrays are not compared: the unrolled reverse rolls them back to the entry values, the
    # fixed-point reverse leaves the state at the fixed point (checked against the primal below)
    m = 0.0; changed = 0
    for (a, ct) in c1.tracked
        ct.kind == :array_float || continue
        m = max(m, sg_relerr(ct.sh, t2[a].sh)); changed += count(!=(0.0), ct.sh .- c1.pre[a].sh)
    end
    m = max(m, sg_scalar_err(n_ret, c1.ret, r2))
    # the state arrays the adjoint leaves are the primal result of the same loop
    pf = STADE.val_compile(primal)
    pargs = STADE.val_primal_call_args(kernel.sig, int_args, Dict{Symbol,Any}(k => deepcopy(v) for (k, v) in vals))
    Base.invokelatest(pf, pargs...)
    vleft = 0.0
    for (i, a) in enumerate(kernel.sig.args)
        kernel.sig.kinds[a] == :array_float && a in fp.loop.arrays && (vleft = max(vleft, sg_relerr(pargs[i], t2[a].val)))
    end
    info = Float64.(s2[end])
    verdict = try STADE.agen_adjoint_fixedpoint_check(info); "" catch e; first(split(sprint(showerror, e), "\n")) end
    fderr = fd ? fpv_fd(kernel, primal, fp, int_args, vals, names_full, :tol) : NaN
    return (err = m, info = info, changed = changed, fderr = fderr, verdict = verdict, vleft = vleft)
end

# the same draw run by the device code of the fixed-point adjoint (`env` of validate_adjoint_split_gpu.jl): the worst relative error of every shadow and
# returned scalar against the CPU fixed-point adjoint, and the fp_info of both
function fpv_device(dir, name, pins, floats, gs, prep, env, regime::Symbol = :converged)
    primal = STADE.io_read_corpus_entry(joinpath(@__DIR__, dir, name * ".jl")); kernel = STADE.parse_kernel(primal)
    pins2 = copy(pins)
    regime === :cap && (pins2[:maxit] = 3)
    regime === :unsettled && (pins2[:maxit] = 14)
    Random.seed!(20261122)
    bl = STADE.val_generate_baseline(kernel, primal; self_check = false, int_pin = pins2, int_lo = 3, int_hi = 5, grow_start = gs, grow_max = 64)
    STADE.val_coerce_int_arrays!(kernel, bl.values)
    int_args = merge(bl.int_args, pins2); vals = bl.values
    for (k, v) in floats; vals[k] = v; end
    regime === :unsettled && for (k, v) in Dict(:a => 1.0, :x0 => 0.0, :omega => 0.1, :tol => 1e-3); vals[k] = v; end
    prep === nothing || prep(vals)
    regime === :zero && (vals[:tol] = 1e30)
    seed = STADE.val_random_values_like(kernel, vals)
    fp = STADE.agen_adjoint_fixedpoint(primal)
    cpu_init = STADE.val_compile(sg_rename(fp.initstacks, :fpv_init_cpu)); cpu_adj = STADE.val_compile(sg_rename(fp.adjoint, :fpv_adj_cpu))
    snames = Symbol[a for a in fp.initstacks.args[1].args[2:end]]
    a1, t1 = sg_build_call(kernel.sig, int_args, vals, seed, identity)
    s1 = sg_init_stacks(cpu_init, sg_stack_extra(snames, int_args, vals, identity))
    # the adjoint raises when the sweeps do not settle (on the CPU and on the device): the returned scalars are not compared then
    raised = false
    r1 = try Base.invokelatest(cpu_adj, a1..., s1...) catch e; raised = true; nothing end
    plan_init = env.plan(fp.initstacks, nothing)
    plan_adj = env.plan(fp.adjoint, env.initmap(plan_init, fp.initstacks))
    dev_init = sg_eval_plan!(env.mod, plan_init); dev_adj = sg_eval_plan!(env.mod, plan_adj)
    a2, t2 = sg_build_call(kernel.sig, int_args, vals, seed, env.wrap)
    s2 = sg_init_stacks(dev_init, sg_stack_extra(snames, int_args, vals, env.wrap))
    devraised = false
    r2 = try Base.invokelatest(dev_adj, a2..., s2...) catch e; devraised = true; first(sprint(showerror, e), 160) end
    err = 0.0
    for (a, ct) in t1
        ct.kind == :array_float || continue
        err = max(err, sg_relerr(ct.sh, env.tohost(t2[a].sh)))
    end
    raised || devraised || (err = max(err, sg_scalar_err(count(a -> kernel.sig.kinds[a] == :scalar_float, kernel.sig.args), r1, r2)))
    return (err = err, cpu_info = Float64.(s1[end]), dev_info = Float64.(env.tohost(s2[end])), raised = raised, devraised = devraised, devmsg = devraised ? r2 : "")
end

# the converged loop whose reverse cannot settle in as many sweeps as the loop has iterations: what the adjoint raises and what it leaves in fp_info
function fpv_unsettled()
    primal = STADE.io_read_corpus_entry(joinpath(@__DIR__, "fp-corpus", "fp_slow_linear.jl")); kernel = STADE.parse_kernel(primal)
    pins = Dict{Symbol,Int}(:maxit => 14)
    Random.seed!(20261122)
    bl = STADE.val_generate_baseline(kernel, primal; self_check = false, int_pin = pins, int_lo = 3, int_hi = 5, grow_start = 4, grow_max = 64)
    int_args = merge(bl.int_args, pins); vals = bl.values
    for (k, v) in Dict(:a => 1.0, :x0 => 0.0, :omega => 0.1, :tol => 1e-3); vals[k] = v; end
    seed = STADE.val_random_values_like(kernel, vals)
    fp = STADE.agen_adjoint_fixedpoint(primal)
    pi_ = STADE.val_compile(sg_rename(fp.initstacks, :fpv_init_slow)); pg = STADE.val_compile(sg_rename(fp.adjoint, :fpv_adj_slow))
    snames = Symbol[a for a in fp.initstacks.args[1].args[2:end]]
    a2, t2 = sg_build_call(kernel.sig, int_args, vals, seed, identity)
    s2 = sg_init_stacks(pi_, sg_stack_extra(snames, int_args, vals, identity))
    msg = try Base.invokelatest(pg, a2..., s2...); "" catch e; sprint(showerror, e) end
    return (msg = msg, info = Float64.(s2[end]))
end

function fpv_unit_checks(say)
    root = @__DIR__
    for (nm, frag) in (("fp_two_loops", "has 2 counted-guard loops"), ("fp_counter_used", "has 0 counted-guard loops"), ("fp_accumulate", "never consumes a state adjoint"))
        primal = STADE.io_read_corpus_entry(joinpath(root, "fp-corpus", nm * ".jl"))
        msg = try STADE.agen_adjoint_fixedpoint(primal); "" catch e; sprint(showerror, e) end
        say(occursin(frag, msg), "$nm is refused with `$frag` (got: $(first(msg, 120)))")
    end
    # the state the detector reads
    want = Dict("fp_newton_sqrt" => (Symbol[], [:x]), "fp_scalar_cos" => (Symbol[], [:x]), "fp_array_tanh" => ([:u, :w], Symbol[]), "fp_gauss_seidel" => ([:u], Symbol[]), "fp_mixed_state" => ([:u], [:m]))
    for (nm, (ar, sc)) in want
        primal = STADE.io_read_corpus_entry(joinpath(root, "fp-corpus", nm * ".jl")); k = STADE.parse_kernel(primal)
        f = STADE.agen_fp_detect(primal, k.sig.kinds)
        say(length(f) == 1 && f[1].arrays == ar && f[1].scalars == sc && f[1].hi === :maxit && f[1].rn === :rn && f[1].tol === :tol, "$nm: detected state arrays=$ar scalars=$sc (got $(isempty(f) ? "none" : (f[1].arrays, f[1].scalars)))")
    end
    # the whitelist rule: every detected loop of the corpora, accepted or refused with a reason
    verdicts = (("fp-corpus", "fp_newton_sqrt", ""), ("fp-corpus", "fp_scalar_cos", ""), ("fp-corpus", "fp_array_tanh", ""), ("fp-corpus", "fp_gauss_seidel", ""), ("fp-corpus", "fp_mixed_state", ""),
                ("counted-corpus", "phc_jacobi2d", ""), ("counted-corpus", "phc_gs1d", ""), ("host-box-corpus", "hb_gs1d", ""), ("inactive-read-corpus", "ir_mg2d_solve_3", ""),
                ("counted-corpus", "phc_cg_laplace2d", "divides by a value computed from a reduction"), ("counted-corpus", "phc_power_iter", "divides by a value computed from a reduction"),
                ("counted-corpus", "phc_mg_solve", "holds an if"), ("counted-corpus", "phc_mg_solve", "deeper than 2"), ("inactive-read-corpus", "ur_norm", "tolerance is a literal"))
    for (dir, nm, frag) in verdicts
        primal = STADE.io_read_corpus_entry(joinpath(root, dir, nm * ".jl")); k = STADE.parse_kernel(primal)
        f = STADE.agen_fp_detect(primal, k.sig.kinds)
        why = length(f) == 1 ? STADE.agen_fp_refusal(f[1], STADE.agen_fp_stmts(primal.args[2])) : String["not detected"]
        say(isempty(frag) ? isempty(why) : any(w -> occursin(frag, w), why), "$nm: the rule " * (isempty(frag) ? "accepts" : "refuses with `$frag`") * " (got: $(isempty(why) ? "accepted" : join(why, " | ")))")
    end
    # a function the validated cases never call is a reason
    for (fn, ok) in ((:sqrt, false), (:tanh, true))
        sq = Meta.parse("function fpw(a, x0, tol, maxit, y)\n x = x0\n rn = 1.0\n for kk = 1:maxit\n if rn > tol\n xn = $fn(x) + a\n rn = (xn - x) * (xn - x)\n x = xn\n end\n end\n y[1] = x\n return nothing\nend")
        k = STADE.parse_kernel(sq); f = STADE.agen_fp_detect(sq, k.sig.kinds)
        why = length(f) == 1 ? STADE.agen_fp_refusal(f[1], STADE.agen_fp_stmts(sq.args[2])) : String["not detected"]
        say(isempty(why) == ok, "a loop that calls $fn: the rule " * (ok ? "accepts" : "refuses") * " (got: $(isempty(why) ? "accepted" : join(why, " | ")))")
    end
    # the verdict raises on each flag
    say(try STADE.agen_adjoint_fixedpoint_check([3.0, 2.0, 0.1, 1.0, 0.0]); false catch; true end, "check raises when the forward loop hit its cap")
    say(try STADE.agen_adjoint_fixedpoint_check([3.0, 2.0, 0.1, 0.0, 1.0]); false catch; true end, "check raises when the sweeps did not settle")
    say(STADE.agen_adjoint_fixedpoint_check([3.0, 2.0, 0.0, 0.0, 0.0]) == 2, "check returns the sweeps when nothing is flagged")
end

function validate_fixedpoint(; verbose = true, only = String[], sabotage = false)
    checks = 0; bad = 0
    say(ok, msg) = (checks += 1; ok || (bad += 1; verbose && println("NOT OK ", msg)))
    fpv_unit_checks(say)
    cap_diff = 0.0
    for (dir, name, pins, floats, gs, prep) in fpv_cases()
        isempty(only) || name in only || continue
        r = fpv_run(dir, name, pins, floats, gs, prep, :converged; fd = true)
        verbose && println(rpad(name, 22), " converged: err=", round(r.err, sigdigits = 3), " fd=", round(r.fderr, sigdigits = 3), " sweeps=", Int(r.info[2]), " last_iter=", Int(r.info[1]), " ratio=", round(r.info[3], sigdigits = 3), " changed=", r.changed)
        say(r.err <= 1e-9, "$name converged: fixed-point vs unrolled adjoint err=$(r.err)")
        say(r.fderr <= 1e-5, "$name converged: fixed-point adjoint vs central difference err=$(r.fderr)")
        say(r.info[2] >= 1 && r.info[1] >= 1 && r.info[4] == 0 && r.info[5] == 0 && r.verdict == "", "$name converged: sweeps ran, no flag ($(r.info)) $(r.verdict)")
        say(r.changed > 0, "$name converged: nothing changed (vacuous)")
        say(r.vleft <= 1e-6, "$name converged: the state the adjoint leaves is not the primal result ($(r.vleft))")
        z = fpv_run(dir, name, pins, floats, gs, prep, :zero)
        say(z.err <= 1e-12 && z.info[1] == 0 && z.info[2] == 0 && z.info[4] == 0 && z.info[5] == 0, "$name zero: err=$(z.err) info=$(z.info)")
        c = fpv_run(dir, name, pins, floats, gs, prep, :cap)
        # a loop that ran to its cap is not at a fixed point: the adjoint takes the unrolled reverse, every iteration, and raises nothing
        say(c.info[4] == 1 && c.info[2] == 0 && c.info[5] == 0, "$name cap: the forward flag is set and no sweep ran ($(c.info))")
        say(c.err <= 1e-12, "$name cap: the adjoint differs from the unrolled adjoint ($(c.err))")
        say(occursin("cap", c.verdict), "$name cap: the host verdict names the cap ($(c.verdict))")
        cap_diff = max(cap_diff, c.err)
    end
    u = fpv_unsettled()
    say(occursin("did not settle", u.msg), "fp_slow_linear: a converged loop whose reverse cannot settle raises (got: $(first(u.msg, 100)))")
    say(u.info[4] == 0 && u.info[5] == 1 && u.info[1] >= 1, "fp_slow_linear: the forward loop converged and the sweeps stopped on their cap ($(u.info))")
    verbose && println("validate_fixedpoint: $(checks - bad)/$checks ok")
    if sabotage
        text = read(joinpath(dirname(@__DIR__), "src", "STADE.jl"), String)
        cases = [
            ("the sweeps replay iteration 1", "                    \$kk = __fp_last\n", "                    \$kk = 1\n", ["fp_scalar_cos", "fp_array_tanh"]),
            ("the last active iteration is the first", "                    __fp_last = __fp_k\n", "                    __fp_last = 1\n", ["fp_scalar_cos", "fp_array_tanh"]),
            ("one sweep only", "                    if __fp_cn <= \$tolx * __fp_c0\n                        __fp_go = 0\n                    end\n", "                    __fp_go = 0\n", ["fp_scalar_cos", "fp_array_tanh"]),
            ("the forward flag is never set", "        if \$fwd_flag\n            __fp_cut = 1\n            fp_info[4] = 1.0\n        end\n", "", ["fp_scalar_cos"]),
            ("the adjoint flag is never set", "            fp_info[5] = __fp_go\n", "            fp_info[5] = 0.0\n", ["fp_scalar_cos"]),
            ("the counter may be read in the body", "        h.var in syms && continue\n", "", ["fp_scalar_cos"]),
            ("a second loop is accepted", "    length(found) == 1 || error(\"agen_adjoint_fixedpoint: `\$(kernel.sig.name)` has", "    length(found) >= 0 || error(\"agen_adjoint_fixedpoint: `\$(kernel.sig.name)` has", ["fp_scalar_cos"]),
            ("accumulation loops are accepted", "    count > 0 || error(\"agen_adjoint_fixedpoint: the reverse", "    count >= 0 || error(\"agen_adjoint_fixedpoint: the reverse", ["fp_scalar_cos"]),
            ("the branch is inverted", "        if __fp_cut == 0\n            __fp_last = 0\n", "        if __fp_cut == 1\n            __fp_last = 0\n", ["fp_scalar_cos"]),
            ("a cut-off loop has no unrolled reverse", "        else\n            \$rev\n        end\n", "        else\n        end\n", ["fp_scalar_cos"]),
            ("the sweep error is not raised", "        if __fp_go == 1\n            error(\$msg)\n        end\n", "", ["fp_scalar_cos"]),
            ("a division by a reduction is not a reason", "    f.div_by_reduction && push!(why,", "    false && push!(why,", ["fp_scalar_cos"]),
            ("an if in the body is not a reason", "    f.inner_if && push!(why,", "    false && push!(why,", ["fp_scalar_cos"]),
            ("a literal tolerance is not a reason", "    f.literal_tol && push!(why,", "    false && push!(why,", ["fp_scalar_cos"]),
            ("deep nesting is not a reason", "    f.depth > 2 && push!(why,", "    f.depth > 99 && push!(why,", ["fp_scalar_cos"]),
            ("every call is validated", "agen_fp_validated_calls() = Set{Symbol}([:div, :mod, :sin, :cos, :tanh])", "agen_fp_validated_calls() = Set{Symbol}([:div, :mod, :sin, :cos, :tanh, :sqrt, :abs])", ["fp_scalar_cos"]),
            ("a scalar assigned from a reduction is not one", "        isempty(intersect(rsyms, red)) || push!(red, e.args[1])\n", "", ["fp_scalar_cos"]),
            ("the first sweep's norm is not kept", "                    if __fp_sw == 0\n                        __fp_c0 = __fp_cn\n                    end\n", "                    __fp_c0 = __fp_cn\n", ["fp_scalar_cos", "fp_array_tanh"]),
        ]
        mktempdir() do tmp
            for (label, old, new, kset) in cases
                isempty(get(ENV, "FPV_SAB_ONLY", "")) || any(t -> occursin(t, label), split(ENV["FPV_SAB_ONLY"], "|")) || continue
                n = length(findall(old, text))
                if n != 1
                    bad += 1; verbose && println("NOT OK sabotage \"$label\": anchor found $n times"); continue
                end
                get(ENV, "SAB_ANCHORS_ONLY", "") == "1" && continue
                path = joinpath(tmp, "STADE_sab.jl"); write(path, replace(text, old => new; count = 1))
                script = """
                    include($(repr(path)))
                    cd($(repr(@__DIR__)))
                    include($(repr(joinpath(@__DIR__, "validate_fixedpoint.jl"))))
                    println("RESULT ", validate_fixedpoint(; verbose = false, only = $(repr(kset))) ? 0 : 1)
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
    exit(validate_fixedpoint(; only = String[a for a in ARGS if !startswith(a, "--")], sabotage = "--sabotage" in ARGS) ? 0 : 1)
end
