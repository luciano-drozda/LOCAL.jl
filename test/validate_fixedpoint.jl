# Regression for agen_adjoint_fixedpoint, the fixed-point reverse of a counted loop with a guard (`for kk = 1:maxit`, one `if rn > tol`), with the one-slot tape.
# For each kernel and four regimes:
#   converged  the loop stops on its tolerance: the fixed-point adjoint (every shadow and returned scalar) equals the full adjoint of the unrolled loop to
#              1e-9, a central difference of the primal (the tolerance held fixed) to 1e-5, the sweeps ran (at least one), no flag is set, the state arrays it
#              leaves hold the primal result (the unrolled reverse rolls them back to their entry values), the result equals that of the same adjoint
#              with the cap-sized tape (`tape = :full`) bit for bit, or to 1e-9 when a solve array was converted to the converged-state rule, and the tape holds the same number of elements for twice the cap
#   zero       the guard is closed before the first iteration: no sweep, the two adjoints equal
#   cap        maxit = 3: the loop stops on its cap, no sweep runs, fp_info[5] = 2 and the adjoint raises (there is no fixed point)
#   nocap      maxit = 0: the loop does not run, nothing is cut off, nothing raises
# and the unsettled case (fp_slow_linear: the loop converges, the reverse sweeps cannot settle: the adjoint raises and fp_info[5] = 1), the refusal rule
# (a whitelist of the features the validated kernels show) and the refusals: two loops, a counter read in the body, an accumulation (no state adjoint consumed),
# the cap read in the body.
#   julia validate_fixedpoint.jl [kernel names] [--sabotage]
isdefined(@__MODULE__, :STADE) || include(joinpath(dirname(@__DIR__), "src", "STADE.jl"))
isdefined(@__MODULE__, :sab_read) || include(joinpath(@__DIR__, "sabotage_run.jl"))
using Random, LinearAlgebra, Logging
isdefined(@__MODULE__, :fps_kernel) || include(joinpath(@__DIR__, "fp_shapes.jl"))
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
    ("fp-corpus", "fp_two_loops", Dict(:maxit => 500), Dict(:a => 0.6, :b => 0.5, :tol => 1e-30), 4, nothing),
    ("counted-corpus", "phc_jacobi2d", Dict(:nint => 6, :maxit => 2000), Dict(:tol => 1e-28, :h2 => 0.02), 8, nothing),
    ("counted-corpus", "phc_heat_steps", Dict(:n => 8, :nsteps => 3, :maxit => 2000), Dict(:tol => 1e-28, :lam => 0.4), 10, nothing),
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

# the tape of a fixed-point adjoint, in elements (every stack but the last, `fp_info`)
fpv_tape_elems(init_fn, names, int_args, vals) = sum(length, sg_init_stacks(init_fn, sg_stack_extra(names, int_args, vals, identity))[1:end-1])

# one kernel in one regime: (err against the unrolled adjoint, fp_info, elements changed, FD error or NaN, the check's verdict string or "", the state left against the
# primal, what the adjoint raised or "", the error against the cap-sized tape, the tape elements for the cap and for twice the cap)
function fpv_run(dir, name, pins, floats, gs, prep, regime::Symbol; fd::Bool = false)
    return fpv_run_expr(STADE.io_read_corpus_entry(joinpath(@__DIR__, dir, name * ".jl")), pins, floats, gs, prep, regime; fd = fd)
end

# the same for a kernel given as an Expr (a generated shape of fp_shapes.jl): the comparison with the cap-sized tape and the state left by the adjoint apply to
# a solve loop at the top level only
function fpv_run_expr(primal, pins, floats, gs, prep, regime::Symbol; fd::Bool = false, seed::Int = 20261122)
    kernel = STADE.parse_kernel(primal)
    dets = STADE.agen_fp_detect(primal, kernel.sig.kinds)
    nested = length(dets) > 1 || any(d -> !isempty(d.enclosing), dets)
    pins2 = copy(pins); floats2 = copy(floats)
    regime === :cap && (pins2[:maxit] = 3)
    Random.seed!(seed)
    bl = STADE.val_generate_baseline(kernel, primal; self_check = false, int_pin = pins2, int_lo = 3, int_hi = 5, grow_start = gs, grow_max = 64)
    STADE.val_coerce_int_arrays!(kernel, bl.values)
    int_args = merge(bl.int_args, pins2); vals = bl.values
    regime === :nocap && (int_args = merge(int_args, Dict(:maxit => 0), haskey(int_args, :maxq) ? Dict(:maxq => 0) : Dict{Symbol,Int}()))
    for (k, v) in floats2; vals[k] = v; end
    prep === nothing || prep(vals)
    regime === :zero && (vals[:tol] = 1e30)
    seed = STADE.val_random_values_like(kernel, vals)
    full = STADE.agen_adjoint_unrolled(primal; keep_push_pop = false)   # the unrolled reference: stade_adjoint itself takes the fixed-point path
    fp = STADE.agen_adjoint_fixedpoint(primal)
    fl = nested ? nothing : STADE.agen_adjoint_fixedpoint(primal; tape = :full)
    fi = STADE.val_compile(full.initstacks); fg = STADE.val_compile(full.adjoint)
    pi_ = STADE.val_compile(sg_rename(fp.initstacks, :fpv_init)); pg = STADE.val_compile(sg_rename(fp.adjoint, :fpv_adj))
    li = nested ? nothing : STADE.val_compile(sg_rename(fl.initstacks, :fpv_init_full)); lg = nested ? nothing : STADE.val_compile(sg_rename(fl.adjoint, :fpv_adj_full))
    names_full = Symbol[a for a in full.initstacks.args[1].args[2:end]]
    names_fp = Symbol[a for a in fp.initstacks.args[1].args[2:end]]   # the one-slot tape can need fewer sizes than the cap-sized one
    n_ret = count(a -> kernel.sig.kinds[a] == :scalar_float, kernel.sig.args)
    c1 = sg_cpu_run(kernel.sig, int_args, vals, seed, fi, fg, names_full)
    a2, t2 = sg_build_call(kernel.sig, int_args, vals, seed, identity)
    s2 = sg_init_stacks(pi_, sg_stack_extra(names_fp, int_args, vals, identity))
    raised = ""; r2 = nothing
    try r2 = Base.invokelatest(pg, a2..., s2...) catch e; raised = first(split(sprint(showerror, e), "\n")) end
    info = Float64.(s2[end])
    verdict = try STADE.agen_adjoint_fixedpoint_check(info); "" catch e; first(split(sprint(showerror, e), "\n")) end
    tape1 = fpv_tape_elems(pi_, names_fp, int_args, vals)
    tape2 = haskey(int_args, :maxit) && regime === :converged ? fpv_tape_elems(pi_, names_fp, merge(int_args, Dict(:maxit => 2 * int_args[:maxit])), vals) : -1
    isempty(raised) || return (err = NaN, info = info, changed = 0, fderr = NaN, verdict = verdict, vleft = NaN, raised = raised, errfull = NaN, tape = tape1, tape2 = tape2, conv = length(fp.converged))
    # the adjoints (every shadow, every returned scalar); the primal arrays are not compared: the unrolled reverse rolls them back to the entry values, the
    # fixed-point reverse leaves the state at the fixed point (checked against the primal below)
    m = 0.0; changed = 0
    for (a, ct) in c1.tracked
        ct.kind == :array_float || continue
        m = max(m, sg_relerr(ct.sh, t2[a].sh)); changed += count(!=(0.0), ct.sh .- c1.pre[a].sh)
    end
    m = max(m, sg_scalar_err(n_ret, c1.ret, r2))
    errfull = 0.0; vleft = 0.0
    if !nested
        # the same adjoint with the cap-sized tape: every shadow, value and returned scalar equal bit for bit (1e-9 when a solve array was converted)
        a3, t3 = sg_build_call(kernel.sig, int_args, vals, seed, identity)
        s3 = sg_init_stacks(li, sg_stack_extra(names_full, int_args, vals, identity))
        r3 = Base.invokelatest(lg, a3..., s3...)
        for (a, ct) in t3
            ct.kind == :array_float || continue
            errfull = max(errfull, maximum(abs, ct.sh .- t2[a].sh; init = 0.0), maximum(abs, ct.val .- t2[a].val; init = 0.0))
        end
        errfull = max(errfull, maximum(abs(x - y) for (x, y) in zip(sg_rets(n_ret, r3), sg_rets(n_ret, r2)); init = 0.0))
        errfull = max(errfull, Float64(abs(s3[end][1] - s2[end][1]) + abs(s3[end][2] - s2[end][2])))
        # the state arrays the adjoint leaves are the primal result of the same loop
        pf = STADE.val_compile(primal)
        pargs = STADE.val_primal_call_args(kernel.sig, int_args, Dict{Symbol,Any}(k => deepcopy(v) for (k, v) in vals))
        Base.invokelatest(pf, pargs...)
        for (i, a) in enumerate(kernel.sig.args)
            kernel.sig.kinds[a] == :array_float && a in fp.loop.arrays && (vleft = max(vleft, sg_relerr(pargs[i], t2[a].val)))
        end
    end
    fderr = fd ? fpv_fd(kernel, primal, fp, int_args, vals, names_fp, :tol) : NaN
    return (err = m, info = info, changed = changed, fderr = fderr, verdict = verdict, vleft = vleft, raised = "", errfull = errfull, tape = tape1, tape2 = tape2, conv = length(fp.converged))
end

# the same draw run by the device code of the fixed-point adjoint (`env` of validate_adjoint_split_gpu.jl): the worst relative error of every shadow and
# returned scalar against the CPU fixed-point adjoint, and the fp_info of both
function fpv_device(dir, name, pins, floats, gs, prep, env, regime::Symbol = :converged)
    return fpv_device_expr(STADE.io_read_corpus_entry(joinpath(@__DIR__, dir, name * ".jl")), pins, floats, gs, prep, env, regime)
end

# the same for a kernel given as an Expr (a generated shape of fp_shapes.jl)
function fpv_device_expr(primal, pins, floats, gs, prep, env, regime::Symbol = :converged)
    kernel = STADE.parse_kernel(primal)
    pins2 = copy(pins)
    regime === :cap && (pins2[:maxit] = 3)
    regime === :unsettled && (pins2[:maxit] = 14)
    Random.seed!(20261122)
    bl = STADE.val_generate_baseline(kernel, primal; self_check = false, int_pin = pins2, int_lo = 3, int_hi = 5, grow_start = gs, grow_max = 64)
    STADE.val_coerce_int_arrays!(kernel, bl.values)
    int_args = merge(bl.int_args, pins2); vals = bl.values
    regime === :nocap && (int_args = merge(int_args, Dict(:maxit => 0), haskey(int_args, :maxq) ? Dict(:maxq => 0) : Dict{Symbol,Int}()))
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

# stade_adjoint is the one place that chooses between the unrolled and the fixed-point reverse; stade_adjoint_file and the validators go through it
function fpv_logged(f)
    io = IOBuffer()
    r = Logging.with_logger(f, Logging.SimpleLogger(io, Logging.Info))
    return (value = r, log = String(take!(io)))
end

fpv_has_fp_info(g) = last(g.adjoint.args[1].args) === :fp_info

function fpv_dispatch_checks(say)
    root = @__DIR__
    rd(dir, nm) = STADE.io_read_corpus_entry(joinpath(root, dir, nm * ".jl"))
    cos_ = rd("fp-corpus", "fp_scalar_cos")
    r = fpv_logged(() -> STADE.stade_adjoint(cos_; keep_push_pop = false))
    say(fpv_has_fp_info(r.value), "stade_adjoint on a counted-guard loop with keep_push_pop = false returns the fixed-point adjoint")
    say(occursin("takes the fixed-point reverse", r.log) && occursin("fp_scalar_cos", r.log), "the fixed-point path says so with one @info line (got: $(first(r.log, 120)))")
    say(count("takes the fixed-point reverse", r.log) == 1, "the acceptance line appears once")
    say(hasproperty(r.value, :primal_pruned) && hasproperty(r.value, :initstacks), "the fixed-point result has the keys of a stade_adjoint result")
    for (kw, frag) in ((:(keep_push_pop = true), "keep_push_pop = true"), (:(fuse_ii_loops = true), "fuse_ii_loops = true"), (:(primal_outputs = false), "primal_outputs = false"))
        kws = kw.args[1] === :keep_push_pop ? (; keep_push_pop = kw.args[2]) : (; keep_push_pop = false, (kw.args[1] => kw.args[2],)...)
        rr = fpv_logged(() -> STADE.stade_adjoint(cos_; kws...))
        say(!fpv_has_fp_info(rr.value) && occursin("keeps its unrolled reverse", rr.log) && occursin(frag, rr.log), "$frag: the unrolled adjoint and one line that names the flag (got: $(first(rr.log, 140)))")
    end
    cg = rd("counted-corpus", "phc_cg_laplace2d")
    rc = fpv_logged(() -> STADE.stade_adjoint(cg; keep_push_pop = false))
    say(!fpv_has_fp_info(rc.value) && occursin("divides by a value computed from a reduction", rc.log), "a refused loop keeps the unrolled adjoint and the log gives the reason (got: $(first(rc.log, 140)))")
    plain = rd("val-corpus", "advection")
    rp = fpv_logged(() -> STADE.stade_adjoint(plain; keep_push_pop = false))
    say(!fpv_has_fp_info(rp.value) && isempty(rp.log), "a kernel without a counted-guard loop: the unrolled adjoint and no line (got: $(first(rp.log, 100)))")
    # independents reach the fixed-point adjoint: the same text as the generator with them, and not the text without them
    ri = fpv_logged(() -> STADE.stade_adjoint(cos_; independents = [:c], keep_push_pop = false)).value
    with_i = STADE.agen_adjoint_fixedpoint(cos_; independents = [:c]); without_i = STADE.agen_adjoint_fixedpoint(cos_)
    say(string(with_i.adjoint) != string(without_i.adjoint), "the independents change the fixed-point adjoint of fp_scalar_cos")
    say(string(ri.adjoint) == string(with_i.adjoint), "stade_adjoint passes independents = [:c] to the fixed-point adjoint")
    # the validators check the unrolled adjoint and the HVP shares its stacks: a fixed-point adjoint here would not call
    mktempdir() do tmp
        f = joinpath(tmp, "fp_scalar_cos.jl"); cp(joinpath(root, "fp-corpus", "fp_scalar_cos.jl"), f); yp = joinpath(tmp, "fp_scalar_cos.yaml")
        fpv_logged(() -> STADE.stade_generate_baseline_file(f; yaml_path = yp, int_lo = 3, int_hi = 5))
        rd_ = fpv_logged(() -> STADE.stade_validate_dotprod_file(f; yaml_path = yp, keep_push_pop = false))
        say(rd_.value.ok && occursin("this validator checks the unrolled adjoint", rd_.log), "stade_validate_dotprod_file: the exact identity holds for the unrolled adjoint and the log says which adjoint it checks (ok = $(rd_.value.ok))")
        ra = fpv_logged(() -> STADE.stade_validate_from_baseline(:adjoint, f, yp; keep_push_pop = false))
        say(ra.value.ok && occursin("this validator checks the unrolled adjoint", ra.log), "stade_validate_from_baseline(:adjoint) on a counted-guard loop (ok = $(ra.value.ok))")
        rh = try fpv_logged(() -> STADE.stade_validate_from_baseline(:hvp, f, yp; keep_push_pop = false)).value.ok catch e; false end
        say(rh, "stade_validate_from_baseline(:hvp) on a counted-guard loop")
    end
    # the file entry goes through the same choice
    mktempdir() do tmp
        for (kpp, want) in ((false, true), (true, false))
            out = joinpath(tmp, "o_$(kpp).jl")
            fpv_logged(() -> STADE.stade_adjoint_file(joinpath(root, "fp-corpus", "fp_scalar_cos.jl"), out; keep_push_pop = kpp))
            say(occursin("fp_info", read(out, String)) == want, "stade_adjoint_file with keep_push_pop = $kpp: the generated file " * (want ? "has" : "has no") * " fp_info")
        end
    end
end

# the candidates of the converged-state rule on a generated shape, and the pieces the rule reads
function fpv_converged_parts(body, nest)
    primal = fps_kernel(body, nest)
    kernel = STADE.norm_insert_dead_entry_resets(STADE.parse_kernel(primal))
    dets = STADE.agen_fp_detect(primal, kernel.sig.kinds; kernel = kernel)
    _, sites = STADE.stade_site_level_tbr_check(kernel; ii_plan = nothing)
    return (kernel = kernel, dets = dets, sites = sites, cands = STADE.agen_fp_converged_cands(kernel, dets, sites))
end

# The guards of the converged-state rule that a whole kernel cannot reach: a write after the solve that has no save pending (the generator's own decision makes every write
# after a nonlinear read saved), and the one that is saved is what the reverse restores.
function fpv_converged_unit_checks(say)
    P = fpv_converged_parts(:buf, :cover_post)
    c = P.cands[findfirst(x -> x.array === :w, P.cands)]
    say(STADE.agen_fp_converged_later_ok(P.kernel, P.dets, c, P.sites), "converged state: the cover_post shape passes with every save pending")
    outside = [k for (k, insolve) in STADE.agen_fp_collect_writes!(P.kernel.body, c.top:length(P.kernel.body), :w, false, P.dets, Any[]) if !insolve && !(k in c.keys)]
    say(length(outside) == 2, "converged state: the cover_post shape writes `w` at two sites outside the solve (got $(length(outside)))")
    for k in outside
        s2 = copy(P.sites); s2[k] = false
        say(!STADE.agen_fp_converged_later_ok(P.kernel, P.dets, c, s2), "converged state: a write of `w` with no save pending refuses the rule")
    end
    # a write inside another solve loop refuses it, whatever its save
    Q = fpv_converged_parts(:buf, :cover_two)
    say(!isempty(Q.cands) && all(c2 -> !STADE.agen_fp_converged_later_ok(Q.kernel, Q.dets, c2, Q.sites), Q.cands), "converged state: two solve loops writing one array refuse the rule")
    # the bound of the pre-write is a scalar assigned in the iteration
    R = fpv_converged_parts(:buf, :cover_nn)
    say(!isempty(R.cands) && !STADE.agen_fp_converged_rolled_back(R.cands[1]), "converged state: a bound assigned in the iteration does not prove the cover")
    # the first loop of the cover shape is covered, the partial one is not
    S = fpv_converged_parts(:buf, :cover)
    say(!isempty(S.cands) && STADE.agen_fp_converged_rolled_back(S.cands[1]), "converged state: the cover shape is rolled back")
    T = fpv_converged_parts(:buf, :cover_partial)
    say(!isempty(T.cands) && !STADE.agen_fp_converged_rolled_back(T.cands[1]), "converged state: a pre-write of fewer elements does not cover")
    # a solve that no loop encloses has one instance
    U = fpv_converged_parts(:buf, :none)
    say(!isempty(U.cands) && STADE.agen_fp_converged_rolled_back(U.cands[1]), "converged state: a solve that no loop encloses is rolled back")
    # the arrays of the body: a scratch array written, read and written again is not a candidate, the carried one is
    M = fpv_converged_parts(:mid, :cover_post)
    say(all(x -> x.array !== :w, M.cands) && any(x -> x.array === :wo, M.cands), "converged state: `w` (read between two writes) is not a candidate, `wo` is (got $([x.array for x in M.cands]))")
end

function fpv_unit_checks(say)
    root = @__DIR__
    fpv_converged_unit_checks(say)
    for (nm, frag) in (("fp_counter_used", "has 0 counted-guard loops"), ("fp_accumulate", "never consumes a state adjoint"))
        primal = STADE.io_read_corpus_entry(joinpath(root, "fp-corpus", nm * ".jl"))
        msg = try STADE.agen_adjoint_fixedpoint(primal); "" catch e; sprint(showerror, e) end
        say(occursin(frag, msg), "$nm is refused with `$frag` (got: $(first(msg, 120)))")
    end
    # two solve loops at the top level are accepted: one forward and one reversed loop each
    tl = STADE.agen_adjoint_fixedpoint(STADE.io_read_corpus_entry(joinpath(root, "fp-corpus", "fp_two_loops.jl")))
    say(length(tl.loops) == 2, "fp_two_loops: both solve loops are rewritten (got $(length(tl.loops)))")
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
        why = length(f) == 1 ? STADE.agen_fp_refusal(f[1]) : String["not detected"]
        say(isempty(frag) ? isempty(why) : any(w -> occursin(frag, w), why), "$nm: the rule " * (isempty(frag) ? "accepts" : "refuses with `$frag`") * " (got: $(isempty(why) ? "accepted" : join(why, " | ")))")
    end
    # the nests of the generator: the accepted ones have no reason, each refused one has its own
    for (body, nest) in fpv_shape_cases()
        pr = fps_kernel(body, nest); k = STADE.parse_kernel(pr)
        why = vcat(String[], [STADE.agen_fp_refusal(d) for d in STADE.agen_fp_detect(pr, k.sig.kinds)]...)
        say(isempty(why), "fps_$(body)_$(nest): the rule accepts (got: $(join(why, " | ")))")
    end
    for (nest, frag) in ((:cold, "independent iterations"), (:indep, "independent iterations"), (:solve_in_solve, "another solve loop"), (:hi_assigned, "assigned inside an enclosing loop"),
                         (:tol_assigned, "assigned inside an enclosing loop"), (:hi_assigned, "used outside the headers of solve loops"),
                         (:ragged, "ragged nest"), (:bound_self, "is assigned inside it"), (:triangle, "depends on an outer counter"), (:read_top, "before a solve loop overwrites it"), (:read_before, "before a solve loop overwrites it"),
                         (:read_between, "before a solve loop overwrites it"), (:read_nl, "before a solve loop overwrites it"))
        pr = fps_kernel(:scalar, nest); k = STADE.parse_kernel(pr)
        why = vcat(String[], [STADE.agen_fp_refusal(d) for d in STADE.agen_fp_detect(pr, k.sig.kinds)]...)
        say(any(w -> occursin(frag, w), why), "fps_scalar_$nest: the rule refuses with `$frag` (got: $(isempty(why) ? "accepted" : join(why, " | ")))")
    end
    pr = fps_kernel(:array, :tdep_nl); k = STADE.parse_kernel(pr)
    why = vcat(String[], [STADE.agen_fp_refusal(d) for d in STADE.agen_fp_detect(pr, k.sig.kinds)]...)
    say(any(w -> occursin("before a solve loop overwrites it", w), why), "fps_array_tdep_nl: an array read after the solve and before the next one overwrites it is a reason")
    # a loop with independent iterations around the solve stays with the unrolled reverse, and says why
    pr = fps_kernel(:scalar, :indep)
    r = fpv_logged(() -> STADE.stade_adjoint(pr; keep_push_pop = false))
    say(!occursin("fp_info", string(r.value.adjoint)) && occursin("independent iterations", r.log), "an enclosing loop with independent iterations keeps the unrolled reverse and the log says why")
    # the cap of the solve loop as the bound of another loop at the top level: the one-slot tape would size that loop wrongly
    sq = Meta.parse("function fpk(a, x0, tol, maxit, v, y)\n x = x0\n rn = 1.0\n for kk = 1:maxit\n if rn > tol\n xn = a * cos(x)\n rn = (xn - x) * (xn - x)\n x = xn\n end\n end\n for j = 1:maxit\n v[j] = x * j\n end\n y[1] = x\n return nothing\nend")
    k = STADE.parse_kernel(sq); why = vcat(String[], [STADE.agen_fp_refusal(d) for d in STADE.agen_fp_detect(sq, k.sig.kinds)]...)
    say(any(w -> occursin("used outside the headers of solve loops", w), why), "a loop outside the solve that runs over the cap is a reason (got: $(isempty(why) ? "accepted" : join(why, " | ")))")
    # a function the validated cases never call is a reason
    for (fn, ok) in ((:sqrt, false), (:tanh, true))
        sq = Meta.parse("function fpw(a, x0, tol, maxit, y)\n x = x0\n rn = 1.0\n for kk = 1:maxit\n if rn > tol\n xn = $fn(x) + a\n rn = (xn - x) * (xn - x)\n x = xn\n end\n end\n y[1] = x\n return nothing\nend")
        k = STADE.parse_kernel(sq); f = STADE.agen_fp_detect(sq, k.sig.kinds)
        why = length(f) == 1 ? STADE.agen_fp_refusal(f[1]) : String["not detected"]
        say(isempty(why) == ok, "a loop that calls $fn: the rule " * (ok ? "accepts" : "refuses") * " (got: $(isempty(why) ? "accepted" : join(why, " | ")))")
    end
    # the one-slot transformer refuses what it cannot prove: a statement after the guard that is no tape write, a read of the branch stack in the guard, the
    # counter used outside the loops. Each is built from the real adjoint of fp_scalar_cos with one statement added.
    cos_ = STADE.io_read_corpus_entry(joinpath(root, "fp-corpus", "fp_scalar_cos.jl")); kc = STADE.parse_kernel(cos_)
    det = STADE.agen_fp_detect(cos_, kc.sig.kinds)[1]
    rr = STADE.stade_adjoint_(cos_, nothing, nothing, false, false, true, false)
    sts = Any[x for x in rr.adjoint.args[2].args if !(x isa LineNumberNode)]
    fi_ = findfirst(x -> (h = STADE.agen_fp_header(x); h !== nothing && h.step === 1 && h.var === det.var), sts)
    trip = :(max(0, div($(det.hi) - 1, 1) + 1))
    fl = sts[fi_]; fb = STADE.agen_fp_stmts(fl.args[2]); bst = first(x.args[1].args[1] for x in STADE.agen_fp_stmts(fb[1].args[3]) if x isa Expr && x.head === :(=) && x.args[1] isa Expr && x.args[1].head === :ref)
    with_post(x) = Expr(:for, fl.args[1], Expr(:block, fb..., x))
    say(try STADE.agen_fp_one_slot_forward(fl, det, Any[trip], Dict{Symbol,Any}()); true catch; false end, "the transformer accepts the forward loop as generated")
    say(try STADE.agen_fp_one_slot_forward(with_post(:(zz = 1.0)), det, Any[trip], Dict{Symbol,Any}()); false catch e; occursin("not a tape write", sprint(showerror, e)) end, "the transformer refuses a statement after the guard that is no tape write")
    say(try STADE.agen_fp_one_slot_forward(with_post(:(__idx_zz = $bst[1])), det, Any[trip], Dict{Symbol,Any}()); false catch e; occursin("used in the guard besides its flag", sprint(showerror, e)) end, "the transformer refuses a read of the branch stack in the guard")
    sts2 = Any[sts[1:end-1]..., :(zz = $(det.var)), sts[end]]
    say(try STADE.agen_fp_rewrite_one_slot(Expr(:function, rr.adjoint.args[1], Expr(:block, sts2...)), rr.initstacks, [det]); false catch e; occursin("used outside the loop", sprint(showerror, e)) end, "the transformer refuses the counter used outside the loops")
    # the cap read in the body is a reason: an inner bound that uses it would be sized wrongly by the one-slot tape
    for (cap_in_body, ok) in (("1:maxit", false), ("1:3", true))
        sq = Meta.parse("function fpc(a, x0, tol, maxit, v, y)\n x = x0\n rn = 1.0\n for kk = 1:maxit\n if rn > tol\n s = 0.0\n for j = $cap_in_body\n s = s + v[j]\n end\n xn = a + 0.5 * x + 0.0 * s\n rn = (xn - x) * (xn - x)\n x = xn\n end\n end\n y[1] = x\n return nothing\nend")
        k = STADE.parse_kernel(sq); f = STADE.agen_fp_detect(sq, k.sig.kinds)
        why = length(f) == 1 ? STADE.agen_fp_refusal(f[1]) : String["not detected"]
        say(ok ? !any(w -> occursin("loop's cap", w), why) : any(w -> occursin("loop's cap", w), why), "an inner loop over $cap_in_body: the rule " * (ok ? "does not refuse" : "refuses") * " for the cap (got: $(isempty(why) ? "accepted" : join(why, " | ")))")
    end
    # the verdict raises on each flag
    say(try STADE.agen_adjoint_fixedpoint_check([3.0, 0.0, 0.0, 1.0, 2.0]); false catch e; occursin("stopped on its cap", sprint(showerror, e)) end, "check raises when the forward loop hit its cap")
    say(try STADE.agen_adjoint_fixedpoint_check([3.0, 2.0, 0.1, 0.0, 1.0]); false catch e; occursin("did not settle", sprint(showerror, e)) end, "check raises when the sweeps did not settle")
    say(STADE.agen_adjoint_fixedpoint_check([3.0, 2.0, 0.0, 0.0, 0.0]) == 2, "check returns the sweeps when nothing is flagged")
end

# generated shapes of fp_shapes.jl that must take the fixed-point reverse
fpv_shape_cases() = Any[(:scalar, :none), (:scalar, :loop1), (:scalar, :loop2), (:scalar, :ifbranch), (:scalar, :solve_first), (:scalar, :solve_last),
                        (:scalar, :two_in_body), (:scalar, :two_loops), (:scalar, :literal), (:scalar, :two_caps), (:scalar, :tdep_first), (:scalar, :pre_nl),
                        (:array, :loop1), (:array, :ifbranch), (:array, :two_loops), (:array, :literal), (:array, :two_caps), (:array, :tdep_first), (:array, :read_before),
                        (:array, :none), (:array, :read_top),
                        (:buf, :none), (:buf, :read_top), (:buf, :loop1), (:buf, :tdep_first), (:buf, :read_before), (:buf, :two_loops),
                        (:buf, :cover), (:buf, :cover_partial), (:buf, :cover_acc), (:buf, :cover_post), (:buf, :cover_postacc), (:buf, :cover_two), (:buf, :cover_if),
                        (:buf, :cover_ifpre), (:buf, :cover_cond), (:buf, :cover_loop2_inner), (:buf, :cover_loop2_outer),
                        (:mid, :none), (:mid, :cover), (:mid, :cover_post)]

# the shapes the host path takes and the GPU ingest cannot: a loop bound assigned in the iteration gives tier-B tables read by an inline index (`prefix_x[t]`), which
# cgen_ingest does not parse (not specific to the fixed-point path)
fpv_shape_cases_host_only() = Any[(:buf, :cover_nn)]

# Whether the sweeps of a generated shape read an array as the solve left it (agen_fp_converged_apply): a solve that no loop encloses, or one that a write with a save covering
# (a write under an `if` that also holds the solve counts, one under an `if` that the solve is outside of does not)
# everything it writes precedes in its iteration, and that nothing after it or in another solve disturbs. The shapes not listed are not asserted.
fpv_expect_converged() = Dict{String,Bool}(
    "fps_array_none" => true, "fps_array_read_top" => true, "fps_array_loop1" => false, "fps_array_tdep_first" => false,
    "fps_buf_none" => true, "fps_buf_read_top" => true, "fps_buf_loop1" => false, "fps_buf_tdep_first" => false, "fps_buf_read_before" => false, "fps_buf_two_loops" => false,
    "fps_buf_cover" => true, "fps_buf_cover_acc" => true, "fps_buf_cover_post" => true, "fps_buf_cover_postacc" => true, "fps_buf_cover_if" => true, "fps_buf_cover_loop2_inner" => true,
    "fps_buf_cover_partial" => false, "fps_buf_cover_two" => false, "fps_buf_cover_ifpre" => true, "fps_buf_cover_cond" => false, "fps_buf_cover_loop2_outer" => false, "fps_buf_cover_nn" => false,
    "fps_mid_none" => false, "fps_mid_cover" => false, "fps_mid_cover_post" => true)

# (pins, floats, array start size, preparation or nothing) of a generated shape
function fpv_shape_params(body, nest)
    primal = fps_kernel(body, nest)
    args = Set(STADE.parse_kernel(primal).sig.args)
    pins = Dict(k => v for (k, v) in (body === :scalar ? Dict(:maxit => 500, :maxq => 500, :nt => 3, :ns => 2) : Dict(:n => 6, :maxit => 800, :maxq => 800, :nt => 3, :ns => 2)) if k in args)
    floats = Dict(k => v for (k, v) in (body === :scalar ? Dict(:c => 0.6, :tol => 1e-30) : Dict(:c => 0.3, :tol => 1e-30)) if k in args)
    prep = :pa in args ? (v -> (v[:pa] .= [iseven(i) ? 0.2 : 0.9 for i in eachindex(v[:pa])])) : nothing
    return (primal = primal, pins = pins, floats = floats, gs = body === :scalar ? 4 : 8, prep = prep)
end

# the four regimes of one case; `run(regime)` is fpv_run or fpv_run_expr bound to the case
function fpv_case_checks(say, name, run, verbose; expect_conv = nothing)
    r = run(:converged)
    expect_conv === nothing || say((r.conv > 0) == expect_conv, "$name converged: the sweeps " * (expect_conv ? "should" : "should not") * " read an array as the solve left it (arrays: $(r.conv))")
    verbose && println(rpad(name, 22), " converged: err=", round(r.err, sigdigits = 3), " fd=", round(r.fderr, sigdigits = 3), " sweeps=", Int(r.info[2]), " iterations=", Int(r.info[1]), " ratio=", round(r.info[3], sigdigits = 3), " changed=", r.changed, " tape=", r.tape)
    say(isempty(r.raised), "$name converged: the adjoint raised ($(r.raised))")
    say(r.err <= 1e-9, "$name converged: fixed-point vs unrolled adjoint err=$(r.err)")
    say(r.fderr <= 1e-5, "$name converged: fixed-point adjoint vs central difference err=$(r.fderr)")
    say(r.info[2] >= 1 && r.info[1] >= 1 && r.info[4] == 0 && r.info[5] == 0 && r.verdict == "", "$name converged: sweeps ran, no flag ($(r.info)) $(r.verdict)")
    say(r.changed > 0, "$name converged: nothing changed (vacuous)")
    say(r.vleft <= 1e-6, "$name converged: the state the adjoint leaves is not the primal result ($(r.vleft))")
    # the same adjoint to the last bit, unless the sweeps read an array as the loop left it (its saves are not taken): the two then differ by the last residual's square root
    say(r.conv == 0 ? r.errfull == 0.0 : r.errfull <= 1e-9, "$name converged: the one-slot tape and the cap-sized tape differ by $(r.errfull)" * (r.conv == 0 ? "" : " ($(r.conv) arrays read at the converged state)"))
    say(r.tape > 0 && r.tape == r.tape2, "$name converged: the tape has $(r.tape) elements for the cap and $(r.tape2) for twice the cap")
    z = run(:zero)
    say(isempty(z.raised) && z.err <= 1e-12 && z.info[1] == 0 && z.info[2] == 0 && z.info[4] == 0 && z.info[5] == 0, "$name zero: err=$(z.err) info=$(z.info) raised=$(z.raised)")
    c = run(:cap)
    # an instance that ran to its cap is not at a fixed point: no sweep runs and the adjoint raises
    say(occursin("stopped on its cap", c.raised), "$name cap: the adjoint raises the cut-off message (got: $(first(c.raised, 100)))")
    say(c.info[4] >= 1 && c.info[2] == 0 && c.info[5] == 2, "$name cap: the cut-off count is set, no sweep ran, the status is 2 ($(c.info))")
    say(occursin("cap", c.verdict), "$name cap: the host verdict names the cap ($(c.verdict))")
    n0 = run(:nocap)
    say(isempty(n0.raised) && all(==(0.0), n0.info) && n0.err <= 1e-12, "$name nocap: a loop with maxit = 0 does not raise and runs no sweep (err=$(n0.err) info=$(n0.info) raised=$(n0.raised))")
    return nothing
end

function validate_fixedpoint(; verbose = true, only = String[], sabotage = false)
    checks = 0; bad = 0
    say(ok, msg) = (checks += 1; ok || (bad += 1; verbose && println("NOT OK ", msg)))
    fpv_unit_checks(say)
    fpv_dispatch_checks(say)
    for (dir, name, pins, floats, gs, prep) in fpv_cases()
        isempty(only) || name in only || continue
        fpv_case_checks(say, name, reg -> fpv_run(dir, name, pins, floats, gs, prep, reg; fd = reg === :converged), verbose)
    end
    # generated shapes: one solve loop, or two, inside loops and branches at any depth (fp_shapes.jl)
    for (body, nest) in vcat(fpv_shape_cases(), fpv_shape_cases_host_only())
        name = "fps_$(body)_$(nest)"
        isempty(only) || name in only || continue
        sp = fpv_shape_params(body, nest); primal = sp.primal
        fpv_case_checks(say, name, reg -> fpv_run_expr(primal, sp.pins, sp.floats, sp.gs, sp.prep, reg; fd = reg === :converged), verbose; expect_conv = get(fpv_expect_converged(), name, nothing))
    end
    u = fpv_unsettled()
    say(occursin("did not settle", u.msg), "fp_slow_linear: a converged loop whose reverse cannot settle raises (got: $(first(u.msg, 100)))")
    say(u.info[4] == 0 && u.info[5] == 1 && u.info[1] >= 1, "fp_slow_linear: the forward loop converged and the sweeps stopped on their cap ($(u.info))")
    verbose && println("validate_fixedpoint: $(checks - bad)/$checks ok")
    if sabotage
        text = read(joinpath(dirname(@__DIR__), "src", "STADE.jl"), String)
        cases = [
            ("the counter may be read in the body", "    h.var in syms && return nothing\n", "", ["fp_scalar_cos"]),
            ("the counter stays in the tape positions", "    body = Any[agen_fp_slot(s, trips, kk, true) for s in Any[then; post]]\n", "    body = Any[agen_fp_slot(s, trips, kk, false) for s in Any[then; post]]\n", ["fp_scalar_cos", "fps_scalar_loop1"]),
            ("the trip count stays in the sizes", "    body_i = Any[agen_fp_slot(st, trips, :__fp_none, false) for st in ib[1:end-1]]\n", "    body_i = Any[st for st in ib[1:end-1]]\n", ["fp_scalar_cos", "fps_scalar_loop1"]),
            ("the trip count stays in the positions outside the solve", "    out = Any[agen_fp_slot(st, trips, :__fp_none, false) for st in nst]\n", "    out = Any[st for st in nst]\n", ["fps_scalar_loop1"]),
            ("only the first cap is substituted", "    trips = Any[agen_fp_trip(hi) for hi in unique(Symbol[d.hi for d in dets])]\n", "    trips = Any[agen_fp_trip(hi) for hi in unique(Symbol[d.hi for d in dets[1:1]])]\n", ["fps_scalar_two_caps"]),
            ("the count stack is not reset before the loop", "    return (Any[:(\$bstack[\$E] = 0), Expr(:for, floop.args[1], Expr(:block, guard))], bstack, E)\n", "    return (Any[Expr(:for, floop.args[1], Expr(:block, guard))], bstack, E)\n", ["fps_scalar_ifbranch", "fps_scalar_loop1"]),
            ("the count stack is read at the wrong instance", "        __fp_cnt = \$bstack[\$E]\n", "        __fp_cnt = \$bstack[1]\n", ["fps_scalar_ifbranch"]),
            ("the sweep counter is not reset per instance", "            __fp_sw = 0\n            __fp_go = 0\n            if __fp_cnt > 0\n", "            __fp_go = 0\n            if __fp_cnt > 0\n", ["fps_scalar_loop1"]),
            ("the sweeps run when no iteration ran", "            if __fp_cnt > 0\n                __fp_go = 1\n            end\n", "            __fp_go = 1\n", ["fp_scalar_cos", "fps_scalar_ifbranch"]),
            ("the first sweep's norm is not kept", "\$(sweep...)\n                    if __fp_sw == 0\n                        __fp_c0 = __fp_cn\n                    end\n", "\$(sweep...)\n                    __fp_c0 = __fp_cn\n", ["fp_scalar_cos"]),
            ("one sweep only", "                    if __fp_cn <= \$tolx * __fp_c0\n                        __fp_go = 0\n                    end\n                end\n            end\n            __fp_iters", "                    __fp_go = 0\n                end\n            end\n            __fp_iters", ["fp_scalar_cos"]),
            ("the cut-off count is never set", "                        __fp_ncut = __fp_ncut + 1\n", "                        __fp_ncut = __fp_ncut + 0\n", ["fp_scalar_cos", "fps_scalar_loop1"]),
            ("the cut-off test runs for maxit = 0", "        cut = :(if \$(d.hi) >= 1\n", "        cut = :(if \$(d.hi) >= 0\n", ["fp_scalar_cos"]),
            ("the cut-off status is the sweeps' status", "        if __fp_ncut > 0\n            fp_info[5] = 2.0\n", "        if __fp_ncut > 0\n            fp_info[5] = 1.0\n", ["fp_scalar_cos"]),
            ("a cut-off instance does not raise", "        if __fp_ncut > 0\n            error(\$msgcut)\n        end\n", "", ["fp_scalar_cos", "fps_scalar_loop1"]),
            ("the cut-off check runs before the forward nest", "    final = Any[start...; out[1:last_fwd]; raise_cut; out[last_fwd+1:end-1]; raise_end; out[end]]\n", "    final = Any[start...; raise_cut; out[1:end-1]; raise_end; out[end]]\n", ["fps_scalar_loop1"]),
            ("the sweeps' status is never set", "        if __fp_unset > 0\n            fp_info[5] = 1.0\n        end\n", "", ["fp_scalar_cos"]),
            ("the sweeps' error is not raised", "        if __fp_unset > 0\n            error(\$msg)\n        end\n", "", ["fp_scalar_cos"]),
            ("the sweeps' unsettled count is never set", "                __fp_unset = __fp_unset + 1\n", "                __fp_unset = __fp_unset + 0\n", ["fp_scalar_cos"]),
            ("the index chain of the sweeps loses a link", "        (agen_fp_uses(st, bstack) || !(lhs isa Symbol && any(t -> agen_fp_uses(t, lhs), later))) && (keep[j] = false)\n", "        (agen_fp_uses(st, bstack) || !(lhs isa Symbol && any(t -> !any(u -> u === t, sl) && agen_fp_uses(t, lhs), later))) && (keep[j] = false)\n", ["fps_scalar_two_loops"]),
            ("the branch stack may be read in the guard", "    any(s -> agen_fp_uses(s, bstack) && !(s === body[flag[1]]), body) && error(", "    false && error(", ["fp_scalar_cos"]),
            ("the counter may be used outside the loops", "        counter_outside(body, d.var) && error(", "        false && error(", ["fp_scalar_cos"]),
            ("the cap may be used outside the solve loops", "    det.cap_elsewhere > 0 && push!(why,", "    false && push!(why,", ["fp_scalar_cos"]),
            ("the cap uses outside the solves are not counted", "cap_elsewhere = total[d.hi] - nhead[d.hi] - inbody", "cap_elsewhere = 0", ["fp_scalar_cos"]),
            ("an enclosing loop with independent iterations is accepted", "        agen_fp_carries(L) || push!(why,", "        true || push!(why,", ["fp_scalar_cos"]),
            ("a ragged bound read from an outer counter is accepted", "        isempty(intersect(bsyms, Set(counters))) || push!(why,", "        true || push!(why,", ["fp_scalar_cos"]),
            ("a bound assigned inside an outer loop is accepted", "            isempty(intersect(bsyms, asg)) || (push!(why,", "            true || (push!(why,", ["fp_scalar_cos"]),
            ("a bound assigned inside its own loop is accepted", "        isempty(intersect(bsyms, own)) || push!(why,", "        true || push!(why,", ["fp_scalar_cos"]),
            ("a tolerance assigned in an enclosing loop is accepted", "        v in asg && push!(why,", "        false && push!(why,", ["fp_scalar_cos"]),
            ("the detector does not look into the bodies of loops", "            agen_fp_walk_solves!(agen_fp_stmts(st.args[2]), Any[enclosing..., st], kinds, out)\n        elseif st.head === :if\n", "        elseif st.head === :if\n", ["fp_scalar_cos", "fps_scalar_loop1"]),
            ("the detector does not look into the branches", "            agen_fp_walk_solves!(agen_fp_stmts(st.args[2]), Any[enclosing..., st], kinds, out)\n            length(st.args) == 3", "            length(st.args) == 3", ["fp_scalar_cos", "fps_scalar_ifbranch"]),
            ("the carried-value test accepts a loop of independent iterations", "    return false\nend\n\n# why the nest around one solve loop", "    return true\nend\n\n# why the nest around one solve loop", ["fp_scalar_cos"]),
            ("a statement after the guard is accepted", "    all(agen_fp_is_tape_stmt, post) || error(", "    true || error(", ["fp_scalar_cos"]),
            ("the cap may be read in the body", "    f.cap_in_body && push!(why,", "    false && push!(why,", ["fp_scalar_cos"]),
            ("the host check ignores a cut-off", "    (info[4] == 0 && info[5] != 2) || error(", "    true || error(", ["fp_scalar_cos"]),
            ("accumulation loops are accepted", "    count > 0 || error(\"agen_adjoint_fixedpoint: the reverse of the loop over `\$kk`", "    count >= 0 || error(\"agen_adjoint_fixedpoint: the reverse of the loop over `\$kk`", ["fp_scalar_cos"]),
            ("a division by a reduction is not a reason", "    f.div_by_reduction && push!(why,", "    false && push!(why,", ["fp_scalar_cos"]),
            ("an if in the body is not a reason", "    f.inner_if && push!(why,", "    false && push!(why,", ["fp_scalar_cos"]),
            ("a literal tolerance is not a reason", "    f.literal_tol && push!(why,", "    false && push!(why,", ["fp_scalar_cos"]),
            ("deep nesting is not a reason", "    f.depth > 2 && push!(why,", "    f.depth > 99 && push!(why,", ["fp_scalar_cos"]),
            ("every call is validated", "agen_fp_validated_calls() = Set{Symbol}([:div, :mod, :sin, :cos, :tanh])", "agen_fp_validated_calls() = Set{Symbol}([:div, :mod, :sin, :cos, :tanh, :sqrt, :abs])", ["fp_scalar_cos"]),
            ("a scalar assigned from a reduction is not one", "        isempty(intersect(rsyms, red)) || push!(red, e.args[1])\n", "", ["fp_scalar_cos"]),
            ("stade_adjoint does not choose the fixed-point reverse", "    fp === nothing || return fp\n", "    fp === nothing || nothing\n", ["fp_scalar_cos"]),
            ("the acceptance is silent", "        @info \"stade_adjoint: `\$(kernel.sig.name)` has a counted-guard loop and takes", "        nothing; \"stade_adjoint: `\$(kernel.sig.name)` has a counted-guard loop and takes", ["fp_scalar_cos"]),
            ("a refusal is silent", "        @info \"stade_adjoint: `\$(kernel.sig.name)` has a counted-guard loop and keeps its unrolled reverse: \$why\"\n", "        nothing\n", ["fp_scalar_cos"]),
            ("the independents are dropped", "agen_adjoint_fixedpoint(expr; independents = independents, dependents = dependents)", "agen_adjoint_fixedpoint(expr)", ["fp_scalar_cos"]),
            ("the arrays read between two writes are candidates", "        (j in ws || (j > first(ws) && j < last(ws))) && return false\n", "", ["fps_mid_none"]),
            ("the sweeps never read the converged state", "    fp_dets === nothing || ((snap_sites, agen_sites) = agen_fp_converged_apply(", "    false && ((snap_sites, agen_sites) = agen_fp_converged_apply(", ["fps_buf_none"]),
            ("a loop inside a loop needs no cover", "    c.lbody === nothing && return true\n", "    return true\n", ["fps_buf_cover_partial", "fps_buf_cover_cond", "fps_buf_loop1"]),
            ("a pre-write of other elements covers", "        any(p -> p.cond && p.form == l.form && isempty(", "        any(p -> p.cond && isempty(", ["fps_buf_cover_partial"]),
            ("a conditional pre-write covers", "        any(p -> p.cond && p.form == l.form && isempty(", "        any(p -> p.form == l.form && isempty(", ["fps_buf_cover_cond"]),
            ("a bound assigned in the iteration proves the cover", "p.form == l.form && isempty(intersect(agen_fp_form_syms(p.form), assigned)), pre)", "p.form == l.form, pre)", ["fps_buf_cover_nn"]),
            ("a write inside another solve loop is accepted", "        (insolve || get(sites, key, false) !== true) && return false\n", "        get(sites, key, false) !== true && return false\n", ["fps_buf_cover_two"]),
            ("a write without a save pending is accepted", "        (insolve || get(sites, key, false) !== true) && return false\n", "        insolve && return false\n", ["fps_buf_none"]),
            ("the writes of the loop's own statement do not count", "    out = agen_fp_collect_writes!(kernel.body, c.top:length(kernel.body), c.array, false, dets, Any[])\n", "    out = agen_fp_collect_writes!(kernel.body, c.top+1:length(kernel.body), c.array, false, dets, Any[])\n", ["fps_buf_cover_two"]),
            ("a candidate that fails the check is kept", "        bad = [j for (j, c) in enumerate(cur) if !agen_fp_converged_later_ok(kernel, dets, c, a3)]\n", "        bad = Int[]\n", ["fps_buf_cover_two", "fps_buf_cover_partial"]),
            ("the converged arrays are never reported", "            append!(report, [(array = c.array, solve = c.solve) for c in cur])\n", "", ["fps_buf_cover"]),
            ("the HVP validator takes the fixed-point adjoint", "        adjoint_out = agen_adjoint_unrolled(primal_expr; keep_push_pop = keep_push_pop, fuse_ii_loops = fuse_ii_loops)\n        hvp_out = stade_hvp(", "        adjoint_out = stade_adjoint(primal_expr; keep_push_pop = keep_push_pop, fuse_ii_loops = fuse_ii_loops)\n        hvp_out = stade_hvp(", ["fp_scalar_cos"]),
            ("a validator says nothing about the adjoint it checks", "    @info \"`\$(kernel.sig.name)`: stade_adjoint takes the fixed-point reverse for this kernel and flags,", "    nothing; \"`\$(kernel.sig.name)`: stade_adjoint takes the fixed-point reverse for this kernel and flags,", ["fp_scalar_cos"]),
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
