# Parity of stade_adjoint_split with the JACC code generator (v0.5.35). The library of two runs: a V100 job (real JACC arrays) and
# validate_adjoint_split_jacc_cpu.jl (the serial stand-in JACC of validate_jacc_glue.jl).
#   For one kernel, on a draw and two seeds:
#   1  the forward half, then the backward half, equal the CPU full adjoint (every array, every shadow, the returned scalars);
#   2  a second backward sweep with the second seed, on the tape and the carry of the first recording and on the primal arrays as the forward half left
#      them, equals the CPU full adjoint with that seed. This is the replay: the forward half runs once.
#   A pass needs elements changed (the CPU reference wrote something) and two seeds that give different adjoints. A kernel with a host stack in
#   `consume` has no device form and is reported as :host_stacks.
# The device is an `env`: (mod, wrap, tohost, flags, atomic_macro). `mod` is where the generated plans are evaluated and run (Main on a device job, a module
# with the stand-in JACC on the CPU), `wrap` moves a host array to the device, `tohost` brings it back, `flags` go to every stade_jacc call.
# Needs only STADE (included by the caller); no GPU package is loaded here.

sg_device_env() = (mod = Main, wrap = x -> Main.JACC.array(x), tohost = x -> Main.JACC.to_host(x), flags = NamedTuple(),
                   atomic_macro = Expr(:., :Atomix, QuoteNode(Symbol("@atomic"))))

function sg_eval_plan!(evmod, plan)
    for h in get(plan, :helpers, Expr[]); Core.eval(evmod, h); end
    for k in plan.kernels; Core.eval(evmod, k); end
    Core.eval(evmod, plan.host)
    return Core.eval(evmod, plan.host.args[1].args[1])
end

# the positional call of a generated adjoint: a float argument is followed by its adjoint, an integer array by nothing. Returns (args, tracked).
function sg_build_call(sig, int_args::Dict, values::Dict, seed::Dict, wrap)
    args = Any[]; tracked = Dict{Symbol,Any}()
    for a in sig.args
        k = sig.kinds[a]
        if k == :scalar_int
            push!(args, int_args[a])
        elseif k == :scalar_float
            push!(args, values[a]); push!(args, seed[a])
        elseif k == :array_float
            v = wrap(deepcopy(values[a])); s = wrap(deepcopy(seed[a]))
            push!(args, v); push!(args, s); tracked[a] = (kind = :array_float, val = v, sh = s)
        else
            v = wrap(deepcopy(values[a])); push!(args, v); tracked[a] = (kind = :array_int, val = v)
        end
    end
    return args, tracked
end

sg_relerr(a, b) = maximum(abs.(a .- b) ./ max.(abs.(a), abs.(b), 1.0); init = 0.0)
sg_relerr_scalar(a, b) = abs(a - b) / max(abs(a), abs(b), 1.0)
sg_stack_extra(names, int_args, values, wrap) = Any[haskey(int_args, n) ? int_args[n] : (values[n] isa AbstractArray ? wrap(deepcopy(values[n])) : values[n]) for n in names]
function sg_init_stacks(fn, extra)
    r = Base.invokelatest(fn, extra...)
    r === nothing && return ()
    return r isa Tuple ? r : (r,)
end

# the worst relative error over every array and shadow, and the number of elements the CPU reference changed
function sg_compare(cpu_tracked::Dict, dev_tracked::Dict, tohost, pre::Dict)
    m = 0.0; changed = 0
    for (a, ct) in cpu_tracked
        dt = dev_tracked[a]
        m = max(m, sg_relerr(ct.val, tohost(dt.val))); changed += count(!=(0.0), ct.val .- pre[a].val)
        if ct.kind == :array_float
            m = max(m, sg_relerr(ct.sh, tohost(dt.sh))); changed += count(!=(0.0), ct.sh .- pre[a].sh)
        end
    end
    return m, changed
end

# a copy of a generated function under another name (the full adjoint and the split share the name `initstacks_NAME_b`)
function sg_rename(ex::Expr, newname::Symbol)
    e = deepcopy(ex); e.args[1].args[1] = newname
    return e
end

sg_rets(n_ret::Int, r) = n_ret == 0 ? () : (n_ret == 1 ? (r,) : r)

# the CPU full adjoint on one seed: the call's arrays (the reference), a copy of them before the call, the returned scalars
function sg_cpu_run(sig, int_args, vals, seed, cpu_init, cpu_gen, names_full)
    args, tr = sg_build_call(sig, int_args, vals, seed, identity)
    pre = Dict{Symbol,Any}(a => (ct.kind == :array_int ? (val = deepcopy(ct.val),) : (val = deepcopy(ct.val), sh = deepcopy(ct.sh))) for (a, ct) in tr)
    stk = sg_init_stacks(cpu_init, sg_stack_extra(names_full, int_args, vals, identity))
    ret = Base.invokelatest(cpu_gen, args..., stk...)
    return (tracked = tr, pre = pre, ret = ret)
end

function sg_finite_run(c)
    for (a, ct) in c.tracked
        all(isfinite, ct.val) || return false
        ct.kind == :array_float && !all(isfinite, ct.sh) && return false
    end
    return true
end

# the two seeds give different adjoints, or the replay compares the same numbers twice
function sg_seeds_differ(c1, c2)
    for (a, ct) in c1.tracked
        ct.kind == :array_float && ct.sh != c2.tracked[a].sh && return true
    end
    return false
end

sg_scalar_err(n_ret::Int, cpu_ret, dev_ret) = (m = 0.0; for (x, y) in zip(sg_rets(n_ret, cpu_ret), sg_rets(n_ret, dev_ret)); m = max(m, sg_relerr_scalar(x, y)); end; m)

# one trial: record once with seed s1, replay with seed s2; returns (worst relative error, elements changed), or nothing when the draw gives no signal
function sg_trial(P, int_args, vals, s1, s2)
    sig = P[:sig]
    # a draw whose arrays and integers disagree fails in the CPU reference (it checks bounds): another draw
    c1 = try sg_cpu_run(sig, int_args, vals, s1, P[:cpu_init], P[:cpu_gen], P[:names_full]) catch; return nothing end
    c2 = try sg_cpu_run(sig, int_args, vals, s2, P[:cpu_init], P[:cpu_gen], P[:names_full]) catch; return nothing end
    sg_finite_run(c1) && sg_finite_run(c2) && sg_seeds_differ(c1, c2) || return nothing
    wrap = P[:wrap]; tohost = P[:tohost]
    # record once
    dev_args, dev_tr = sg_build_call(sig, int_args, vals, s1, wrap)
    stk = sg_init_stacks(P[:dev_init], sg_stack_extra(P[:names_sp], int_args, vals, wrap))
    carry = Any[]
    if P[:carryfn] !== nothing
        for x in Base.invokelatest(P[:carryfn]); push!(carry, isempty(x) ? x : wrap(x)); end
    end
    Base.invokelatest(P[:dev_fwd], dev_args..., stk..., carry...)
    after_fwd = Dict{Symbol,Any}(a => tohost(dt.val) for (a, dt) in dev_tr)
    ret1 = Base.invokelatest(P[:dev_bwd], dev_args..., stk..., carry...)
    m, n = sg_compare(c1.tracked, dev_tr, tohost, c1.pre)
    m = max(m, sg_scalar_err(P[:n_ret], c1.ret, ret1))
    # replay: the second seed, the same tape and carry, the primal arrays as the forward half left them
    rargs = Any[]; rtr = Dict{Symbol,Any}()
    for a in sig.args
        k = sig.kinds[a]
        if k == :scalar_int
            push!(rargs, int_args[a])
        elseif k == :scalar_float
            push!(rargs, vals[a]); push!(rargs, s2[a])
        elseif k == :array_float
            v = wrap(copy(after_fwd[a])); sh = wrap(deepcopy(s2[a]))
            push!(rargs, v); push!(rargs, sh); rtr[a] = (kind = :array_float, val = v, sh = sh)
        else
            v = wrap(copy(after_fwd[a])); push!(rargs, v); rtr[a] = (kind = :array_int, val = v)
        end
    end
    ret2 = Base.invokelatest(P[:dev_bwd], rargs..., stk..., carry...)
    m2, n2 = sg_compare(c2.tracked, rtr, tohost, c2.pre)
    m2 = max(m2, sg_scalar_err(P[:n_ret], c2.ret, ret2))
    err = (isnan(m) || isnan(m2)) ? Inf : max(m, m2)
    # the device full adjoint with the first seed: its error against the same CPU reference
    ferr = NaN
    if P[:dev_full_gen] !== nothing
        try
            fargs, ftr = sg_build_call(sig, int_args, vals, s1, wrap)
            fstk = sg_init_stacks(P[:dev_full_init], sg_stack_extra(P[:names_full], int_args, vals, wrap))
            fret = Base.invokelatest(P[:dev_full_gen], fargs..., fstk...)
            ferr, _ = sg_compare(c1.tracked, ftr, tohost, c1.pre)
            ferr = max(ferr, sg_scalar_err(P[:n_ret], c1.ret, fret))
        catch
            ferr = NaN
        end
    end
    return (err, n + n2, ferr)
end

# the plans of one kernel: the CPU full adjoint, the device plans of the tape allocator and of the two halves. Returns a NamedTuple, or a status tuple.
function sg_setup(primal, kernel, env)
    sig = kernel.sig
    full = try STADE.stade_adjoint(primal; keep_push_pop = false) catch; return (:refused, "", NaN, 0) end
    sp = try STADE.stade_adjoint_split(primal) catch e; return (:split_refused, first(split(sprint(showerror, e), "\n")), NaN, 0) end
    isempty(sp.consume) || return (:host_stacks, "consume=" * join(sp.consume, ","), NaN, 0)
    # CPU reference: the full adjoint, evaluated in Main (an anonymous Module() in a function crashes the Julia 1.10 compiler here); the stand-in lives in its own module
    cpu_init = STADE.val_compile(full.initstacks); cpu_gen = STADE.val_compile(full.adjoint)
    names_full = Symbol[a for a in full.initstacks.args[1].args[2:end]]
    fl = env.flags
    plan_init = STADE.stade_jacc(sp.initstacks; fl...)
    initmap = Dict{Symbol,Expr}(sp.initstacks.args[1].args[1] => sp.initstacks)
    plan_f = try STADE.stade_jacc(sp.forward; fl..., init_hosts = initmap) catch e; return (:gen_error, "forward: " * first(split(sprint(showerror, e), "\n")), NaN, 0) end
    plan_b = try STADE.stade_jacc(sp.backward; fl..., init_hosts = initmap) catch e; return (:gen_error, "backward: " * first(split(sprint(showerror, e), "\n")), NaN, 0) end
    dev_init = sg_eval_plan!(env.mod, plan_init); dev_fwd = sg_eval_plan!(env.mod, plan_f); dev_bwd = sg_eval_plan!(env.mod, plan_b)
    # the full adjoint on the device, the same draw, to tell a defect of the split from one of the JACC path itself
    fi_name = Symbol(string(full.initstacks.args[1].args[1]) * "full"); fg_name = Symbol(string(full.adjoint.args[1].args[1]) * "full")
    full_init_ex = sg_rename(full.initstacks, fi_name); full_gen_ex = sg_rename(full.adjoint, fg_name)
    dev_full_init = nothing; dev_full_gen = nothing
    try
        pfi = STADE.stade_jacc(full_init_ex; fl...)
        pfg = STADE.stade_jacc(full_gen_ex; fl..., init_hosts = Dict{Symbol,Expr}(fi_name => full_init_ex))
        dev_full_init = sg_eval_plan!(env.mod, pfi); dev_full_gen = sg_eval_plan!(env.mod, pfg)
    catch
        dev_full_init = nothing; dev_full_gen = nothing
    end
    names_sp = Symbol[a for a in sp.initstacks.args[1].args[2:end]]
    carryfn = nothing
    if sp.carry_init !== nothing
        Core.eval(env.mod, sp.carry_init); carryfn = Core.eval(env.mod, sp.carry_init.args[1].args[1])
    end
    atomic_sites = 0
    for k in vcat(plan_f.kernels, plan_b.kernels); atomic_sites += STADE.val_count_atomic_sites(k, env.atomic_macro); end
    n_ret = 0
    for a in sig.args; sig.kinds[a] == :scalar_float && (n_ret += 1); end
    # a Dict, not a NamedTuple: a NamedTuple type that holds function types of an anonymous module crashes the Julia 1.10 compiler
    return Dict{Symbol,Any}(:sig => sig, :cpu_init => cpu_init, :cpu_gen => cpu_gen, :names_full => names_full, :dev_init => dev_init, :dev_fwd => dev_fwd, :dev_bwd => dev_bwd, :dev_full_init => dev_full_init, :dev_full_gen => dev_full_gen,
                            :names_sp => names_sp, :carryfn => carryfn, :n_ret => n_ret, :rtol => 2.5e-14 * max(1, atomic_sites), :wrap => env.wrap, :tohost => env.tohost,
                            :detail => "carry=$(length(sp.carry_f) + length(sp.carry_i)) restore=$(length(sp.restore))")
end

# one kernel; returns (status, detail, max_rel_err, elements_changed)
function sg_kernel(path; env = sg_device_env(), trials::Int = 2, int_lo::Int = 3, int_hi::Int = 5, attempts::Int = 8)
    primal = STADE.io_read_corpus_entry(path)
    kernel = STADE.parse_kernel(primal)
    P = sg_setup(primal, kernel, env)
    P isa Tuple && return P   # a status tuple: refused, split_refused, host_stacks, gen_error
    max_err = 0.0; changed = 0; ints = ""; full_err = NaN
    for attempt in 1:attempts
        baseline = try STADE.val_generate_baseline(kernel, primal; self_check = false, int_lo = int_lo, int_hi = int_hi) catch; return (:nodraw, "", NaN, 0) end
        STADE.val_coerce_int_arrays!(kernel, baseline.values)
        int_args = baseline.int_args; vals = baseline.values
        seeds = [STADE.val_random_values_like(kernel, vals) for _ in 1:max(trials, 2)]
        a_err = 0.0; a_changed = 0; a_ferr = 0.0; good = true
        for t in 1:trials
            r = sg_trial(P, int_args, vals, seeds[t], seeds[mod1(t + 1, length(seeds))])
            if r === nothing
                good = false; break
            end
            a_err = max(a_err, r[1]); a_changed += r[2]; a_ferr = max(a_ferr, isnan(r[3]) ? 0.0 : r[3])
        end
        good || continue
        max_err = a_err; changed = a_changed; full_err = a_ferr
        ints = join(["$k=$v" for (k, v) in sort(collect(int_args))], ",")
        break
    end
    status = changed == 0 ? :VACUOUS : (max_err <= P[:rtol] ? :ok : :FAIL)
    return (status, P[:detail] * " rtol=$(round(P[:rtol], sigdigits = 3)) full_err=$(round(full_err, sigdigits = 3)) ints=" * ints, max_err, changed)
end

# every kernel of `dir` (or those named in `only`); prints one line per kernel and a summary, returns the rows
function validate_adjoint_split_gpu(dir::String; only = String[], env = sg_device_env(), trials::Int = 2, int_lo::Int = 3, int_hi::Int = 5, verbose::Bool = true)
    rows = Any[]
    for f in sort(readdir(dir))
        endswith(f, ".jl") || continue
        nm = f[1:end-3]
        isempty(only) || nm in only || continue
        r = try sg_kernel(joinpath(dir, f); env = env, trials = trials, int_lo = int_lo, int_hi = int_hi) catch e
            (:error, first(split(sprint(showerror, e), "\n")), NaN, 0)
        end
        push!(rows, (nm, r...))
        verbose && println(rpad("$nm [split/jacc]", 40), " ", r[1], isnan(r[3]) ? "" : "  max_rel_err=$(round(r[3], sigdigits = 4))", "  changed=$(r[4])", isempty(r[2]) ? "" : "  " * r[2])
    end
    bad = count(r -> !(r[2] in (:ok, :refused, :host_stacks, :nodraw)), rows)
    verbose && println("\n", length(rows) - bad, "/", length(rows), " checks passed", bad == 0 ? "" : "   *** $bad NOT OK ***")
    return rows
end
