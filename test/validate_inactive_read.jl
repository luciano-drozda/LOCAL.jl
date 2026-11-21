# Regression for v0.5.31: the value-read set of site-level TBR counts an operand of a nonlinear call only when a partial that holds it
# belongs to an ACTIVE operand.
#   Defect: snap_fwd_walk! / agen_fwd_walk! gated the written variable on activity, but built the set of values read nonlinearly with no regard for
#   activity. In `u = u + w * r` and `y = x / d` the call is nonlinear, so r and x counted as read although w and d were inactive (`independents`
#   restricted), the partials that need them were never built, and every overwrite of r and x kept a snapshot nothing reads. A restricted-independents
#   adjoint of a linear multigrid cycle taped 52,130 slots per cycle (n1 = 63) for nothing.
#   Fix: snap_var_value_needed_act! / agen_var_value_needed_act! (twins), used by the two forward walks only.
# Part 1 counts the per-site decisions per variable, with the independents restricted and with the defaults (shapes that must keep their snapshots keep them).
# Part 2 checks the restricted-independents adjoint on the independent components against (a) a central finite difference along a random direction,
#   (b) the default-independents adjoint, (c) the exact tangent-adjoint dot-product identity, (d) the HVP against a finite difference of the adjoint,
#   in both stack modes, with and without fuse_ii_loops, with a zero-trip draw (int_lo = 0). No device is used.
isdefined(@__MODULE__, :STADE) || include(joinpath(dirname(@__DIR__), "src", "STADE.jl"))
include(joinpath(@__DIR__, "sabotage_run.jl"))
using Random

# kernel => (independents, sites per variable that snapshot with them restricted)
# a variable absent from the Dict has 0 expected. `same` names the variables whose count must equal the default-independents count.
const IR_CASES = Dict(
    "ir_prod"      => (indep = [:u, :f],       expect = Dict(),                       same = Symbol[]),            # r read only by the inactive w's partial: no tape
    "ir_quot"      => (indep = [:x, :c],       expect = Dict(),                       same = Symbol[]),            # x read only by the inactive d's partial: no tape
    "ir_nested"    => (indep = [:x, :z],       expect = Dict(:x => 1, :z => 1),       same = [:x, :z]),            # value of w * x feeds the partial of z: kept
    "ir_sin"       => (indep = [:x],           expect = Dict(:x => 1),                same = [:x]),                # partial of sin needs x: kept
    "ir_mixed"     => (indep = [:x, :z],       expect = Dict(:x => 1),                same = [:x]),                # second product needs x: kept
    "ir_maxmin"    => (indep = [:x],           expect = Dict(:x => 1),                same = [:x]),                # the selection compares x: kept
    "ir_cond"      => (indep = [:x],           expect = Dict(:x => 1),                same = [:x]),                # a condition always counts: kept
    "ir_mg2d_cycle_3" => (indep = [:u1, :f1],  expect = Dict(),                       same = Symbol[]),
    "ir_mg2d_solve_3" => (indep = [:u1, :f1],  expect = Dict(:r1 => 4, :rn => 2),     same = Symbol[]),            # the residual norm only steers the guard, yet its partials keep r1 and rn: the activity analysis has no "useful" pass
)
const IR_PIN = Dict(
    "ir_mg2d_cycle_3" => (pin = Dict(:n1 => 7, :nu1 => 2, :nu2 => 1), fr = Dict(:h1 => (0.05, 0.2), :w => (0.5, 1.0)), grow = 81 + 8),
    "ir_mg2d_solve_3" => (pin = Dict(:n1 => 7, :nu1 => 2, :nu2 => 1, :maxit => 3), fr = Dict(:h1 => (0.05, 0.2), :w => (0.5, 1.0), :tol => (1e-30, 1e-29)), grow = 81 + 8),
)

function ir_counts(primal, indep)
    k = STADE.parse_override_indep_dep(STADE.parse_kernel(primal), indep, nothing)
    d = STADE.snap_value_needed_sites(k); v = STADE.snap_site_vars(k)
    cnt = Dict{Symbol,Int}()
    for (key, need) in d
        need && (cnt[v[key]] = get(cnt, v[key], 0) + 1)
    end
    return cnt, STADE.agen_value_needed_sites(k) == d
end

# unit vector over the flattened float arguments named in `names`
ir_mask(kernel, base, names) = begin
    m = falses(length(STADE.val_flatten(kernel, base.values)))
    for nm in names
        v = STADE.val_zeros_like(kernel, base.values)
        v[nm] isa AbstractArray ? (v[nm] .= 1.0) : (v[nm] = 1.0)
        m .|= STADE.val_flatten(kernel, v) .!= 0
    end
    m
end

function ir_check(nm, path, ind, lo, yp, say)
    primal = STADE.io_read_corpus_entry(path); kernel = STADE.parse_kernel(primal)
    base = STADE.io_read_baseline_yaml(yp); STADE.val_coerce_int_arrays!(kernel, base.values)
    mask = ir_mask(kernel, base, ind)
    x0 = STADE.val_flatten(kernel, base.values); n = length(x0); ia = base.int_args
    Random.seed!(sum(Int.(codeunits(nm))))
    seed = STADE.val_random_values_like(kernel, base.values); seedf = STADE.val_flatten(kernel, seed)
    obs = STADE.val_compile(STADE.val_primal_observing_expr(kernel, primal))
    fev(x) = (y = STADE.val_call_primal_observed(obs, kernel, ia, STADE.val_unflatten(kernel, ia, base.values, x)); sum(y .* seedf))
    relerr(a, b) = sqrt(sum(abs2, a .- b)) / max(sqrt(sum(abs2, b)), 1e-12)
    for (kp, fz) in ((true, false), (false, false), (false, true))
        tag = "$nm keep_push_pop=$kp fuse=$fz int_lo=$lo"
        g = STADE.stade_adjoint(primal; independents = ind, keep_push_pop = kp, fuse_ii_loops = fz)
        gd = STADE.stade_adjoint(primal; keep_push_pop = kp, fuse_ii_loops = fz)
        sn(o) = kp ? Symbol[] : Symbol[a for a in o.initstacks.args[1].args[2:end]]
        # both codes carry the same function names, so the default one is compiled and run first and the restricted one defined afterwards
        xbd = STADE.val_call_adjoint(STADE.val_compile(gd.adjoint), STADE.val_compile(gd.initstacks), kernel, ia, deepcopy(base.values), seed; stack_arg_names = sn(gd))
        af = STADE.val_compile(g.adjoint); isf = STADE.val_compile(g.initstacks)
        adj(vals, sd) = STADE.val_call_adjoint(af, isf, kernel, ia, vals, sd; stack_arg_names = sn(g))
        xb = adj(deepcopy(base.values), seed)
        # (b) against the default-independents adjoint, on the independent components
        say(relerr(xb[mask], xbd[mask]) <= 1e-12, "$tag: restricted adjoint differs from the default adjoint, rel $(relerr(xb[mask], xbd[mask]))")
        # (a) central finite difference along random directions inside the independent subspace
        worst = 0.0
        for _ in 1:6
            d = STADE.val_randn(n) .* mask; d ./= max(sqrt(sum(abs2, d)), 1e-300)
            fd = (fev(x0 .+ 1e-6 .* d) - fev(x0 .- 1e-6 .* d)) / 2e-6
            ad = sum(xb .* d)
            worst = max(worst, abs(fd - ad) / max(abs(fd), abs(ad), 1e-12))
        end
        say(worst <= 1e-4, "$tag: finite difference vs restricted adjoint, rel $worst")
        # (c) exact dot-product identity, tangent and adjoint both restricted
        tg = STADE.stade_tangent(primal; independents = ind, keep_push_pop = kp, fuse_ii_loops = fz)
        tf = STADE.val_compile(tg)
        worst = 0.0
        for _ in 1:4
            xd = STADE.val_random_values_like(kernel, base.values); xdf = STADE.val_flatten(kernel, xd) .* mask
            xdv = STADE.val_unflatten(kernel, ia, base.values, xdf)
            yb = STADE.val_random_values_like(kernel, base.values)
            yd = STADE.val_call_tangent(tf, kernel, ia, deepcopy(base.values), xdv)
            xbb = adj(deepcopy(base.values), yb)
            lhs = sum(STADE.val_flatten(kernel, yb) .* yd); rhs = sum(xbb .* xdf)
            worst = max(worst, abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0))
        end
        say(worst <= 1e-11, "$tag: dot-product identity, rel $worst")
        # (d) HVP against a central difference of the restricted adjoint, on the independent components
        hv = STADE.stade_hvp(primal; independents = ind, keep_push_pop = kp, fuse_ii_loops = fz)
        hf = STADE.val_compile(hv.hvp)
        d = STADE.val_randn(n) .* mask; d ./= max(sqrt(sum(abs2, d)), 1e-300)
        dv = STADE.val_unflatten(kernel, ia, STADE.val_zeros_like(kernel, base.values), d)
        dseed = STADE.val_zeros_like(kernel, base.values)
        h = STADE.val_call_hv(hf, isf, kernel, ia, deepcopy(base.values), seed, dv, dseed; stack_arg_names = sn(g))
        gp = adj(STADE.val_unflatten(kernel, ia, base.values, x0 .+ 1e-6 .* d), seed)
        gm = adj(STADE.val_unflatten(kernel, ia, base.values, x0 .- 1e-6 .* d), seed)
        fdh = (gp .- gm) ./ 2e-6
        say(relerr(h[mask], fdh[mask]) <= 1e-4, "$tag: HVP vs finite difference of the adjoint, rel $(relerr(h[mask], fdh[mask]))")
    end
end

# Sabotage: each case rewrites the guard in a copy of STADE.jl (every anchor must occur exactly once) and some check must fail.
function ir_sabotage(; verbose = true)
    bad = 0
    text = read(joinpath(dirname(@__DIR__), "src", "STADE.jl"), String)
    sn = "ni = parts === nothing || any(j -> actv[j] && snap_expr_holds(parts[j], a), eachindex(args))"
    ag = "ni = parts === nothing || any(j -> actv[j] && agen_expr_holds(parts[j], a), eachindex(args))"
    cases = [
        ("every operand counts as read", [(sn, "ni = true"), (ag, "ni = true")]),
        ("no operand counts as read", [(sn, "ni = false"), (ag, "ni = false")]),
        ("a partial counts only for its own operand", [("actv[j] && snap_expr_holds(parts[j], a)", "actv[j] && j == i && snap_expr_holds(parts[j], a)"),
                                                       ("actv[j] && agen_expr_holds(parts[j], a)", "actv[j] && j == i && agen_expr_holds(parts[j], a)")]),
        ("the partial of an inactive operand counts", [("actv[j] && snap_expr_holds(parts[j], a)", "snap_expr_holds(parts[j], a)"),
                                                       ("actv[j] && agen_expr_holds(parts[j], a)", "agen_expr_holds(parts[j], a)")]),
        ("an array element is never active", [("snap_expr_active(e, am) = e isa Symbol ? get(am, e, false) :\n    (e isa Expr && e.head == :ref) ? get(am, e.args[1], false) :",
                                               "snap_expr_active(e, am) = e isa Symbol ? get(am, e, false) :\n    (e isa Expr && e.head == :ref) ? false :"),
                                              ("agen_expr_active(e, am) = e isa Symbol ? get(am, e, false) :\n    (e isa Expr && e.head == :ref) ? get(am, e.args[1], false) :",
                                               "agen_expr_active(e, am) = e isa Symbol ? get(am, e, false) :\n    (e isa Expr && e.head == :ref) ? false :")]),
        ("a needed context is filtered again", [("if needed || snap_linear_call(op, args)", "if snap_linear_call(op, args)"),
                                                ("if needed || agen_linear_call(op, args)", "if agen_linear_call(op, args)")]),
        ("only the snap_ twin is edited", [(sn, "ni = false")]),
        ("only the agen_ twin is edited", [(ag, "ni = false")]),
    ]
    mktempdir() do tmp
        for (label, edits) in cases
            t = text; okanchor = true
            for (old, new) in edits
                n = length(findall(old, t))
                n == 1 || (okanchor = false; verbose && println("NOT OK sabotage \"$label\": anchor found $n times"); bad += 1)
                t = replace(t, old => new; count = 1)
            end
            okanchor || continue
            get(ENV, "SAB_ANCHORS_ONLY", "") == "1" && continue
            path = joinpath(tmp, "STADE_sab.jl"); write(path, t)
            script = """
                include($(repr(path)))
                cd($(repr(@__DIR__)))
                include($(repr(joinpath(@__DIR__, "validate_inactive_read.jl"))))
                println("RESULT ", validate_inactive_read(; verbose = false, phases = (:units,), only = ["ir_prod", "ir_nested", "ir_mixed", "ir_sin"]) ? 0 : 1)
            """
            out = sab_read(script)
            m = match(r"RESULT (\d)", out)
            caught = out == "ERR" || m === nothing || m.captures[1] == "1"
            caught || (bad += 1)
            verbose && println(caught ? "ok     " : "NOT OK ", "sabotage \"$label\": ", caught ? "a check fails" : "NOT CAUGHT")
        end
    end
    return bad == 0
end

function validate_inactive_read(; verbose = true, only = String[], report = false, phases = (:units, :sabotage))
    bad = 0; checks = 0
    say(ok, msg) = (checks += 1; ok || (bad += 1; verbose && println("NOT OK ", msg)))
    dir = joinpath(@__DIR__, "inactive-read-corpus")
    tmp = mktempdir()
    names = sort(String[f[1:end-3] for f in readdir(dir) if endswith(f, ".jl")])
    isempty(only) || (names = filter(n -> n in only, names))
    for nm in (:units in phases ? names : String[])
        path = joinpath(dir, nm * ".jl"); c = IR_CASES[nm]
        primal = STADE.io_read_corpus_entry(path)
        cr, same_r = ir_counts(primal, c.indep)
        cd, same_d = ir_counts(primal, nothing)
        report && println(rpad(nm, 18), " restricted: ", sort(collect(cr)), "  default: ", sort(collect(cd)))
        say(same_r && same_d, "$nm: snap_ and agen_ site decisions differ")
        if c.expect !== nothing
            for v in union(keys(c.expect), keys(cr))
                say(get(cr, v, 0) == get(c.expect, v, 0), "$nm: sites of $v that snapshot, restricted = $(get(cr, v, 0)), expected $(get(c.expect, v, 0))")
            end
        end
        for v in c.same
            say(get(cr, v, 0) == get(cd, v, 0), "$nm: sites of $v restricted $(get(cr, v, 0)) differ from default $(get(cd, v, 0)) (must be kept)")
        end
        # the restriction can only remove snapshots, never add one
        for (v, k) in cr
            say(k <= get(cd, v, 0), "$nm: restricted independents ADD a snapshot of $v ($k > $(get(cd, v, 0)))")
        end
        for lo in (3, 0)
            yp = joinpath(tmp, "$nm.lo$lo.yaml")
            pin = get(IR_PIN, nm, nothing)
            lo == 0 && pin !== nothing && continue   # sizes are pinned for the multigrid kernels
            if pin === nothing
                STADE.stade_generate_baseline_file(path; yaml_path = yp, int_lo = lo, int_hi = 5)
            else
                STADE.stade_generate_baseline_file(path; yaml_path = yp, int_pin = pin.pin, float_range = pin.fr, grow_max = pin.grow)
            end
            ir_check(nm, path, c.indep, lo, yp, say)
        end
    end
    verbose && println("validate_inactive_read: $(checks - bad)/$checks ok")
    ok = bad == 0
    :sabotage in phases && (ok = ir_sabotage(; verbose = verbose) && ok)
    return ok
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_inactive_read(; only = String[a for a in ARGS if !startswith(a, "--")], report = "--report" in ARGS, phases = "--no-sabotage" in ARGS ? (:units,) : (:units, :sabotage)) ? 0 : 1)
end
