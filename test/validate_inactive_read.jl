# Regression for v0.5.31 to v0.5.34. v0.5.31: the value-read set of site-level TBR counts an operand of a nonlinear call only when a partial that holds it
# belongs to an ACTIVE operand.
#   Defect: snap_fwd_walk! / agen_fwd_walk! gated the written variable on activity, but built the set of values read nonlinearly with no regard for
#   activity. In `u = u + w * r` and `y = x / d` the call is nonlinear, so r and x counted as read although w and d were inactive (`independents`
#   restricted), the partials that need them were never built, and every overwrite of r and x kept a snapshot nothing reads. A restricted-independents
#   adjoint of a linear multigrid cycle taped 52,130 slots per cycle (n1 = 63) for nothing.
#   Fix: snap_var_value_needed_act! / agen_var_value_needed_act! (twins), used by the two forward walks only.
# v0.5.33: activity is varied AND useful (act_useful: the variable reaches a dependent). Defect: a variable that only steers a condition, or that is
#   not a requested dependent, still got partials, so its operands kept snapshots nothing needed (the squared residual norm of a guarded solve kept the
#   residual's tape). Fix: act_analyze = act_varied AND act_useful for the adjoint and HVP stages, lin_build_stmt builds no partial for an assignment
#   to an inactive float, and the two forward walks read nothing from such a right-hand side. The tangent stages keep act_varied.
# v0.5.34: a condition is not a read. The reverse sweep replays an `if` from its recorded flag, so the walks count a condition only inside a loop the fusing
#   pass recomputes (agen_ii_recompute_stmts re-emits the condition there). The classifier never covers a loop that holds an `if`, so that branch is dead today
#   and a hand-built plan drives it (ir_plan_unit). Fix: replay/ii_plan arguments of the four walk functions, and the plan reaches the site decisions through
#   stade_site_level_tbr_check. The boundary snapshots still come from the whole-variable set, which counts conditions, so a float local with a value stack but
#   no site (uc_zinit) needs its own initial value: agen_local_primal_inits takes the value-stack variables (zero-trip loops, GPU glue code).
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
    "ir_cond"      => (indep = [:x],           expect = Dict(),                       same = Symbol[]),            # a condition reads nothing (replayed from its flag): x is read by it and by w's partial of y only, w inactive
    "ir_overwrite" => (indep = [:x],           expect = Dict(:w => 1),                same = [:w]),                # the inactive scalar w is overwritten and read next to the active x: w keeps a snapshot, x none
    "ir_local"     => (indep = [:x],           expect = Dict(:t => 1),                same = [:t]),
    "ir_array"     => (indep = [:x],           expect = Dict(:m => 1),                same = [:m]),                # an inactive coefficient array: same
    "ir_carried"   => (indep = [:x],           expect = Dict(:c => 1),                same = [:c]),                # carried inside one loop
    "ir_ifbr"      => (indep = [:x],           expect = Dict(:w => 1),                same = [:w]),                # read in one arm of a branch: w kept; x only steers the condition
    "ir_inert"     => (indep = [:x],           expect = Dict(),                       same = Symbol[]),            # feeds an inactive array only: no snapshot
    "ir_time"      => (indep = [:x],           expect = Dict(:t => 1),                same = [:t]),                # time advanced in the loop, read in a forcing                # an inactive local built from constants, overwritten every sweep: same
    "ir_firstwrite" => (indep = [:x],          expect = Dict(:t => 1),                same = [:t]),                # an inactive local first assigned inside the loop: its first push precedes any assignment
    "ur_norm"      => (indep = [:x, :r],       expect = Dict(),                       same = Symbol[]),   # the norm only steers the guard: r loses its snapshot, and rn none (a condition is replayed from its recorded flag)
    "ur_dep"       => (indep = [:x, :y, :z],   expect = Dict(),                       same = Symbol[], dep = [:y]),     # z is not a dependent: x is not read for it
    "ur_chain"     => (indep = [:x],           expect = Dict(),                       same = Symbol[]),                 # q reaches y only through p: both stay active
    "ur_cond"      => (indep = [:x],           expect = Dict(),                       same = Symbol[]),                 # s only selects the branch
    "ur_via"       => (indep = [:x],           expect = Dict(:x => 1),                same = [:x], dep = [:y]),         # control: w is not a dependent but reaches y
    "ur_arm"       => (indep = [:x],           expect = Dict(:x => 1),                same = [:x]),                     # control: q is assigned in both arms and reaches y
    "ur_live"      => (indep = [:x],           expect = Dict(:x => 1),                same = [:x]),                     # control: the accumulation reaches y[1]
    "uc_recompute" => (indep = [:x],           expect = Dict(:t => 1),                same = [:t]),                     # an if in a loop: no covered kind can hold it, so c (read only by the condition) has no snapshot
    "uc_zinit"     => (indep = [:s, :w],       expect = nothing,                      same = Symbol[]),                 # g is read only by a condition but saved at a block boundary: it needs an initial value (zero-trip loop)
    "ir_mg2d_cycle_3" => (indep = [:u1, :f1],  expect = Dict(),                       same = Symbol[]),
    "ir_mg2d_solve_3" => (indep = [:u1, :f1],  expect = Dict(),                       same = Symbol[]),            # r1 lost its four snapshots with the useful analysis, rn the two its guard condition read
)
const IR_PIN = Dict(
    "ir_mg2d_cycle_3" => (pin = Dict(:n1 => 7, :nu1 => 2, :nu2 => 1), fr = Dict(:h1 => (0.05, 0.2), :w => (0.5, 1.0)), grow = 81 + 8),
    "ir_mg2d_solve_3" => (pin = Dict(:n1 => 7, :nu1 => 2, :nu2 => 1, :maxit => 3), fr = Dict(:h1 => (0.05, 0.2), :w => (0.5, 1.0), :tol => (1e-30, 1e-29)), grow = 81 + 8),
)

# plan = nothing: fusing off, a condition reads nothing; :all: every condition read kept (what the fusing pass is classified from); a Dict: that ii plan.
# A plan is keyed by objectid of the statement vectors, so build it from the same parsed kernel that is counted: use ir_counts_kernel.
function ir_counts_kernel(k; plan = nothing)
    d = STADE.snap_value_needed_sites(k; ii_plan = plan); v = STADE.snap_site_vars(k)
    cnt = Dict{Symbol,Int}()
    for (key, need) in d
        need && (cnt[v[key]] = get(cnt, v[key], 0) + 1)
    end
    return cnt, STADE.agen_value_needed_sites(k; ii_plan = plan) == d
end
ir_counts(primal, indep, dep = nothing; plan = nothing) =
    ir_counts_kernel(STADE.parse_override_indep_dep(STADE.parse_kernel(primal), indep, dep); plan = plan)

# The replay branch cannot be reached through the classifier (it never covers a loop holding an `if`), so a hand-built plan drives it.
function ir_plan_unit(say)
    path = joinpath(@__DIR__, "inactive-read-corpus", "uc_recompute.jl")
    k = STADE.parse_override_indep_dep(STADE.parse_kernel(STADE.io_read_corpus_entry(path)), [:x], nothing)
    fors = [i for (i, s) in enumerate(k.body) if s.kind == :for]
    say(length(fors) == 2, "uc_recompute: expected two top-level loops, found $(length(fors))")
    length(fors) == 2 || return
    plan(i, kind) = Dict(STADE.agen_site_key(k.body, fors[i]) => kind)
    c(p) = begin cnt, same = ir_counts_kernel(k; plan = p); say(same, "uc_recompute: snap_ and agen_ site decisions differ under $(p isa Dict ? collect(values(p)) : p)"); get(cnt, :c, 0) end
    say(c(nothing) == 0, "uc_recompute: with fusing off a condition read must not snapshot c, got $(c(nothing))")
    say(c(:all) == 1, "uc_recompute: the conservative walk must keep the condition read of c, got $(c(:all))")
    for kind in (:reduction, :mixed, :recompute)
        say(c(plan(1, kind)) == 1, "uc_recompute: a loop covered as $kind re-emits its if, so c must snapshot, got $(c(plan(1, kind)))")
        say(c(plan(2, kind)) == 0, "uc_recompute: covering the loop WITHOUT the if must not keep c, got $(c(plan(2, kind)))")
    end
    say(c(plan(1, :independent)) == 0, "uc_recompute: an :independent loop replays from its flag, so c must not snapshot, got $(c(plan(1, :independent)))")
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

function ir_check(nm, path, ind, dep, lo, yp, say)
    primal = STADE.io_read_corpus_entry(path); kernel = STADE.parse_kernel(primal)
    base = STADE.io_read_baseline_yaml(yp); STADE.val_coerce_int_arrays!(kernel, base.values)
    mask = ir_mask(kernel, base, ind)
    x0 = STADE.val_flatten(kernel, base.values); n = length(x0); ia = base.int_args
    Random.seed!(sum(Int.(codeunits(nm))))
    seed = STADE.val_random_values_like(kernel, base.values); seedf = STADE.val_flatten(kernel, seed)
    # with restricted dependents only their components carry a seed
    dmask = dep === nothing ? trues(n) : ir_mask(kernel, base, dep)
    seedf = seedf .* dmask; seed = STADE.val_unflatten(kernel, ia, base.values, seedf)
    obs = STADE.val_compile(STADE.val_primal_observing_expr(kernel, primal))
    fev(x) = (y = STADE.val_call_primal_observed(obs, kernel, ia, STADE.val_unflatten(kernel, ia, base.values, x)); sum(y .* seedf))
    relerr(a, b) = sqrt(sum(abs2, a .- b)) / max(sqrt(sum(abs2, b)), 1e-12)
    for (kp, fz) in ((true, false), (false, false), (false, true))
        tag = "$nm keep_push_pop=$kp fuse=$fz int_lo=$lo"
        g = STADE.agen_adjoint_unrolled(primal; independents = ind, dependents = dep, keep_push_pop = kp, fuse_ii_loops = fz)
        gd = STADE.agen_adjoint_unrolled(primal; keep_push_pop = kp, fuse_ii_loops = fz)
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
        tg = STADE.stade_tangent(primal; independents = ind, dependents = dep, keep_push_pop = kp, fuse_ii_loops = fz)
        tf = STADE.val_compile(tg)
        worst = 0.0
        for _ in 1:4
            xd = STADE.val_random_values_like(kernel, base.values); xdf = STADE.val_flatten(kernel, xd) .* mask
            xdv = STADE.val_unflatten(kernel, ia, base.values, xdf)
            yb = STADE.val_random_values_like(kernel, base.values)
            yb = STADE.val_unflatten(kernel, ia, base.values, STADE.val_flatten(kernel, yb) .* dmask)
            yd = STADE.val_call_tangent(tf, kernel, ia, deepcopy(base.values), xdv)
            xbb = adj(deepcopy(base.values), yb)
            lhs = sum(STADE.val_flatten(kernel, yb) .* yd); rhs = sum(xbb .* xdf)
            worst = max(worst, abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0))
        end
        say(worst <= 1e-11, "$tag: dot-product identity, rel $worst")
        # (d) HVP against a central difference of the restricted adjoint, on the independent components
        hv = STADE.stade_hvp(primal; independents = ind, dependents = dep, keep_push_pop = kp, fuse_ii_loops = fz)
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
    v33 = ["ur_norm", "ur_dep", "ur_chain", "ur_cond", "ur_via", "ur_live", "ur_arm"]
    v34 = ["ur_norm", "uc_recompute"]
    v32 = ["ir_local", "ir_overwrite", "ir_time", "ir_array", "ir_firstwrite"]
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
        # v0.5.32: a write to an inactive float is decided by the read set alone, and the emitters see it as taped
        ("the written variable must be active again", [("snap_var_value_needed_act!(stmt.rhs, local_reads, false, (floats !== nothing && var in floats && !active_map[var]) ? Dict{Symbol,Bool}() : active_map)\n            decisions[agen_site_key(body, idx)] = (floats === nothing ? active_map[var] : var in floats) &&",
                                                       "snap_var_value_needed_act!(stmt.rhs, local_reads, false, (floats !== nothing && var in floats && !active_map[var]) ? Dict{Symbol,Bool}() : active_map)\n            decisions[agen_site_key(body, idx)] = active_map[var] &&"),
                                                      ("agen_var_value_needed_act!(stmt.rhs, local_reads, false, (floats !== nothing && var in floats && !active_map[var]) ? Dict{Symbol,Bool}() : active_map)\n            decisions[agen_site_key(body, idx)] = (floats === nothing ? active_map[var] : var in floats) &&",
                                                       "agen_var_value_needed_act!(stmt.rhs, local_reads, false, (floats !== nothing && var in floats && !active_map[var]) ? Dict{Symbol,Bool}() : active_map)\n            decisions[agen_site_key(body, idx)] = active_map[var] &&")], v32),
        ("only the snap_ twin keeps the activity gate", [("snap_var_value_needed_act!(stmt.rhs, local_reads, false, (floats !== nothing && var in floats && !active_map[var]) ? Dict{Symbol,Bool}() : active_map)\n            decisions[agen_site_key(body, idx)] = (floats === nothing ? active_map[var] : var in floats) &&",
                                                          "snap_var_value_needed_act!(stmt.rhs, local_reads, false, (floats !== nothing && var in floats && !active_map[var]) ? Dict{Symbol,Bool}() : active_map)\n            decisions[agen_site_key(body, idx)] = active_map[var] &&")], v32),
        ("the emitters never see the taped inactive variable", [("push_pop isa AbstractDict || return active_map", "return active_map")], v32),
        ("the forward sweep of the adjoint ignores the tape map", [("fwd = agen_forward_body(kernel.body, kernel.sig.kinds, tape_map, value_needed", "fwd = agen_forward_body(kernel.body, kernel.sig.kinds, active_map, value_needed")], v32),
        ("the backward sweep of the adjoint ignores the tape map", [("bwd = agen_backward_body(lin_plan, kernel.body, kernel.sig.kinds, tape_map, unsafe", "bwd = agen_backward_body(lin_plan, kernel.body, kernel.sig.kinds, active_map, unsafe")], v32),
        ("the forward sweep of the HVP ignores the tape map", [("fwd = emit_cse_stmts(agen_forward_body(kernel.body, sig.kinds, tape_map, value_needed,", "fwd = emit_cse_stmts(agen_forward_body(kernel.body, sig.kinds, active_map, value_needed,")], v32),
        ("an inactive local is not initialised before its first push", [("pinit = agen_local_primal_inits(kernel, tape_map, agen_itbr_vars(stacks), agen_value_stack_vars(stacks))\n    sinit = agen_local_shadow_inits(kernel, active_map)\n    append!(body, pinit)",
                                                                          "pinit = agen_local_primal_inits(kernel, active_map, agen_itbr_vars(stacks))\n    sinit = agen_local_shadow_inits(kernel, active_map)\n    append!(body, pinit)")], v32),
        ("usefulness is ignored: activity is varied only", [("(varied[v] && get(useful, v, false))", "varied[v]")], v33),
        ("the useful scan makes one pass", [("    bound = length(useful) + 1\n", "    bound = 1\n")], v33),
        ("a dependent does not seed usefulness", [("    for v in kernel.sig.dependents\n        useful[v] = true\n    end\n", "")], v33),
        ("a read through an array is not followed", [("changed = act_useful_mark!(expr.args[1], useful, kinds)   # indices", "changed = false   # indices")], v33),
        ("a condition makes its variables useful", [("            changed = act_useful_pass!(stmt.then, useful, kinds) || changed\n", "            changed = act_useful_mark!(stmt.cond, useful, kinds) || changed\n            changed = act_useful_pass!(stmt.then, useful, kinds) || changed\n")], v33),
        ("an assignment to an inactive float still builds partials", [("dead ? Dict{Symbol,Bool}() : active_map)", "active_map)")], v33),
        ("only the snap_ twin reads the rhs of an inactive float", [("snap_var_value_needed_act!(stmt.rhs, local_reads, false, (floats !== nothing && var in floats && !active_map[var]) ? Dict{Symbol,Bool}() : active_map)", "snap_var_value_needed_act!(stmt.rhs, local_reads, false, active_map)")], v33),
        ("only the agen_ twin reads the rhs of an inactive float", [("agen_var_value_needed_act!(stmt.rhs, local_reads, false, (floats !== nothing && var in floats && !active_map[var]) ? Dict{Symbol,Bool}() : active_map)", "agen_var_value_needed_act!(stmt.rhs, local_reads, false, active_map)")], v33),

        # v0.5.34: a condition reads a value only inside a loop the fusing pass recomputes
        ("every condition counts again (no replay)", [("snap_float_vars(kernel),\n                   ii_plan === :all, ii_plan === :all ? nothing : ii_plan)", "snap_float_vars(kernel),\n                   true, ii_plan === :all ? nothing : ii_plan)"),
                                                      ("agen_float_vars(kernel),\n                   ii_plan === :all, ii_plan === :all ? nothing : ii_plan)", "agen_float_vars(kernel),\n                   true, ii_plan === :all ? nothing : ii_plan)")], v34),
        ("a condition never counts", [("replay && snap_var_value_needed!(stmt.cond, seen, true)", "false && snap_var_value_needed!(stmt.cond, seen, true)"),
                                      ("replay && agen_var_value_needed!(stmt.cond, seen, true)", "false && agen_var_value_needed!(stmt.cond, seen, true)")], v34),
        ("a recomputed loop does not replay its conditions", [("replay_in = replay || (ii_plan !== nothing && get(ii_plan, agen_site_key(body, idx), :none) in (:reduction, :mixed, :recompute))\n            seen = snap_fwd_walk_loop!(", "replay_in = replay\n            seen = snap_fwd_walk_loop!("),
                                                              ("replay_in = replay || (ii_plan !== nothing && get(ii_plan, agen_site_key(body, idx), :none) in (:reduction, :mixed, :recompute))\n            seen = agen_fwd_walk_loop!(", "replay_in = replay\n            seen = agen_fwd_walk_loop!(")], v34),
        ("an :independent loop replays its conditions", [("replay_in = replay || (ii_plan !== nothing && get(ii_plan, agen_site_key(body, idx), :none) in (:reduction, :mixed, :recompute))\n            seen = snap_fwd_walk_loop!(", "replay_in = replay || (ii_plan !== nothing && get(ii_plan, agen_site_key(body, idx), :none) in (:reduction, :mixed, :recompute, :independent))\n            seen = snap_fwd_walk_loop!("),
                                                         ("replay_in = replay || (ii_plan !== nothing && get(ii_plan, agen_site_key(body, idx), :none) in (:reduction, :mixed, :recompute))\n            seen = agen_fwd_walk_loop!(", "replay_in = replay || (ii_plan !== nothing && get(ii_plan, agen_site_key(body, idx), :none) in (:reduction, :mixed, :recompute, :independent))\n            seen = agen_fwd_walk_loop!(")], v34),
        ("only the snap_ twin replays a recomputed loop", [("replay_in = replay || (ii_plan !== nothing && get(ii_plan, agen_site_key(body, idx), :none) in (:reduction, :mixed, :recompute))\n            seen = snap_fwd_walk_loop!(", "replay_in = replay\n            seen = snap_fwd_walk_loop!(")], v34),
        ("only the agen_ twin replays a recomputed loop", [("replay_in = replay || (ii_plan !== nothing && get(ii_plan, agen_site_key(body, idx), :none) in (:reduction, :mixed, :recompute))\n            seen = agen_fwd_walk_loop!(", "replay_in = replay\n            seen = agen_fwd_walk_loop!(")], v34),
        ("the adjoint ignores the plan for its sites", [("    # the plan is classified from the conservative sites; the emitted sites then keep a condition read only inside a loop the plan recomputes\n    snap_sites, agen_sites = stade_site_level_tbr_check(kernel; ii_plan = ii_plan)",
                                                          "    snap_sites, agen_sites = stade_site_level_tbr_check(kernel)")], v34),
        ("a float with a value stack is not initialised", [("(active_map[v] || v in stacked)", "active_map[v]")], ["uc_zinit"]),
        ("the HVP ignores the plan for its sites", [("    ii_plan = fuse_ii_loops ? stade_ii_plan_check(kernel) : nothing\n    snap_sites, agen_sites = stade_site_level_tbr_check(kernel; ii_plan = ii_plan)",
                                                      "    ii_plan = fuse_ii_loops ? stade_ii_plan_check(kernel) : nothing\n    snap_sites, agen_sites = stade_site_level_tbr_check(kernel)")], v34),
    ]
    mktempdir() do tmp
        for case in cases
            label, edits = case[1], case[2]
            isempty(get(ENV, "IR_SAB_ONLY", "")) || any(t -> occursin(t, label), split(ENV["IR_SAB_ONLY"], "|")) || continue   # run one case while developing
            kset = length(case) > 2 ? case[3] : ["ir_prod", "ir_nested", "ir_mixed", "ir_sin"]
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
                println("RESULT ", validate_inactive_read(; verbose = false, phases = (:units,), only = $(repr(kset))) ? 0 : 1)
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

# The condition of the guard reads rn; with it no longer a read, only the two block-boundary snapshots of rn remain (four before v0.5.34, eight in the HVP).
function ir_nopush(say)
    primal = STADE.io_read_corpus_entry(joinpath(@__DIR__, "inactive-read-corpus", "ur_norm.jl"))
    for kp in (true, false), fz in (false, true), mode in (:adjoint, :hvp)
        g = mode == :adjoint ? STADE.agen_adjoint_unrolled(primal; independents = [:x, :r], keep_push_pop = kp, fuse_ii_loops = fz).adjoint :
                               STADE.stade_hvp(primal; independents = [:x, :r], keep_push_pop = kp, fuse_ii_loops = fz).hvp
        src = string(g)
        n = kp ? count("push!(rn_stack", src) : count(r"rn_stack\[[^\]]*\]\s*=[^=]", src)
        want = (kp && mode == :hvp) ? 4 : 2
        say(n == want, "ur_norm $mode keep_push_pop=$kp fuse_ii_loops=$fz: $n writes to rn_stack, expected $want")
    end
end

function validate_inactive_read(; verbose = true, only = String[], report = false, phases = (:units, :sabotage))
    bad = 0; checks = 0
    say(ok, msg) = (checks += 1; ok || (bad += 1; verbose && println("NOT OK ", msg)))
    dir = joinpath(@__DIR__, "inactive-read-corpus")
    tmp = mktempdir()
    names = sort(String[f[1:end-3] for f in readdir(dir) if endswith(f, ".jl")])
    isempty(only) || (names = filter(n -> n in only, names))
    :units in phases && "uc_recompute" in names && ir_plan_unit(say)
    :units in phases && "ur_norm" in names && ir_nopush(say)
    for nm in (:units in phases ? names : String[])
        path = joinpath(dir, nm * ".jl"); c = IR_CASES[nm]
        primal = STADE.io_read_corpus_entry(path)
        cr, same_r = ir_counts(primal, c.indep, get(c, :dep, nothing))
        cd, same_d = ir_counts(primal, nothing, nothing)
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
            ir_check(nm, path, c.indep, get(c, :dep, nothing), lo, yp, say)
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
