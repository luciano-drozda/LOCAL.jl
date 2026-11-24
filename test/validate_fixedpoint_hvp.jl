# The HVP of a kernel with a solve loop (a counted loop with a guard) on the fixed-point path: stade_hvp with keep_push_pop = false and no fusion.
# The fixed-point HVP keeps one slot of tape per instance and repeats the reverse of the last active iteration (its sweeps) until the state adjoint AND its tangent settle,
# so the tape does not grow with the cap. The unrolled HVP (agen_hvp_unrolled) differentiates the iterations that ran: the oracle for a converged loop.
# For every case of validate_fixedpoint.jl (the corpus kernels and the generated shapes of fp_shapes.jl), in four regimes:
#   converged  the fixed-point HVP against the unrolled one to 1e-8 for three seedings (seed only, tangent seed only, both: the second one makes the state adjoint zero while its
#              tangent is not, so a stop rule that ignores the tangent stops after one sweep), the arrays the loops do not write equal bit for bit, the sweeps ran, no flag is
#              set, and the tape has the same number of elements for twice the cap (and fewer than the unrolled one)
#   zero       the guard is closed before the first iteration: no sweep, the two HVPs equal
#   cap        maxit = 3: the HVP raises the cut-off message, fp_info = [_, 0, _, >= 1, 2]
#   nocap      maxit = 0: no raise, fp_info all zero
# and the dispatch: stade_hvp takes the fixed-point path only with keep_push_pop = false and no fusion, a refused loop keeps the unrolled HVP, and no corpus kernel without a
# solve loop is affected.
# usage: julia validate_fixedpoint_hvp.jl [kernel ...] [--sabotage]
include(joinpath(@__DIR__, "validate_fixedpoint_tangent_hvp.jl"))

# one call of a generated HVP: the flat output (like val_call_hv), the stacks (the last one is fp_info for a fixed-point HVP), the arrays after the call, what it raised
function fph_call(fn, initfn, kernel, int_args, vals, seed, D, dseed, names; wrap = identity, tohost = identity)
    sig = kernel.sig
    extra = Any[haskey(int_args, n) ? int_args[n] : wrap(deepcopy(vals[n])) for n in names]
    stacks = STADE.val_init_stacks(initfn, extra)
    args = STADE.val_hvp_call_args(sig, int_args, vals, seed, D, dseed, stacks)
    # a device run: the arrays of the kernel move to the device, the stacks (fp_info included) already are there
    for i in 1:(length(args) - length(stacks)); args[i] isa AbstractArray && eltype(args[i]) === Float64 && (args[i] = wrap(args[i])); end
    raised = ""; ret = nothing
    try ret = Base.invokelatest(fn, args...) catch e; raised = first(split(sprint(showerror, e), "\n")) end
    isempty(raised) || return (out = nothing, stacks = stacks, args = args, raised = raised)
    scalar_args = [a for a in sig.args if sig.kinds[a] == :scalar_float]
    ret_tuple = isempty(scalar_args) ? () : ret
    hv_of = Dict{Symbol,Any}(a => ret_tuple[2i] for (i, a) in enumerate(scalar_args))
    n_lead = count(a -> sig.kinds[a] == :scalar_int, sig.args) + count(a -> sig.kinds[a] == :array_int, sig.args) + 2 * count(a -> sig.kinds[a] in (:scalar_float, :array_float), sig.args)
    out = Dict{Symbol,Any}(); i = 0
    for a in sig.args
        sig.kinds[a] in (:scalar_float, :array_float) || continue
        out[a] = sig.kinds[a] == :scalar_float ? hv_of[a] : tohost(args[n_lead + 2i + 2])
        i += 1
    end
    return (out = STADE.val_flatten(kernel, out), stacks = stacks, args = args, raised = "")
end

fph_info(r; tohost = identity) = Float64.(tohost(r.stacks[end]))
# elements of the stacks that initstacks allocates, `fp_info` excluded
fph_tape(initfn, names, int_args, vals) = sum(length, STADE.val_init_stacks(initfn, Any[haskey(int_args, n) ? int_args[n] : deepcopy(vals[n]) for n in names])[1:end-1])

# one case: the verdicts of every regime
function fph_case(say, name, primal, pins, floats, gs, prep, dprep; verbose = true, seed::Int = 20261122)
    kernel = STADE.parse_kernel(primal)
    sig = kernel.sig
    Random.seed!(seed)
    bl = STADE.val_generate_baseline(kernel, primal; self_check = false, int_pin = copy(pins), int_lo = 3, int_hi = 5, grow_start = gs, grow_max = 64)
    STADE.val_coerce_int_arrays!(kernel, bl.values)
    int0 = merge(bl.int_args, pins); vals0 = bl.values
    for (k, v) in floats; vals0[k] = v; end
    prep === nothing || prep(vals0)
    fp = fpv_logged(() -> STADE.stade_hvp(primal; keep_push_pop = false)).value
    say(last(fp.hvp.args[1].args) === :fp_info, "$name: stade_hvp with keep_push_pop = false does not take the fixed-point path")
    last(fp.hvp.args[1].args) === :fp_info || return nothing
    un = STADE.agen_hvp_unrolled(primal; keep_push_pop = false)
    # the HVP drops the saves the adjoint drops (the converged-state rule), not more and not fewer
    fa = STADE.agen_adjoint_fixedpoint(primal)
    say(length(fp.converged) == length(fa.converged), "$name: the HVP converts $(length(fp.converged)) solve arrays to the converged state, the adjoint $(length(fa.converged))")
    fpn = Symbol[a for a in fp.initstacks.args[1].args[2:end]]; unn = Symbol[a for a in un.initstacks.args[1].args[2:end]]
    ffn = STADE.val_compile(sg_rename(fp.hvp, :fph_hvp)); fin = STADE.val_compile(sg_rename(fp.initstacks, :fph_init))
    ufn = STADE.val_compile(sg_rename(un.hvp, :fph_uhvp)); uin = STADE.val_compile(sg_rename(un.initstacks, :fph_uinit))
    loop_arrays = Set{Symbol}(a for l in fp.loops for a in l.arrays)
    zeros_() = STADE.val_zeros_like(kernel, vals0)
    mask = zeros_(); mask[:tol] = 1.0
    mi = findfirst(!=(0.0), STADE.val_flatten(kernel, mask))
    function direction()
        d = STADE.val_randn(length(STADE.val_flatten(kernel, vals0))); d[mi] = 0.0
        D = STADE.val_unflatten(kernel, int0, vals0, d)
        dprep === nothing || dprep(D)
        return D
    end
    function both(int_args, vals, seedv, D, dseed)
        a = fph_call(ffn, fin, kernel, int_args, vals, seedv, D, dseed, fpn)
        b = fph_call(ufn, uin, kernel, int_args, vals, seedv, D, dseed, unn)
        return (a, b)
    end
    # converged: three seedings
    Random.seed!(seed + 1)
    seeds = [("seed", STADE.val_random_values_like(kernel, vals0), zeros_()), ("tangent seed", zeros_(), STADE.val_random_values_like(kernel, vals0)),
             ("both", STADE.val_random_values_like(kernel, vals0), STADE.val_random_values_like(kernel, vals0))]
    worst = 0.0; vacuous = true; prim = 0.0; sweeps = 0.0
    for (label, sv, dsv) in seeds
        (a, b) = both(int0, vals0, sv, direction(), dsv)
        say(isempty(a.raised), "$name converged ($label): the fixed-point HVP raised ($(a.raised))")
        isempty(a.raised) || continue
        e = fpt_rel(a.out, b.out); worst = max(worst, e); vacuous = vacuous && maximum(abs, a.out) == 0.0
        info = fph_info(a); sweeps = max(sweeps, info[2])
        say(e <= 1e-8, "$name converged ($label): fixed-point HVP against the unrolled one, err=$e")
        say(info[2] >= 1 && info[1] >= 1 && info[4] == 0 && info[5] == 0, "$name converged ($label): sweeps ran, no flag ($info)")
        # the arrays that no solve loop writes hold the same primal values after both HVPs (the forward pass is the same)
        pos = 1
        for x in sig.args
            k = sig.kinds[x]
            if k == :array_float && !(x in loop_arrays)
                prim = max(prim, maximum(abs, a.args[pos] .- b.args[pos]; init = 0.0))
            end
            pos += k in (:scalar_float, :array_float) ? 2 : 1
        end
    end
    say(prim == 0.0, "$name converged: an array that no solve loop writes differs between the two HVPs ($prim)")
    say(!vacuous, "$name converged: the HVP is zero for every seeding (vacuous)")
    verbose && println(rpad(name, 26), " converged: worst err=", round(worst, sigdigits = 3), " sweeps=", Int(sweeps))
    # the tape: independent of the cap, smaller than the unrolled one
    if haskey(int0, :maxit)
        t1 = fph_tape(fin, fpn, int0, vals0); t2 = fph_tape(fin, fpn, merge(int0, Dict(:maxit => 2 * int0[:maxit])), vals0); tu = fph_tape(uin, unn, int0, vals0)
        say(t1 > 0 && t1 == t2, "$name converged: the tape has $t1 elements for the cap and $t2 for twice the cap")
        say(t1 < tu, "$name converged: the tape has $t1 elements, the unrolled one $tu")
    end
    # zero: the guard is closed
    zv = deepcopy(vals0); zv[:tol] = 1e30
    (a, b) = both(int0, zv, STADE.val_random_values_like(kernel, vals0), direction(), zeros_())
    ok = isempty(a.raised) && fpt_rel(a.out, b.out) <= 1e-12 && all(==(0.0), fph_info(a)[[1, 2, 4, 5]])
    say(ok, "$name zero: raised=$(a.raised) err=$(isempty(a.raised) ? fpt_rel(a.out, b.out) : NaN) info=$(fph_info(a))")
    # cap: three iterations do not reach the tolerance
    (a, _) = both(merge(int0, Dict(:maxit => 3)), vals0, STADE.val_random_values_like(kernel, vals0), direction(), zeros_())
    inf = fph_info(a)
    say(occursin("stopped on its cap", a.raised), "$name cap: the HVP raises the cut-off message (got: $(first(a.raised, 100)))")
    say(inf[4] >= 1 && inf[2] == 0 && inf[5] == 2, "$name cap: the cut-off count is set, no sweep ran, the status is 2 ($inf)")
    # nocap: maxit = 0 runs no iteration
    ni = merge(int0, Dict(:maxit => 0), haskey(int0, :maxq) ? Dict(:maxq => 0) : Dict{Symbol,Int}())
    (a, b) = both(ni, vals0, STADE.val_random_values_like(kernel, vals0), direction(), zeros_())
    say(isempty(a.raised) && all(==(0.0), fph_info(a)) && fpt_rel(a.out, b.out) <= 1e-12, "$name nocap: raised=$(a.raised) info=$(fph_info(a))")
    return nothing
end

# the dispatch of stade_hvp
function fph_dispatch(say)
    root = @__DIR__
    cos_ = STADE.io_read_corpus_entry(joinpath(root, "fp-corpus", "fp_scalar_cos.jl"))
    r1 = fpv_logged(() -> STADE.stade_hvp(cos_; keep_push_pop = false))
    say(last(r1.value.hvp.args[1].args) === :fp_info && occursin("takes the fixed-point reverse", r1.log), "stade_hvp(keep_push_pop = false) takes the fixed-point path and says so")
    say(!occursin("fp_info", string(r1.value.hvp)) || last(r1.value.hvp.args[1].args) === :fp_info, "the fixed-point HVP has fp_info as its last argument only")
    r2 = fpv_logged(() -> STADE.stade_hvp(cos_; keep_push_pop = true))
    say(string(r2.value.hvp) == string(STADE.agen_hvp_unrolled(cos_; keep_push_pop = true).hvp) && occursin("keep_push_pop = true", r2.log), "stade_hvp(keep_push_pop = true) is the unrolled HVP and gives the reason")
    r3 = fpv_logged(() -> STADE.stade_hvp(cos_; keep_push_pop = false, fuse_ii_loops = true))
    say(string(r3.value.hvp) == string(STADE.agen_hvp_unrolled(cos_; keep_push_pop = false, fuse_ii_loops = true).hvp) && occursin("fuse_ii_loops = true", r3.log), "stade_hvp(fuse_ii_loops = true) is the unrolled HVP and gives the reason")
    # a loop the detector does not accept (fp_counter_used) is no solve loop: nothing is said. One that it accepts and the generator refuses (fp_accumulate) is said once.
    for (k, said) in (("fp_counter_used", false), ("fp_accumulate", true))
        e = STADE.io_read_corpus_entry(joinpath(root, "fp-corpus", k * ".jl"))
        r = fpv_logged(() -> STADE.stade_hvp(e; keep_push_pop = false))
        say(string(r.value.hvp) == string(STADE.agen_hvp_unrolled(e; keep_push_pop = false).hvp), "$k: a refused loop keeps the unrolled HVP")
        say(occursin("keeps its unrolled", r.log) == said, "$k: the log " * (said ? "says why the unrolled HVP is kept" : "is silent, as for the adjoint"))
    end
    # the generator itself refuses a kernel without a solve loop
    e0 = STADE.io_read_corpus_entry(joinpath(root, "val-corpus", first(f for f in sort(readdir(joinpath(root, "val-corpus"))) if endswith(f, ".jl") && !occursin(r"_(b|d|hv)\.jl$", f))))
    msg = try STADE.agen_hvp_fixedpoint(e0); "" catch err; err isa ErrorException ? err.msg : "other" end
    say(startswith(msg, "agen_hvp_fixedpoint:"), "agen_hvp_fixedpoint of a kernel without a solve loop raises its own error (got: $(first(msg, 60)))")
    # a kernel without a solve loop is not affected: the dispatch finds nothing
    n = 0; hit = String[]
    for f in sort(readdir(joinpath(root, "val-corpus")))
        endswith(f, ".jl") && !occursin(r"_(b|d|hv)\.jl$", f) || continue
        e = try STADE.io_read_corpus_entry(joinpath(root, "val-corpus", f)) catch; continue end
        k = try STADE.parse_kernel(e) catch; continue end
        n += 1
        isempty(STADE.agen_fp_detect(e, k.sig.kinds; kernel = k)) || push!(hit, f)
    end
    say(n > 100 && isempty(hit), "val-corpus: $n kernels read, solve loops found in $(length(hit)) ($(join(first(hit, 5), ", ")))")
    return nothing
end

function validate_fixedpoint_hvp(; verbose = true, only = String[], sabotage = false)
    checks = 0; bad = 0
    say(ok, msg) = (checks += 1; ok || (bad += 1; verbose && println("FAIL ", msg)))
    (isempty(only) || "dispatch" in only) && fph_dispatch(say)
    for (name, primal, pins, floats, gs, prep, dprep) in fpt_cases()
        isempty(only) || name in only || continue
        try fph_case(say, name, primal, pins, floats, gs, prep, dprep; verbose = verbose)
        catch e; say(false, "$name: raised ($(first(split(sprint(showerror, e), "\n"))))") end
    end
    verbose && println("validate_fixedpoint_hvp: $(checks - bad)/$checks ok")
    if sabotage
        text = read(joinpath(dirname(@__DIR__), "src", "STADE.jl"), String)
        cases = [
            ("the stop rule reads the adjoint and not its tangent", "    hvp && union!(names, Symbol[tgen_shadow(n) for n in names if tgen_shadow(n) in adjargs || tgen_shadow(n) in adjsyms])\n", "", ["fp_scalar_cos", "fp_array_tanh"]),
            ("the sweeps are built for an adjoint", "agen_fp_sweeps(rev[i], d, trips, bstack, E, argset, syms; hvp = hvp)", "agen_fp_sweeps(rev[i], d, trips, bstack, E, argset, syms; hvp = false)", ["fp_scalar_cos", "fp_array_tanh"]),
            ("a store into a shadow stack is not a tape statement", " || endswith(string(s.args[1].args[1]), \"_stack_d\")", "", ["fp_scalar_cos", "fp_array_tanh"]),
            ("the HVP keeps the saves the converged state makes dead", "    # same site decision as stade_adjoint_: the sweeps read the arrays of a solve loop as the loop left them where that is sound\n    fp_dets === nothing || ((snap_sites, agen_sites) = agen_fp_converged_apply(kernel, active_map, fp_dets, snap_sites, agen_sites, ii_plan, fp_report))\n", "", ["phc_heat_steps", "fp_mixed_state"]),
            ("stade_hvp never takes the fixed-point path", "    fp = agen_fp_adjoint_or_nothing(expr, keep_push_pop, fuse_ii_loops, true, independents, dependents; hvp = true)\n    fp === nothing || return fp\n", "", ["dispatch"]),
            ("agen_hvp_fixedpoint accepts a kernel without a solve loop", "    isempty(found) && error(\"agen_hvp_fixedpoint:", "    false && error(\"agen_hvp_fixedpoint:", ["dispatch"]),
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
                    include($(repr(joinpath(@__DIR__, "validate_fixedpoint_hvp.jl"))))
                    println("RESULT ", validate_fixedpoint_hvp(; verbose = false, only = $(repr(kset))) ? 0 : 1)
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
    exit(validate_fixedpoint_hvp(; only = String[a for a in ARGS if !startswith(a, "--")], sabotage = "--sabotage" in ARGS) ? 0 : 1)
end
