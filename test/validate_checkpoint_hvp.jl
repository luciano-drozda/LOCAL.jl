# The checkpointed HVP (stade_hvp(...; checkpoint = true), agen_ckpt_adjoint(...; hvp = true)). For each shape of ckpt_shapes.jl with a region, each number of steps and each segment
# length ck, and for each kernel of validate_fixedpoint.jl in each regime (converged, zero, cap, nocap): every argument after the call, every returned scalar and `fp_info` equal, bit for bit,
# those of stade_hvp(keep_push_pop = false) (the fixed-point HVP where a solve loop is), for three seedings: the adjoint seed, the tangent seed, both. In the cap regime both raise the same message.
#   julia validate_checkpoint_hvp.jl [names] [--sabotage]
isdefined(@__MODULE__, :STADE) || include(joinpath(dirname(@__DIR__), "src", "STADE.jl"))
isdefined(@__MODULE__, :fpt_cases) || include(joinpath(@__DIR__, "validate_fixedpoint_tangent_hvp.jl"))
isdefined(@__MODULE__, :ckv_call) || include(joinpath(@__DIR__, "validate_checkpoint.jl"))
using Random, Logging

# one call of a generated HVP: (returned scalars, the arguments after the call without the stacks, the stacks), or the first line of the error
function ckh_call(fn, initfn, sig, int_args, vals, seed, D, dseed, names)
    extra = Any[haskey(int_args, n) ? int_args[n] : deepcopy(vals[n]) for n in names]
    stacks = STADE.val_init_stacks(initfn, extra)
    args = STADE.val_hvp_call_args(sig, int_args, vals, seed, D, dseed, stacks)
    ret = try Base.invokelatest(fn, args...) catch e; return String(first(split(sprint(showerror, e), "\n"))) end
    return (ret = ret, args = args[1:length(args) - length(stacks)], stacks = stacks)
end

# the checkpointed HVP of `primal` against the plain one on one draw, for every seeding and every segment length: (status, note)
function ckh_compare(primal::Expr, int_args, vals, cks, seed::Integer; tol_masked::Bool = false)
    kernel = STADE.parse_kernel(primal); sig = kernel.sig
    lg = ckv_logged(() -> try STADE.agen_ckpt_adjoint(primal; min_trip = 0, hvp = true) catch e; e end)
    ck = lg.value
    ck isa Exception && return (:error, first(split(sprint(showerror, ck), "\n")))
    ck === nothing && return (:no_region, "")
    ref = with_logger(NullLogger()) do; STADE.stade_hvp(primal; keep_push_pop = false) end
    Random.seed!(seed)
    zeros_() = STADE.val_zeros_like(kernel, vals)
    rnd() = STADE.val_random_values_like(kernel, vals)
    function direction()
        d = STADE.val_randn(length(STADE.val_flatten(kernel, vals)))
        if tol_masked
            mask = zeros_(); mask[:tol] = 1.0
            d[findfirst(!=(0.0), STADE.val_flatten(kernel, mask))] = 0.0
        end
        return STADE.val_unflatten(kernel, int_args, vals, d)
    end
    seedings = [rnd(), zeros_(), rnd()], [zeros_(), rnd(), rnd()]
    rn = Symbol[a for a in ref.initstacks.args[1].args[2:end]]
    rfn = STADE.val_compile(sg_rename(ref.hvp, :ckh_ref)); rin = STADE.val_compile(sg_rename(ref.initstacks, :ckh_ref_init))
    ksig = STADE.parse_kernel(ck.primal).sig
    cn = Symbol[a for a in ck.initstacks.args[1].args[2:end]]
    cfn = STADE.val_compile(sg_rename(ck.adjoint, :ckh_ck)); cin = STADE.val_compile(sg_rename(ck.initstacks, :ckh_ck_init))
    bad = String[]; nonzero = false
    for (k, (sv, dsv)) in enumerate(zip(seedings...))
        D = direction()
        r0 = ckh_call(rfn, rin, sig, int_args, vals, sv, D, dsv, rn)
        for c in cks
            r = ckh_call(cfn, cin, ksig, merge(int_args, Dict(:ck => c)), vals, sv, D, dsv, cn)
            if r0 isa String || r isa String
                (r0 isa String && r isa String && r0 == r) || push!(bad, "seeding $k ck=$c: " * (r0 isa String ? "plain raises ($r0)" : r isa String ? "checkpointed raises ($r)" : ""))
                continue
            end
            ki = findfirst(==(:ck), Symbol[a for a in ck.adjoint.args[1].args[2:end]])
            same = ki !== nothing && ckv_same(r0.ret, r.ret) && length(r.args) == length(r0.args) + 1 && all(ckv_same(x, y) for (x, y) in zip(r0.args, r.args[[i for i in eachindex(r.args) if i != ki]]))
            f0 = ckv_fpinfo(ref.initstacks, r0.stacks); f1 = ckv_fpinfo(ck.initstacks, r.stacks)
            same &= (f0 === nothing) == (f1 === nothing) && (f0 === nothing || ckv_same(f0, f1))
            same || push!(bad, "seeding $k ck=$c")
            nonzero |= any(x -> x isa AbstractArray ? any(!=(0.0), x) : x != 0.0, r0.ret isa Tuple ? collect(r0.ret) : [r0.ret])
        end
    end
    isempty(bad) || return (:differs, join(bad, "; "))
    return (:checked, "$(length(ck.regions)) region(s)" * (nonzero ? "" : ", all returned scalars zero"))
end

function validate_checkpoint_hvp(; verbose = true, only = String[])
    checks = 0; bad = 0
    say(ok, msg) = (checks += 1; ok || (bad += 1; verbose && println("NOT OK ", msg)))
    # 1. the shapes of ckpt_shapes.jl
    for (name, src, expect) in ckp_shapes()
        expect === :region || continue
        isempty(only) || name in only || continue
        primal = Meta.parse(src)
        kernel = STADE.parse_kernel(primal)
        opts = ckv_options(name)
        for nt in ckv_nsteps()
            pins = Dict{Symbol,Int}()
            for a in kernel.sig.args
                kernel.sig.kinds[a] == :scalar_int || continue
                pins[a] = a === :n ? 6 : a === :nt2 ? nt + 2 : nt
            end
            merge!(pins, opts.pins)
            Random.seed!(20261122)
            bl = STADE.val_generate_baseline(kernel, primal; self_check = false, int_pin = copy(pins), int_lo = 3, int_hi = 5, grow_start = 4, grow_max = 64)
            STADE.val_coerce_int_arrays!(kernel, bl.values)
            vals = bl.values
            for (k, v) in opts.floats; haskey(vals, k) && (vals[k] = v); end
            (st, note) = try ckh_compare(primal, merge(bl.int_args, pins), vals, ckv_ints(nt), 20261123)
                         catch e; (:error, first(split(sprint(showerror, e), "\n"))) end
            say(st === :checked || st === :no_region, "$name nt=$nt: $st $note")
            verbose && println(rpad(name, 18), " nt=", lpad(nt, 2), " ", rpad(string(st), 10), first(note, 120))
            flush(stdout)
        end
    end
    # 2. the kernels with a solve loop, in the four regimes
    for (name, primal, pins, floats, gs, prep, dprep) in fpt_cases()
        isempty(only) || name in only || continue
        for regime in (:converged, :zero, :cap, :nocap)
            kernel = STADE.parse_kernel(primal)
            Random.seed!(20261122)
            bl = STADE.val_generate_baseline(kernel, primal; self_check = false, int_pin = copy(pins), int_lo = 3, int_hi = 5, grow_start = gs, grow_max = 64)
            STADE.val_coerce_int_arrays!(kernel, bl.values)
            int_args = merge(bl.int_args, pins); vals = bl.values
            regime === :cap && (int_args = merge(int_args, Dict(:maxit => 3)))
            regime === :nocap && (int_args = merge(int_args, Dict(:maxit => 0), haskey(int_args, :maxq) ? Dict(:maxq => 0) : Dict{Symbol,Int}()))
            for (k, v) in floats; vals[k] = v; end
            prep === nothing || prep(vals)
            regime === :zero && (vals[:tol] = 1e30)
            (st, note) = try ckh_compare(primal, int_args, vals, (1, 2, 3, 7), 20261123; tol_masked = true)
                         catch e; (:error, first(split(sprint(showerror, e), "\n"))) end
            say(st === :checked || st === :no_region, "$name $regime: $st $note")
            verbose && println(rpad(name, 28), rpad(regime, 11), rpad(string(st), 10), first(note, 120))
            flush(stdout)
        end
    end
    verbose && println("validate_checkpoint_hvp: $(checks - bad)/$checks ok")
    return bad == 0
end

# sabotage: each guard of the HVP path is taken out of a copy of STADE.jl and a check must fail
function validate_checkpoint_hvp_sabotage(; verbose = true)
    text = read(joinpath(dirname(@__DIR__), "src", "STADE.jl"), String)
    bad = 0
    cases = (
        ("the tangents are not in the sets", "hvp && ((sets, kinds, base) = agen_ckpt_hvp_sets(sets, kinds, rawtext))", "nothing", ["ck_array", "ck_scalar"]),
        ("a tangent array is not saved", "push!(out, d); kinds2[d] = kinds[a]; base[d] = a", "kinds2[d] = kinds[a]; base[d] = a", ["ck_array"]),
        ("a tangent scalar is not saved", "get(kinds, a, :none) in (:array_float, :scalar_float) || continue", "get(kinds, a, :none) in (:array_float,) || continue", ["ck_scalar"]),
        ("the tangent stacks are sized from the tangent", ":(\$n = length(\$(get(base, a, a))))", ":(\$n = length(\$a))", ["ck_array"]),
        ("the CSE temporaries of an HVP are not inlined in a tape index", "agen_fp_collect_defs!(abody, defs); agen_ckpt_collect_hdefs!(abody, defs)", "agen_fp_collect_defs!(abody, defs)", ["ck_scalar"]),
        ("the tangent stacks are no parameters of initstacks", "get(base, a, a) in iparams || push!(iparams, get(base, a, a))", "a in iparams || push!(iparams, a)", ["ck_array"]),
    )
    mktempdir() do tmp
        for (label, old, new, names) in cases
            n = length(findall(old, text))
            if n < 1
                bad += 1; verbose && println("FAIL sabotage \"$label\": anchor not found"); continue
            end
            path = joinpath(tmp, "STADE_sab.jl"); write(path, replace(text, old => new))
            script = """
                include($(repr(path)))
                cd($(repr(@__DIR__)))
                include($(repr(joinpath(@__DIR__, "validate_checkpoint_hvp.jl"))))
                println("RESULT ", validate_checkpoint_hvp(; verbose = false, only = $(repr(names))) ? 0 : 1)
            """
            out = try read(pipeline(`$(Base.julia_cmd()) --startup-file=no -e $script`; stderr = devnull), String) catch; "ERR" end
            m = match(r"RESULT (\d)", out)
            caught = out == "ERR" || m === nothing || m.captures[1] == "1"
            caught || (bad += 1)
            verbose && println(caught ? "ok     " : "NOT OK ", "sabotage \"$label\": ", caught ? "a check fails" : "NOT CAUGHT")
        end
    end
    return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    ok = validate_checkpoint_hvp(; only = String[a for a in ARGS if !startswith(a, "--")])
    "--sabotage" in ARGS && (ok = validate_checkpoint_hvp_sabotage() == 0 && ok)
    exit(ok ? 0 : 1)
end
