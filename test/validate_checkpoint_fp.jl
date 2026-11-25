# The checkpointed adjoint on the fixed-point kernels (validate_fixedpoint.jl: the corpus kernels and every generated shape), a region forced on each loop that can be one.
# For each kernel, each regime of validate_fixedpoint.jl (converged: the loop stops on its tolerance; zero: the guard is closed before the first iteration; cap: the loop
# stops on its cap and the adjoint raises; nocap: maxit = 0) and each segment length ck in (1, 2, 3, 7): every argument after the call, every returned scalar and
# `fp_info` equal, bit for bit, those of stade_adjoint(keep_push_pop = false), the fixed-point adjoint. In the cap regime both raise the same message.
#   julia validate_checkpoint_fp.jl [kernel names]
isdefined(@__MODULE__, :STADE) || include(joinpath(dirname(@__DIR__), "src", "STADE.jl"))
isdefined(@__MODULE__, :fpt_cases) || include(joinpath(@__DIR__, "validate_fixedpoint_tangent_hvp.jl"))
isdefined(@__MODULE__, :ckv_call) || include(joinpath(@__DIR__, "validate_checkpoint.jl"))
using Random, Logging

# one case in one regime: (status, note)
function ckf_run(primal, pins, floats, gs, prep, regime::Symbol; cks = (1, 2, 3, 7), seed::Int = 20261122)
    kernel = STADE.parse_kernel(primal); sig = kernel.sig
    ck = lg = nothing
    lg = ckv_logged(() -> try STADE.agen_ckpt_adjoint(primal; min_trip = 0) catch e; e end)
    ck = lg.value
    ck isa Exception && return (:error, first(split(sprint(showerror, ck), "\n")))
    ck === nothing && return (:no_region, "")
    ref = with_logger(NullLogger()) do; STADE.stade_adjoint(primal; keep_push_pop = false) end
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
    rn = Symbol[a for a in ref.initstacks.args[1].args[2:end]]
    rfn = STADE.val_compile(sg_rename(ref.adjoint, :ckf_ref)); rin = STADE.val_compile(sg_rename(ref.initstacks, :ckf_ref_init))
    r0 = try ckv_call(rfn, rin, sig, int_args, vals, seedv, rn) catch e; String(first(split(sprint(showerror, e), "\n"))) end
    ksig = STADE.parse_kernel(ck.primal).sig
    cn = Symbol[a for a in ck.initstacks.args[1].args[2:end]]
    cfn = STADE.val_compile(sg_rename(ck.adjoint, :ckf_ck)); cin = STADE.val_compile(sg_rename(ck.initstacks, :ckf_ck_init))
    bad = Int[]
    for c in cks
        r = try ckv_call(cfn, cin, ksig, merge(int_args, Dict(:ck => c)), vals, seedv, cn) catch e; String(first(split(sprint(showerror, e), "\n"))) end
        if r0 isa String || r isa String
            (r0 isa String && r isa String && r0 == r) || push!(bad, c)
            continue
        end
        same = ckv_same(r0.ret, r.ret) && length(r.args) == length(r0.args) + 1 && all(ckv_same(x, y) for (x, y) in zip(r0.args, r.args[1:end - 1]))
        f0 = ckv_fpinfo(ref.initstacks, r0.stacks); f1 = ckv_fpinfo(ck.initstacks, r.stacks)
        same &= (f0 === nothing) == (f1 === nothing) && (f0 === nothing || ckv_same(f0, f1))
        same || push!(bad, c)
    end
    return isempty(bad) ? (:checked, r0 isa String ? "both raise" : "$(length(ck.regions)) region(s)") : (:differs, "ck=$(join(bad, ","))")
end

function validate_checkpoint_fp(; verbose = true, only = String[])
    counts = Dict{Symbol,Int}()
    for (name, primal, pins, floats, gs, prep, dprep) in fpt_cases()
        isempty(only) || name in only || continue
        for regime in (:converged, :zero, :cap, :nocap)
            (st, note) = try ckf_run(primal, pins, floats, gs, prep, regime)
                         catch e; (:error, first(split(sprint(showerror, e), "\n"))) end
            counts[st] = get(counts, st, 0) + 1
            verbose && println(rpad(name, 28), rpad(regime, 11), rpad(st, 11), first(note, 150))
            flush(stdout)
        end
    end
    verbose && println("validate_checkpoint_fp: ", join(["$k=$v" for (k, v) in sort(collect(counts); by = first)], " "))
    return get(counts, :differs, 0) == 0 && get(counts, :error, 0) == 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_checkpoint_fp(; only = String[a for a in ARGS if !startswith(a, "--")]) ? 0 : 1)
end
