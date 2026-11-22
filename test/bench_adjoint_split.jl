# Device timing of the record/replay split (v0.5.37). For one kernel on one size, on N seeds:
#   full    N calls of the full adjoint, each with its own tape allocation (what a user runs without the split)
#   split   the tape allocation and the forward half once, then N calls of the backward half, each after a device copy of the primal arrays the
#           forward half left (the backward half rewrites them) and with a fresh seed
# Every number is the minimum over `reps` repetitions of the whole sequence, in seconds, after a warm-up of each path and a device synchronisation.
# The tape sizes are those of the allocators: the split tape holds every store (a half drops none), the full adjoint's tape is the one the stage shrinks.
# Needs the library validate_adjoint_split_gpu.jl (the caller includes it). No correctness check here: the parity is validate_adjoint_split_gpu.jl.
#   sg_bench(path; env, sync, pins, floats, nseeds, reps) returns a NamedTuple-like Dict.

sg_bytes(stk) = sum((length(x) * sizeof(eltype(x)) for x in stk if x isa AbstractArray); init = 0)

# the tape bytes of the two allocators for one integer draw (no kernel run)
function sg_tape_bytes(P, kernel, int_args, vals, env)
    wrap = P[:wrap]
    sstk = sg_init_stacks(P[:dev_init], sg_stack_extra(P[:names_sp], int_args, vals, wrap))
    fstk = P[:dev_full_gen] === nothing ? () : sg_init_stacks(P[:dev_full_init], sg_stack_extra(P[:names_full], int_args, vals, wrap))
    return (split = sg_bytes(sstk), full = sg_bytes(fstk))
end

function sg_bench(path; env = sg_device_env(), sync = () -> nothing, pins = Dict{Symbol,Int}(), floats = Dict{Symbol,Float64}(), nseeds::Int = 8, reps::Int = 3,
                  grow_start::Int = 4, grow_max::Int = 1024, seedval::Int = 20261122)
    primal = STADE.io_read_corpus_entry(path)
    kernel = STADE.parse_kernel(primal)
    P = sg_setup(primal, kernel, env)
    P isa Tuple && error("sg_bench: setup returned $(P[1]) $(P[2])")
    P[:dev_full_gen] === nothing && error("sg_bench: no device full adjoint")
    Random.seed!(seedval)
    baseline = STADE.val_generate_baseline(kernel, primal; self_check = false, int_pin = pins, int_lo = 3, int_hi = 5, grow_start = grow_start, grow_max = grow_max)
    STADE.val_coerce_int_arrays!(kernel, baseline.values)
    int_args = merge(baseline.int_args, pins); vals = baseline.values
    for (k, v) in floats; vals[k] = v; end
    sig = P[:sig]; wrap = P[:wrap]; tohost = P[:tohost]
    seeds = [STADE.val_random_values_like(kernel, vals) for _ in 1:nseeds]
    tapes = sg_tape_bytes(P, kernel, int_args, vals, env)
    n_float_arrays = count(a -> sig.kinds[a] == :array_float, sig.args)

    # the primal arrays after one forward half (the state a replay starts from), on the host
    function record()
        dargs, dtr = sg_build_call(sig, int_args, vals, seeds[1], wrap)
        stk = sg_init_stacks(P[:dev_init], sg_stack_extra(P[:names_sp], int_args, vals, wrap))
        carry = Any[]
        if P[:carryfn] !== nothing
            for x in Base.invokelatest(P[:carryfn]); push!(carry, isempty(x) ? x : wrap(x)); end
        end
        Base.invokelatest(P[:dev_fwd], dargs..., stk..., carry...)
        return dargs, dtr, stk, carry
    end
    # warm-up: compile both paths
    dargs, dtr, stk, carry = record(); sync()
    Base.invokelatest(P[:dev_bwd], dargs..., stk..., carry...); sync()
    fa, ftr = sg_build_call(sig, int_args, vals, seeds[1], wrap)
    fs = sg_init_stacks(P[:dev_full_init], sg_stack_extra(P[:names_full], int_args, vals, wrap))
    Base.invokelatest(P[:dev_full_gen], fa..., fs...); sync()

    t_full = Inf; t_init = Inf; t_fwd = Inf; t_bwd = Inf; t_replays = Inf
    for rep in 1:reps
        # full adjoint, N calls (argument sets built outside the timer)
        sets = [sg_build_call(sig, int_args, vals, seeds[k], wrap)[1] for k in 1:nseeds]
        sync(); t0 = time_ns()
        for k in 1:nseeds
            fs = sg_init_stacks(P[:dev_full_init], sg_stack_extra(P[:names_full], int_args, vals, wrap))
            Base.invokelatest(P[:dev_full_gen], sets[k]..., fs...)
        end
        sync(); t_full = min(t_full, (time_ns() - t0) * 1e-9)

        # split: record once
        a1, tr1 = sg_build_call(sig, int_args, vals, seeds[1], wrap)
        sync(); t0 = time_ns()
        s1 = sg_init_stacks(P[:dev_init], sg_stack_extra(P[:names_sp], int_args, vals, wrap))
        sync(); t_init = min(t_init, (time_ns() - t0) * 1e-9)
        carry = Any[]
        if P[:carryfn] !== nothing
            for x in Base.invokelatest(P[:carryfn]); push!(carry, isempty(x) ? x : wrap(x)); end
        end
        sync(); t0 = time_ns()
        Base.invokelatest(P[:dev_fwd], a1..., s1..., carry...)
        sync(); t_fwd = min(t_fwd, (time_ns() - t0) * 1e-9)
        pristine = Dict{Symbol,Any}(a => tohost(tr.val) for (a, tr) in tr1)
        pristine_dev = Dict{Symbol,Any}(a => wrap(copy(v)) for (a, v) in pristine)
        # replays: N seeds; the primal arrays are restored by a device copy before each
        rsets = Any[]; rtrs = Any[]
        for k in 1:nseeds
            ra = Any[]; rt = Dict{Symbol,Any}()
            for a in sig.args
                kd = sig.kinds[a]
                if kd == :scalar_int
                    push!(ra, int_args[a])
                elseif kd == :scalar_float
                    push!(ra, vals[a]); push!(ra, seeds[k][a])
                elseif kd == :array_float
                    v = wrap(copy(pristine[a])); sh = wrap(deepcopy(seeds[k][a])); push!(ra, v); push!(ra, sh); rt[a] = v
                else
                    v = wrap(copy(pristine[a])); push!(ra, v); rt[a] = v
                end
            end
            push!(rsets, ra); push!(rtrs, rt)
        end
        sync(); t0 = time_ns()
        tb = 0.0
        for k in 1:nseeds
            for (a, v) in rtrs[k]; copyto!(v, pristine_dev[a]); end
            Base.invokelatest(P[:dev_bwd], rsets[k]..., s1..., carry...)
        end
        sync(); tr_all = (time_ns() - t0) * 1e-9
        t_replays = min(t_replays, tr_all)
        t_bwd = min(t_bwd, tr_all / nseeds)
    end
    t_split_total = t_init + t_fwd + t_replays
    # break-even: the N at which init + fwd + N*bwd equals N*full_per_call (full_per_call = t_full / nseeds)
    per_full = t_full / nseeds
    be = per_full > t_bwd ? (t_init + t_fwd) / (per_full - t_bwd) : Inf
    return Dict{Symbol,Any}(:N => nseeds, :t_full_total => t_full, :t_full_per_call => per_full, :t_init => t_init, :t_fwd => t_fwd, :t_bwd_per_replay => t_bwd,
                            :t_split_total => t_split_total, :speedup => t_full / t_split_total, :break_even_N => be,
                            :tape_split_bytes => tapes.split, :tape_full_bytes => tapes.full, :ints => join(["$k=$v" for (k, v) in sort(collect(int_args))], ","))
end

# the tape size alone for a range of one integer (a counted-guard loop sizes the tape for its cap `maxit`)
function sg_tape_scan(path, name::Symbol, values::Vector{Int}; env = sg_device_env(), pins = Dict{Symbol,Int}(), grow_start::Int = 4, grow_max::Int = 1024, seedval::Int = 20261122)
    primal = STADE.io_read_corpus_entry(path)
    kernel = STADE.parse_kernel(primal)
    P = sg_setup(primal, kernel, env)
    P isa Tuple && error("sg_tape_scan: setup returned $(P[1]) $(P[2])")
    rows = Tuple{Int,Int,Int}[]
    for v in values
        Random.seed!(seedval)
        pn = merge(pins, Dict{Symbol,Int}(name => v))
        baseline = STADE.val_generate_baseline(kernel, primal; self_check = false, int_pin = pn, int_lo = 3, int_hi = 5, grow_start = grow_start, grow_max = grow_max)
        STADE.val_coerce_int_arrays!(kernel, baseline.values)
        t = sg_tape_bytes(P, kernel, merge(baseline.int_args, pn), baseline.values, env)
        push!(rows, (v, t.split, t.full))
    end
    return rows
end
