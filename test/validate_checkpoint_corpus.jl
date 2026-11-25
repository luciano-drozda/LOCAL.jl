# The checkpointed adjoint on the loops of the corpora, forced: every top-level loop that can be a region is one (the literal trip-count rule is off), whatever the kernel.
# For each kernel of the directories and each segment length ck in (1, 2, 3, 7): every argument after the call and every returned scalar of the checkpointed adjoint
# equal, bit for bit, those of agen_adjoint_unrolled(keep_push_pop = false) on the same random draw. A kernel the adjoint without checkpoints cannot size is skipped.
# Every kernel gets one row: no loop that can be a region, no region (the reasons), checked, differs, error (the first line of the message).
#   julia validate_checkpoint_corpus.jl [dir ...] [-k kernel,kernel]
isdefined(@__MODULE__, :STADE) || include(joinpath(dirname(@__DIR__), "src", "STADE.jl"))
isdefined(@__MODULE__, :ckv_call) || include(joinpath(@__DIR__, "validate_checkpoint.jl"))
using Random, Logging

function ckc_kernel(path::String; cks = (1, 2, 3, 7), seed_name::String = "")
    name = splitext(basename(path))[1]
    primal = STADE.io_read_corpus_entry(path)
    kernel = STADE.parse_kernel(primal); sig = kernel.sig
    (cands, why) = STADE.agen_ckpt_candidates(primal, kernel; min_trip = 0)
    isempty(cands) && return (name = name, status = :no_candidate, note = isempty(why) ? "" : first(why))
    ref = try STADE.agen_adjoint_unrolled(primal; keep_push_pop = false)
          catch e; return (name = name, status = :plain_refuses, note = first(split(sprint(showerror, e), "\n"))) end
    lg = ckv_logged(() -> try STADE.agen_ckpt_adjoint(primal; min_trip = 0) catch e; e end)
    ck = lg.value
    ck isa Exception && return (name = name, status = :error, note = first(split(sprint(showerror, ck), "\n")))
    if ck === nothing
        why = [replace(l, "┌ Info: " => "", "stade_adjoint: checkpoint: " => "") for l in split(lg.log, "\n") if occursin("checkpoint:", l)]
        return (name = name, status = :no_region, note = isempty(why) ? "" : why[end])
    end
    Random.seed!(STADE.val_stable_seed(name))
    bl = STADE.val_generate_baseline(kernel, primal; self_check = false, int_lo = 3, int_hi = 5, grow_start = 8, grow_max = 64)
    STADE.val_coerce_int_arrays!(kernel, bl.values)
    int_args = bl.int_args; vals = bl.values
    seedv = STADE.val_random_values_like(kernel, vals)
    rn = Symbol[a for a in ref.initstacks.args[1].args[2:end]]
    rfn = STADE.val_compile(sg_rename(ref.adjoint, :ckc_ref)); rin = STADE.val_compile(sg_rename(ref.initstacks, :ckc_ref_init))
    r0 = ckv_call(rfn, rin, sig, int_args, vals, seedv, rn)
    ksig = STADE.parse_kernel(ck.primal).sig
    cn = Symbol[a for a in ck.initstacks.args[1].args[2:end]]
    cfn = STADE.val_compile(sg_rename(ck.adjoint, :ckc_ck)); cin = STADE.val_compile(sg_rename(ck.initstacks, :ckc_ck_init))
    bad = Int[]
    for c in cks
        r = try ckv_call(cfn, cin, ksig, merge(int_args, Dict(:ck => c)), vals, seedv, cn)
            catch e; return (name = name, status = :error, note = "ck=$c: " * first(split(sprint(showerror, e), "\n"))) end
        same = ckv_same(r0.ret, r.ret) && length(r.args) == length(r0.args) + 1 && all(ckv_same(x, y) for (x, y) in zip(r0.args, r.args[1:end - 1]))
        same || push!(bad, c)
    end
    return isempty(bad) ? (name = name, status = :checked, note = "$(length(ck.regions)) region(s)") : (name = name, status = :differs, note = "ck=$(join(bad, ","))")
end

function validate_checkpoint_corpus(dirs::Vector{String}; only::Vector{String} = String[], verbose::Bool = true)
    rows = NamedTuple[]
    for d in dirs
        for f in sort(readdir(d))
            (endswith(f, ".jl") && !(endswith(f, "_b.jl") || endswith(f, "_d.jl") || endswith(f, "_hv.jl"))) || continue
            isempty(only) || splitext(f)[1] in only || continue
            r = try ckc_kernel(joinpath(d, f))
                catch e
                    msg = first(split(sprint(showerror, e), "\n"))
                    # a kernel the parser or the baseline generator refuses is no subject for this test
                    (name = splitext(f)[1], status = (startswith(msg, "parse_kernel:") || startswith(msg, "val_generate_baseline:") || startswith(msg, "agen_split_uev!:")) ? :unusable : :error, note = msg)
                end
            r = merge(r, (dir = basename(d),))
            push!(rows, r)
            verbose && println(rpad(r.dir, 18), rpad(r.name, 34), rpad(string(r.status), 14), first(r.note, 160))
            flush(stdout)
        end
    end
    counts = Dict{Symbol,Int}()
    for r in rows; counts[r.status] = get(counts, r.status, 0) + 1; end
    verbose && println("validate_checkpoint_corpus: ", join(["$k=$v" for (k, v) in sort(collect(counts); by = first)], " "))
    return get(counts, :differs, 0) == 0 && get(counts, :error, 0) == 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = [a for a in ARGS]
    only = String[]
    i = findfirst(==("-k"), args)
    i === nothing || (only = String.(split(args[i + 1], ",")); deleteat!(args, i:i + 1))
    dirs = isempty(args) ? [joinpath(@__DIR__, d) for d in ("val-corpus", "counted-corpus", "scalar-acc-corpus", "int-state-corpus", "inactive-read-corpus", "plain-read-corpus", "boxed-reset-corpus", "while-corpus", "split-corpus", "battle-corpus")] : String[abspath(a) for a in args]
    exit(validate_checkpoint_corpus(dirs; only = only) ? 0 : 1)
end
