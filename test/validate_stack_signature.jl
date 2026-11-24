# The stacks an adjoint (or HVP) takes as arguments are the stacks its initstacks function creates, for every kernel of every corpus, with keep_push_pop = false and
# fuse_ii_loops off and on. A stack in the signature that initstacks never creates makes the generated adjoint impossible to call. The unused-stack cleanup used to
# run only when fusion was on, or an array was saved, or a remat crossed blocks; remat_unused_stack (a top-level `z = z * x` on an unchanged input) escaped all three.
#   julia validate_stack_signature.jl
isdefined(@__MODULE__, :STADE) || include(joinpath(dirname(@__DIR__), "src", "STADE.jl"))
using Logging

# names of the stack arguments of `fn` (a generated function Expr) that `initstacks` does not return
function vss_missing(fn, initstacks)
    ret = String[]
    for st in initstacks.args[2].args
        st isa Expr && st.head === :return || continue
        v = st.args[1]
        v isa Expr && v.head === :tuple && append!(ret, string.(v.args))
        v isa Symbol && push!(ret, string(v))
    end
    sig = [string(a) for a in fn.args[1].args[2:end] if a isa Symbol && endswith(string(a), "_stack")]
    return sort(setdiff(sig, ret))
end

function vss_run(; verbose::Bool = true)
    bad = String[]; n = 0
    for d in sort(readdir(@__DIR__; join = true))
        (isdir(d) && endswith(d, "-corpus")) || continue
        for f in sort(readdir(d; join = true))
            endswith(f, ".jl") && !any(s -> endswith(f, s), ("_b.jl", "_d.jl", "_hv.jl")) || continue
            p = STADE.io_read_corpus_entry(f)
            for fuse in (false, true)
                for (nm, gen) in (("adjoint", () -> STADE.agen_adjoint_unrolled(p; keep_push_pop = false, fuse_ii_loops = fuse)),
                                  ("hvp", () -> STADE.stade_hvp(p; keep_push_pop = false, fuse_ii_loops = fuse)))
                    r = try with_logger(NullLogger()) do; gen(); end catch; continue end   # a kernel refused in this mode has no code to check
                    fn = nm == "adjoint" ? r.adjoint : r.hvp
                    m = vss_missing(fn, r.initstacks); n += 1
                    isempty(m) || push!(bad, "$(basename(f)) $nm fuse=$fuse: $(join(m, ", "))")
                end
            end
        end
    end
    verbose && println("stack signature: ", n, " generated functions checked, ", length(bad), " with a stack initstacks does not create")
    foreach(b -> println("  ", b), bad)
    return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    bad = vss_run()
    exit(isempty(bad) ? 0 : 1)
end
