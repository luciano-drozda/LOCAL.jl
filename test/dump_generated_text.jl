# The generated text of the adjoint, the HVP and the tangent for every kernel of the corpora, with the flags off (no `checkpoint`), written to one file: two copies of STADE.jl are
# compared by dumping with each and comparing the files byte for byte (the check that a new option leaves the default output unchanged).
#   julia dump_generated_text.jl path/to/STADE.jl out.txt [dir ...]
include(abspath(ARGS[1]))
using Logging
out = ARGS[2]
dirs = length(ARGS) > 2 ? ARGS[3:end] : [joinpath(@__DIR__, d) for d in sort(readdir(@__DIR__)) if endswith(d, "-corpus")]
txt(x) = x isa Expr ? string(Base.remove_linenums!(deepcopy(x))) : string(x)
open(out, "w") do io
    for d in dirs
        isdir(d) || continue
        for f in sort(readdir(d))
            (endswith(f, ".jl") && !(endswith(f, "_b.jl") || endswith(f, "_d.jl") || endswith(f, "_hv.jl"))) || continue
            primal = try STADE.io_read_corpus_entry(joinpath(d, f)) catch e; println(io, "## ", f, " unreadable"); continue end
            for (label, gen) in (("adj-kpp", () -> STADE.stade_adjoint(primal; keep_push_pop = true)),
                                 ("adj", () -> STADE.stade_adjoint(primal; keep_push_pop = false)),
                                 ("adj-fuse", () -> STADE.stade_adjoint(primal; keep_push_pop = false, fuse_ii_loops = true)),
                                 ("hvp-kpp", () -> STADE.stade_hvp(primal; keep_push_pop = true)),
                                 ("hvp", () -> STADE.stade_hvp(primal; keep_push_pop = false)),
                                 ("tan", () -> STADE.stade_tangent(primal)))
                r = try with_logger(NullLogger()) do; gen() end catch e; println(io, "## ", f, " ", label, " error ", first(split(sprint(showerror, e), "\n"))); continue end
                println(io, "## ", f, " ", label)
                for k in propertynames(r); v = getproperty(r, k); v isa Expr && println(io, k, ": ", txt(v)); end
            end
        end
    end
end
println("done ", out)
