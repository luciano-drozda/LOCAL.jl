# The generated CUDA and JACC text (host code and device kernels) of the adjoint and the HVP for every kernel of the corpora, written to one file: two copies of
# STADE.jl are compared by dumping with each and comparing the files byte for byte (the check that a change of the GPU stage leaves every corpus kernel's device code unchanged).
#   julia dump_gpu_text.jl path/to/STADE.jl out.txt [dir ...]
include(abspath(ARGS[1]))
using Logging
out = ARGS[2]
dirs = length(ARGS) > 2 ? ARGS[3:end] : [joinpath(@__DIR__, d) for d in sort(readdir(@__DIR__)) if endswith(d, "-corpus")]
txt(x) = x isa Expr ? string(Base.remove_linenums!(deepcopy(x))) : string(x)
function dump_plan(io, plan)
    for k in propertynames(plan)
        v = getproperty(plan, k)
        if v isa Expr; println(io, k, ": ", txt(v))
        elseif v isa AbstractVector; for (i, e) in enumerate(v); println(io, k, "[", i, "]: ", txt(e)); end
        elseif v isa Union{String,Symbol,Number,Bool}; println(io, k, ": ", v)
        end
    end
end
open(out, "w") do io
    for d in dirs
        isdir(d) || continue
        for f in sort(readdir(d))
            (endswith(f, ".jl") && !(endswith(f, "_b.jl") || endswith(f, "_d.jl") || endswith(f, "_hv.jl"))) || continue
            primal = try STADE.io_read_corpus_entry(joinpath(d, f)) catch e; println(io, "## ", f, " unreadable"); continue end
            for (label, gen) in (("adj", () -> STADE.stade_adjoint(primal; keep_push_pop = false)),
                                 ("adj-fuse", () -> STADE.stade_adjoint(primal; keep_push_pop = false, fuse_ii_loops = true)),
                                 ("hvp", () -> STADE.stade_hvp(primal; keep_push_pop = false)))
                r = try with_logger(NullLogger()) do; gen() end catch e; println(io, "## ", f, " ", label, " error"); continue end
                code = label == "hvp" ? r.hvp : r.adjoint
                for (bk, mk) in (("cuda", () -> STADE.stade_gpu(code, STADE.cgen_backend_cuda())), ("jacc", () -> STADE.stade_jacc(code)))
                    println(io, "## ", f, " ", label, " ", bk)
                    p = try with_logger(NullLogger()) do; mk() end catch e; println(io, "error ", first(split(sprint(showerror, e), "\n"))); continue end
                    dump_plan(io, p)
                end
            end
        end
    end
end
println("done ", out)
