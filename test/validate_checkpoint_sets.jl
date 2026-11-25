# The checkpoint set of every corpus kernel that has a region (the literal trip-count rule off): the arrays and scalars each segment saves, the scratch arrays that no segment saves, the inputs.
#   julia validate_checkpoint_sets.jl [dir ...]
isdefined(@__MODULE__, :STADE) || include(joinpath(dirname(@__DIR__), "src", "STADE.jl"))
using Logging

function validate_checkpoint_sets(dirs::Vector{String}; verbose::Bool = true)
    nreg = 0; nscratch = 0; narr = 0
    for d in dirs
        isdir(d) || continue
        for f in sort(filter(x -> endswith(x, ".jl"), readdir(d)))
            name = splitext(f)[1]
            r = try with_logger(NullLogger()) do; STADE.agen_ckpt_adjoint(STADE.io_read_corpus_entry(joinpath(d, f)); min_trip = 0) end
                catch e; nothing end
            r === nothing && continue
            for s in r.sets
                perseg = [a for a in s.arrays if !(a in s.scratch)]
                nreg += 1; nscratch += length(s.scratch); narr += length(s.arrays)
                verbose && println(rpad(basename(d), 20), rpad(name, 30), "saved ", perseg, " scratch ", s.scratch, " scalars ", s.scalars, " inputs ", s.in_arrays, s.in_scalars)
            end
        end
    end
    verbose && println("validate_checkpoint_sets: $nreg regions, $narr written arrays, $nscratch of them scratch (not saved by a segment)")
    return true
end

if abspath(PROGRAM_FILE) == @__FILE__
    dirs = isempty(ARGS) ? [joinpath(@__DIR__, d) for d in ("val-corpus", "counted-corpus", "scalar-acc-corpus", "int-state-corpus", "inactive-read-corpus", "plain-read-corpus", "boxed-reset-corpus", "while-corpus", "split-corpus", "battle-corpus", "fp-corpus")] : String[abspath(a) for a in ARGS]
    exit(validate_checkpoint_sets(dirs) ? 0 : 1)
end
