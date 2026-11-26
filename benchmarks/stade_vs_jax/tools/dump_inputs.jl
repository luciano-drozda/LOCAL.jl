# Writes the input arrays of one benchmark case, for the framework timing runs. No STADE needed.
#   julia dump_inputs.jl NAME SIZE OUTDIR
# Files in OUTDIR: meta.json (kernel, size, args) and NAME.in.bin per array argument (little-endian Float64 or Int64).
# The roles (which arrays are outputs and which are differentiated) come from the small case written by dump_case.jl.
include(joinpath(@__DIR__, "inputs.jl"))

function dump_inputs(name, size, outdir)
    mkpath(outdir)
    args = bench_inputs(name, size)
    ent = String[]
    for (n, v) in args
        if v isa AbstractArray
            kind = eltype(v) <: AbstractFloat ? "float" : "int"
            open(joinpath(outdir, "$n.in.bin"), "w") do io
                write(io, kind == "float" ? Float64.(vec(v)) : Int64.(vec(v)))
            end
            push!(ent, "{\"name\":\"$n\",\"kind\":\"$kind\",\"length\":$(length(v))}")
        else
            push!(ent, "{\"name\":\"$n\",\"kind\":\"scalar\",\"value\":$v}")
        end
    end
    open(joinpath(outdir, "meta.json"), "w") do io
        print(io, "{\"kernel\":\"$name\",\"size\":\"$size\",\"args\":[", join(ent, ","), "]}")
    end
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    dump_inputs(ARGS[1], ARGS[2], ARGS[3])
end
