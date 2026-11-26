# Writes one test case for the cross-framework parity check.
#   julia dump_case.jl NAME SIZE OUTDIR
# Files in OUTDIR: meta.json, and raw little-endian binaries
#   NAME.in.bin    input value of every argument array (Float64 or Int64)
#   NAME.out.bin   value after the STADE primal kernel (every float array the kernel writes)
#   NAME.seed.bin  cotangent used for the adjoint (those same arrays)
#   NAME.grad.bin  STADE adjoint result, gradient of <seed, out> with respect to each independent float array
haskey(ENV, "STADE_SRC") || error("set STADE_SRC to the path of STADE.jl/src/STADE.jl")
include(joinpath(@__DIR__, "check_kernel.jl"))

function dump_case(name, size, outdir)
    mkpath(outdir)
    path = bench_kernel_path(name)
    args = bench_inputs(name, size)
    deps, indeps = ck_roles(path)
    rng = MersenneTwister(7)
    seedv = Dict(n => randn(rng, length(v)) for (n, v) in args if ck_isfloatarr(v) && n in deps)
    mp = Module(:Prim); Base.include(mp, path)
    out = ck_run_primal(mp, name, args)
    mb, _ = ck_load(path, name, :b; kpp = true, fuse = false, tag = 0)
    adj = ck_run_adjoint(mb, name, args, seedv, deps)
    meta = Dict{String,Any}("kernel" => name, "size" => size, "args" => Any[], "deps" => sort(string.(collect(deps))), "indeps" => sort(string.(collect(indeps))))
    wr(f, v) = open(f, "w") do io; write(io, v isa AbstractArray ? (eltype(v) <: AbstractFloat ? Float64.(vec(v)) : Int64.(vec(v))) : v); end
    for (n, v) in args
        e = Dict{String,Any}("name" => string(n))
        if v isa AbstractArray
            e["kind"] = eltype(v) <: AbstractFloat ? "float" : "int"; e["length"] = length(v)
            wr(joinpath(outdir, "$n.in.bin"), v)
            n in deps && ck_isfloatarr(v) && (wr(joinpath(outdir, "$n.out.bin"), out[n]); wr(joinpath(outdir, "$n.seed.bin"), seedv[n]))
            n in indeps && ck_isfloatarr(v) && wr(joinpath(outdir, "$n.grad.bin"), adj[n])
        else
            e["kind"] = "scalar"; e["value"] = v
        end
        push!(meta["args"], e)
    end
    open(joinpath(outdir, "meta.json"), "w") do io; print(io, STADE_json(meta)); end
end

# minimal JSON writer (no dependencies)
STADE_json(x::AbstractString) = "\"" * x * "\""
STADE_json(x::Number) = string(x)
STADE_json(x::AbstractVector) = "[" * join(STADE_json.(x), ",") * "]"
STADE_json(x::AbstractDict) = "{" * join(["\"$k\":" * STADE_json(v) for (k, v) in x], ",") * "}"

if abspath(PROGRAM_FILE) == (@__FILE__)
    dump_case(ARGS[1], ARGS[2], ARGS[3])
end
