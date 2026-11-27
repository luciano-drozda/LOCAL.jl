# Probe job 2: does nsys and ncu profile a CUDA.jl kernel on the cluster V100? Prints one JSON line. Run it as a V100 job (see cluster/go.sh).
function sh(cmd; n = 2500, tail = true)
    try
        s = strip(read(pipeline(ignorestatus(`bash -lc $cmd`); stderr = stdout), String))
        length(s) > n ? (tail ? "... " * last(s, n) : first(s, n) * " ...") : s
    catch e
        "ERROR " * first(string(e), 200)
    end
end
esc(s) = replace(replace(replace(replace(s, "\\" => "\\\\"), "\"" => "\\\""), "\n" => "\\n"), r"[\x00-\x1f\x7f]" => " ")   # no raw control character in the JSON
out = Pair{String,String}[]
add(k, v) = push!(out, k => v)
d = mktempdir()
write(joinpath(d, "tiny.jl"), """
using CUDA
function k!(y, x)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(x)
        CUDA.@atomic y[1] += x[i]
    end
    return nothing
end
x = CUDA.rand(Float64, 1_000_000); y = CUDA.zeros(Float64, 1)
for _ in 1:3
    @cuda threads = 256 blocks = cld(1_000_000, 256) k!(y, x); CUDA.synchronize()
end
println("TINY OK ", Array(y)[1] > 0)
""")
jl = joinpath(Sys.BINDIR, "julia")
add("julia", sh("$jl --version; echo \$JULIA_DEPOT_PATH"))
add("tiny_plain", sh("cd $d && timeout 300 $jl --startup-file=no tiny.jl 2>&1 | tail -5"))
add("nsys_version", sh("nsys --version; ncu --version | tail -3"))
add("nsys_run", sh("cd $d && timeout 400 nsys profile --trace=cuda --sample=none --cpuctxsw=none -o prof --force-overwrite true $jl --startup-file=no tiny.jl 2>&1 | tail -8"))
add("nsys_stats", sh("cd $d && timeout 120 nsys stats --report cuda_gpu_kern_sum --format csv prof.nsys-rep 2>&1 | head -12"))
add("ncu_run", sh("cd $d && timeout 400 ncu --target-processes all --kernel-name regex:k --launch-count 1 --section SpeedOfLight --metrics smsp__inst_executed_op_global_red.sum,smsp__inst_executed_op_global_atom.sum $jl --startup-file=no tiny.jl 2>&1 | tail -30"; n = 1800))
println("{" * join(["\"$k\":\"$(esc(v))\"" for (k, v) in out], ",") * "}")
