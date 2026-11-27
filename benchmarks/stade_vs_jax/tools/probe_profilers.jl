# Probe job: which GPU profilers can the cluster V100 node use? Prints one JSON line. Run it as a V100 job (see cluster/go.sh).
function sh(cmd; n = 1500)
    try
        s = read(pipeline(ignorestatus(`bash -lc $cmd`); stderr = devnull), String)
        s = strip(s)
        length(s) > n ? first(s, n) * " ..." : s
    catch e
        "ERROR " * first(string(e), 200)
    end
end
esc(s) = replace(replace(replace(replace(s, "\\" => "\\\\"), "\"" => "\\\""), "\n" => "\\n"), "\t" => " ")
out = Pair{String,String}[]
add(k, v) = push!(out, k => v)
add("host", sh("hostname"))
add("gpu", sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"))
add("which", sh("for t in nsys ncu nsight-sys nv-nsight-cu-cli nvprof nvcc; do p=\$(command -v \$t) && echo \"\$t=\$p\"; done; echo done"))
add("cuda_env", sh("env | grep -i -E 'cuda|nsight|cupti' | sort | head -20"))
add("cuda_dirs", sh("ls -d /usr/local/cuda* /opt/nvidia/* /opt/nvidia/nsight* /usr/local/NVIDIA* 2>&1 | head -20"))
add("find_tools", sh("find /usr /opt /apps /soft /share -maxdepth 6 \\( -name nsys -o -name ncu -o -name 'nsight-systems*' -o -name 'nsight-compute*' -o -name nvprof \\) 2>/dev/null | head -20; echo done"))
add("modules", sh("(module avail 2>&1 || true) | grep -i -E 'cuda|nsight|nvhpc|nvidia|cupti|profil' | head -20; echo done"))
add("restrict_profiling", sh("grep -i -E 'RestrictProfiling|Profiling' /proc/driver/nvidia/params 2>&1; echo done"))
add("perf_paranoid", sh("cat /proc/sys/kernel/perf_event_paranoid 2>&1"))
add("python_profilers", sh("module load python/anaconda3.12.7 >/dev/null 2>&1; V=\$WORK/stade_fw_venv; ls \$V/bin >/dev/null 2>&1 && env -u LD_LIBRARY_PATH \$V/bin/python -m pip list 2>/dev/null | grep -i -E 'jax|tensorboard|xprof|profile|cupti|nvidia' | head -20; echo done"))
# CUPTI through CUDA.jl, without nsys
cj = try
    @eval begin
        using CUDA
        x = CUDA.rand(Float64, 1_000_000)
        y = CUDA.zeros(Float64, 1_000_000)
        function k!(y, x)
            i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            if i <= length(y)
                CUDA.@atomic y[1] += x[i]
            end
            return nothing
        end
        @cuda threads = 256 blocks = cld(1_000_000, 256) k!(y, x); CUDA.synchronize()   # compile first
        r = CUDA.@profile trace = true begin
            @cuda threads = 256 blocks = cld(1_000_000, 256) k!(y, x)
            CUDA.synchronize()
        end
        sprint(show, r)
    end
catch e
    "ERROR " * first(sprint(showerror, e), 400)
end
add("cuda_jl_profile", length(cj) > 1800 ? first(cj, 1800) * " ..." : cj)
add("cuda_jl_version", try string(pkgversion(CUDA)) catch; "?" end)
println("{" * join(["\"$k\":\"$(esc(v))\"" for (k, v) in out], ",") * "}")
