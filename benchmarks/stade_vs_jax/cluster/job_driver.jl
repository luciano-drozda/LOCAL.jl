# Driver of one benchmark job. It is the "_script" of a gpu_v100 payload (build_job.py fills the three __TOKENS__).
# It unpacks the bundle under $WORK, runs cluster/job_run.sh there, and prints ONE JSON line with the results of every run.
using Base64
const TGZ = "__TGZ__"
const HARD_S = __HARD_S__
const BUDGET_S = __BUDGET_S__
const STADE_TMO = __STADE_TMO__
const JAX_TMO = __JAX_TMO__
function esc(s)
    io = IOBuffer(); write(io, '"')
    for c in String(s)
        c == '"' ? write(io, "\\\"") : c == '\\' ? write(io, "\\\\") : c == '\n' ? write(io, "\\n") : (c < ' ' ? write(io, ' ') : write(io, c))
    end
    write(io, '"'); return String(take!(io))
end
rawjson(path) = (isfile(path) && (l = strip(read(path, String)); startswith(l, "{")) ? l : "null")
function info(path)
    d = Dict{String,String}()
    isfile(path) || return d
    for l in split(read(path, String), "\n")
        m = match(r"^([a-z_]+)=(.*)$", l); m === nothing || (d[m.captures[1]] = m.captures[2])
    end
    return d
end
function main()
    work = get(ENV, "WORK", "/tmp")
    root = joinpath(work, "stade_bench_runs"); mkpath(root)
    dir = mktempdir(root)
    write(joinpath(dir, "b.tgz"), base64decode(TGZ))
    run(`tar xzf $(joinpath(dir, "b.tgz")) -C $dir`)
    env = Dict("BUNDLE" => dir, "JULIA_BIN" => Base.julia_cmd().exec[1], "JPROJ" => Base.active_project(),
               "JOB_BUDGET_S" => string(BUDGET_S), "STADE_RUN_TMO" => string(STADE_TMO), "JAX_RUN_TMO" => string(JAX_TMO))
    vj = joinpath(dir, "versions.json")
    for (k, v) in (("STADE_COMMIT", "stade_commit"), ("KIT_HASH", "kit_hash"))
        m = match(Regex("\"$v\":\"([^\"]*)\""), read(vj, String)); m === nothing || (env[k] = m.captures[1])
    end
    t0 = time()
    log = read(pipeline(ignorestatus(setenv(`timeout -k 30 $HARD_S bash $(joinpath(dir, "cluster", "job_run.sh"))`, merge(ENV, env))); stderr = stdout), String)
    runs = String[]
    od = joinpath(dir, "out")
    for r in (isdir(od) ? sort(readdir(od)) : String[])
        d = joinpath(od, r); inf = info(joinpath(d, "info.txt"))
        fields = join(("\"$k\":" * esc(get(inf, k, "")) for k in ("fw", "config", "kernel", "size", "phase", "status")), ",")
        ex = get(inf, "exit", ""); sc = get(inf, "seconds", "")
        push!(runs, "{" * fields * ",\"exit\":" * (isempty(ex) ? "null" : ex) * ",\"seconds\":" * (isempty(sc) ? "null" : sc) *
                    ",\"gpu_before\":" * rawjson(joinpath(d, "gpu_before.json")) * ",\"gpu\":" * rawjson(joinpath(d, "gpu.json")) *
                    ",\"result\":" * rawjson(joinpath(d, "out.json")) *
                    ",\"stderr_tail\":" * esc(isfile(joinpath(d, "err_tail.txt")) ? read(joinpath(d, "err_tail.txt"), String) : "") * "}")
    end
    println("{\"host\":" * esc(gethostname()) * ",\"slurm_job\":" * esc(get(ENV, "SLURM_JOB_ID", "")) * ",\"seconds\":" * string(round(time() - t0; digits = 1)) *
            ",\"dir\":" * esc(dir) * ",\"versions\":" * rawjson(vj) * ",\"log_tail\":" * esc(log[max(1, end - 2000):end]) *
            ",\"runs\":[" * join(runs, ",") * "]}")
end
main()
