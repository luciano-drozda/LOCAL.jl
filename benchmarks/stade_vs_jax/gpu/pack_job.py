"""usage: pack_job.py OUT.json TIMEOUT_S script1.jl script2.jl ...
Builds a gpu_v100 job payload. The payload script runs each runner script in its own Julia process
(a crash in one cannot lose the others) and prints one JSON line with every result."""
import sys, json, base64, os
out, tmo = sys.argv[1], int(sys.argv[2]); paths = sys.argv[3:]
b = {os.path.basename(p)[:-3]: base64.b64encode(open(p, "rb").read()).decode() for p in paths}
driver = r'''
using Base64
const SCRIPTS = Dict{String,String}(%s)
const TMO = %d
function esc(s)
    io = IOBuffer(); write(io, '"')
    for c in String(s)
        c == '"' ? write(io, "\\\"") : c == '\\' ? write(io, "\\\\") : c == '\n' ? write(io, "\\n") : c == '\t' ? write(io, "\\t") : (c < ' ' ? write(io, ' ') : write(io, c))
    end
    write(io, '"'); return String(take!(io))
end
function runone(name, b64)
    dir = mktempdir(); f = joinpath(dir, name * ".jl")
    write(f, base64decode(b64))
    so = joinpath(dir, "out.txt"); se = joinpath(dir, "err.txt")
    t0 = time()
    proc = run(pipeline(ignorestatus(`timeout $TMO $(Base.julia_cmd()) --project=$(Base.active_project()) $f`); stdout = so, stderr = se))
    el = time() - t0
    lines = filter(!isempty, split(read(so, String), "\n"))
    last = isempty(lines) ? "" : String(lines[end])
    err = read(se, String)
    return "{\"name\":" * esc(name) * ",\"exit\":" * string(proc.exitcode) * ",\"seconds\":" * string(round(el; digits = 1)) *
           ",\"result\":" * (startswith(last, "{") ? last : "null") * ",\"stderr_tail\":" * esc(err[max(1, end - 1200):end]) * "}"
end
function main()
    runs = String[]
    for n in sort(collect(keys(SCRIPTS)))
        push!(runs, runone(n, SCRIPTS[n]))
    end
    println("{\"host\":" * esc(gethostname()) * ",\"runs\":[" * join(runs, ",") * "]}")
end
main()
''' % (", ".join(f'"{n}" => "{v}"' for n, v in b.items()), tmo)
json.dump({"input": {"_script": driver, "_topology": "gpu_v100"}}, open(out, "w"))
print(out, len(driver)//1000, "KB script,", len(paths), "runner scripts")
