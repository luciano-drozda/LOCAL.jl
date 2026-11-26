"""usage: pack_py.py OUT.json TIMEOUT_S BUNDLE.tgz SCRIPT.sh
Builds a gpu_v100 payload. The Julia driver unpacks BUNDLE.tgz into a fresh directory under $WORK (exported as BUNDLE),
runs SCRIPT.sh with bash there, and prints one JSON line with the tail of its combined output."""
import sys, json, base64
out, tmo, tgz, sh = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
b64 = base64.b64encode(open(tgz, "rb").read()).decode()
script = base64.b64encode(open(sh, "rb").read()).decode()
driver = r'''
using Base64
const TGZ = "%s"
const SH = "%s"
function esc(s)
    io = IOBuffer(); write(io, '"')
    for c in String(s)
        c == '"' ? write(io, "\\\"") : c == '\\' ? write(io, "\\\\") : c == '\n' ? write(io, "\\n") : c == '\t' ? write(io, "  ") : (c < ' ' ? write(io, ' ') : write(io, c))
    end
    write(io, '"'); return String(take!(io))
end
function main()
    work = get(ENV, "WORK", "/tmp")
    dir = mktempdir(joinpath(work, "stade_fw_runs"; ) |> (d -> (mkpath(d); d)))
    write(joinpath(dir, "b.tgz"), base64decode(TGZ)); write(joinpath(dir, "run.sh"), base64decode(SH))
    run(`tar xzf $(joinpath(dir, "b.tgz")) -C $dir`)
    t0 = time()
    log = read(pipeline(ignorestatus(setenv(`timeout %d bash $(joinpath(dir, "run.sh"))`, merge(ENV, Dict("BUNDLE" => dir)))); stderr = stdout), String)
    println("{\"seconds\":" * string(round(time() - t0; digits = 1)) * ",\"dir\":" * esc(dir) * ",\"log\":" * esc(log[max(1, end - 60000):end]) * "}")
end
main()
''' % (b64, script, tmo)
json.dump({"input": {"_script": driver, "_topology": "gpu_v100"}}, open(out, "w"))
