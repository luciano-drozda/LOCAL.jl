# Driver of one corpus-validation job. It is the "_script" of a gpu_v100 payload (tools/build_corpus_job.py fills the __TOKENS__).
# It unpacks a bundle (src/STADE.jl and test/ with the corpus) under $WORK, runs validate_corpus_gpu on the attached GPU in one Julia process,
# and prints ONE JSON line: the exit code, the seconds, the summary line, every result line of the validator, and the tail of its log.
using Base64
const TGZ = "__TGZ__"
const HARD_S = __HARD_S__
const CALL = "__CALL__"
function esc(s)
    io = IOBuffer(); write(io, '"')
    for c in String(s)
        c == '"' ? write(io, "\\\"") : c == '\\' ? write(io, "\\\\") : c == '\n' ? write(io, "\\n") : (c < ' ' ? write(io, ' ') : write(io, c))
    end
    write(io, '"'); return String(take!(io))
end
function main()
    work = get(ENV, "WORK", "/tmp")
    root = joinpath(work, "stade_corpus_runs"); mkpath(root)
    dir = mktempdir(root)
    write(joinpath(dir, "b.tgz"), base64decode(TGZ))
    run(`tar xzf $(joinpath(dir, "b.tgz")) -C $dir`)
    tdir = joinpath(dir, "test")
    # The test file ends with a bare call validate_corpus_gpu() that runs the corpus with the defaults when the file is included.
    # The mapexpr drops that call, so the corpus runs once, with CALL only (CORP1 to CORP3 ran the default run first, a duplicate).
    code = "include(ex -> (ex isa Expr && ex.head === :call && ex.args[1] === :validate_corpus_gpu) ? :nothing : ex, \"validate_corpus_gpu.jl\"); validate_corpus_gpu(\"val-corpus\"; $CALL)"
    logf = joinpath(dir, "corpus.log")
    t0 = time()
    cmd = setenv(`timeout -k 30 $HARD_S $(Base.julia_cmd()) --project=$(Base.active_project()) -e $code`, merge(ENV, Dict("JULIA_PKG_OFFLINE" => "true")); dir = tdir)
    p = run(pipeline(ignorestatus(cmd); stdout = logf, stderr = joinpath(dir, "corpus.err")))
    secs = round(time() - t0; digits = 1)
    log = read(logf, String); err = isfile(joinpath(dir, "corpus.err")) ? read(joinpath(dir, "corpus.err"), String) : ""
    lines = split(log, "\n")
    res = [l for l in lines if occursin(r"\[(adjoint|hvp)/(cuda|jacc)\]", l) && occursin(r"\s(ok|FAIL|VACUOUS|gen_error)\b", l)]
    bang = [l for l in lines if startswith(strip(l), "!!")]
    summ = [l for l in lines if occursin("checks passed", l)]
    println("{\"host\":" * esc(gethostname()) * ",\"slurm_job\":" * esc(get(ENV, "SLURM_JOB_ID", "")) * ",\"seconds\":" * string(secs) *
            ",\"exit\":" * string(p.exitcode) * ",\"call\":" * esc(CALL) * ",\"summary\":" * esc(join(summ, " | ")) *
            ",\"results\":[" * join((esc(l) for l in res), ",") * "],\"errors\":[" * join((esc(first(l, 1500)) for l in bang), ",") *
            "],\"log_tail\":" * esc(log[max(1, end - 3000):end]) * ",\"err_tail\":" * esc(err[max(1, end - 3000):end]) * "}")
end
main()
