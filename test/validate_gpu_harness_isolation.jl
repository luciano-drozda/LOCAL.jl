# ===================== validate_gpu_harness_isolation.jl =====================
# Structural gate for validate_corpus_gpu.jl (no device needed; the script runs on include, so its text is read).
# One kernel whose baseline cannot be generated (arrays of rank 3 or more, which io_write_baseline_yaml refused before it learned them; a draw that never avoids a domain error, as in
# phc_colebrook) used to raise out of the whole run and drop every kernel after it: a group of 25 battle kernels came back with no row at all.
#   1. ISOLATION: the baseline call sits inside a `try`; its `catch` prints the kernel and the message, records a `:no_baseline` row and goes on to the next kernel.
#   2. USAGE ERRORS STAY LOUD: the check of a stored baseline against a pinned integer (`delete it`) is outside that `try`, so a wrong pin still stops the run.
#   3. NOT A PASS: the summary counts every status other than :ok as NOT OK, so a skipped kernel is never reported as a pass.
# Each check is sabotaged: a copy of the text with the guard removed must fail it.
#   julia validate_gpu_harness_isolation.jl
function gh_checks(txt::String)
    r = Bool[]
    m = match(r"try\n\s+isfile\(yp\) \|\| STADE\.stade_generate_baseline_file\(.*?\)\n\s+catch e\n(.*?)\n\s+continue\n\s+end\n"s, txt)
    push!(r, m !== nothing && occursin("[baseline]", m.captures[1]) && occursin("status = :no_baseline", m.captures[1]) && occursin("sprint(showerror, e)", m.captures[1]))
    i = findfirst("status = :no_baseline", txt); j = findfirst("delete it", txt)
    push!(r, i !== nothing && j !== nothing && first(j) > first(i))
    push!(r, occursin("bad = count(r -> r.status != :ok, results)", txt))
    return r
end

function validate_gpu_harness_isolation()
    ok = true
    check(c, msg) = (println(c ? "ok   " : "FAIL ", msg); ok &= c)
    txt = read(joinpath(@__DIR__, "validate_corpus_gpu.jl"), String)
    names = ["the baseline call is inside a try whose catch reports the kernel and records :no_baseline",
             "a pinned-integer mismatch (delete it) is outside the try",
             "the summary counts every status other than :ok as NOT OK"]
    base = gh_checks(txt)
    for (k, nm) in enumerate(names); check(base[k], nm); end
    sab = [replace(txt, "catch e\n            println(\"  !! \", name, \" [baseline] \"" => "catch e\n            println(\"  !! \", name, \" [base] \""; count = 1),
           replace(txt, "status = :no_baseline" => "status = :gen_error"; count = 1),
           replace(txt, "bad = count(r -> r.status != :ok, results)" => "bad = count(r -> r.status == :FAIL, results)"; count = 1)]
    for (k, (nm, which)) in enumerate([("sabotage: the catch no longer says [baseline]", 1), ("sabotage: the row is not recorded as :no_baseline", 1), ("sabotage: the summary ignores a skipped kernel", 3)])
        check(txt != sab[k] && !gh_checks(sab[k])[which], nm)
    end
    # the pin check read before the try (it needs the baseline, so this is the wrong order): check 2 must see it
    seg = txt[first(findfirst("for (a, v) in pins_of[name]", txt)):last(findfirst("delete it\")", txt))]
    moved = replace(replace(txt, seg => ""; count = 1), "# a kernel that cannot get a baseline" => seg * "\n        # a kernel that cannot get a baseline"; count = 1)
    check(moved != txt && !gh_checks(moved)[2], "sabotage: the pin check placed before the try is seen")
    println(ok ? "ALL OK" : "FAILED")
    return ok
end
validate_gpu_harness_isolation() || exit(1)
