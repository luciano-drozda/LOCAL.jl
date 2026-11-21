# ===================== validate_generated_untracked.jl =====================
# The corpus runners (validate_corpus.jl, validate_corpus_gpu.jl) delete and rewrite every `_b.jl`, `_d.jl` and `_hv.jl` next to a kernel, so those files are
# outputs, never inputs. They are not tracked: a tracked copy only goes stale and shows as a change after every run (commit 7674701 carried 96 of them).
#   1. IGNORED: `.gitignore` lists the three generated suffixes for the corpus directories.
#   2. NO KERNEL HIDES: every file with a generated suffix in a corpus directory has its source kernel next to it. A kernel that is itself named `x_b.jl` would be
#      ignored and deleted by the runners, so it must be renamed.
# Both checks are sabotaged: a `.gitignore` without a pattern, and a lone `foo_b.jl`, must fail.
#   julia validate_generated_untracked.jl
gu_gen(f) = match(r"^(.*)_(b|d|hv)\.jl$", f)
function gu_orphans(root::String)
    out = String[]
    for d in sort(filter(x -> endswith(x, "-corpus") && isdir(joinpath(root, x)), readdir(root)))
        fs = readdir(joinpath(root, d))
        for f in fs
            m = gu_gen(f)
            m !== nothing && !(m.captures[1] * ".jl" in fs) && push!(out, d * "/" * f)
        end
    end
    return out
end
gu_ignored(txt::String) = all(any(l -> strip(l) == p, split(txt, '\n')) for p in ("test/*-corpus/*_b.jl", "test/*-corpus/*_d.jl", "test/*-corpus/*_hv.jl"))

function validate_generated_untracked()
    ok = true
    check(c, msg) = (println(c ? "ok   " : "FAIL ", msg); ok &= c)
    test_dir = @__DIR__
    gi = read(joinpath(test_dir, "..", ".gitignore"), String)
    check(gu_ignored(gi), ".gitignore lists the three generated suffixes for the corpus directories")
    orph = gu_orphans(test_dir)
    check(isempty(orph), "every generated-suffix file in a corpus directory has its source kernel" * (isempty(orph) ? "" : " (orphans: " * join(orph[1:min(3, end)], ", ") * ")"))
    check(!gu_ignored(replace(gi, "test/*-corpus/*_hv.jl" => "")), "sabotage: a .gitignore without the _hv pattern fails check 1")
    tmp = mktempdir(); mkpath(joinpath(tmp, "x-corpus"))
    write(joinpath(tmp, "x-corpus", "foo_b.jl"), "")
    check(gu_orphans(tmp) == ["x-corpus/foo_b.jl"], "sabotage: a lone foo_b.jl is reported as an orphan")
    write(joinpath(tmp, "x-corpus", "foo.jl"), "")
    check(isempty(gu_orphans(tmp)), "a foo_b.jl next to foo.jl is not an orphan")
    rm(tmp; recursive = true, force = true)
    println(ok ? "ALL OK" : "FAILED")
    return ok
end
validate_generated_untracked() || exit(1)
