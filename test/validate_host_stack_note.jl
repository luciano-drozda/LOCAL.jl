# ==================== validate_host_stack_note.jl =======================
# Checks the note of stade_adjoint_file / stade_hvp_file (v0.5.26) for a kernel whose keep_push_pop = false code still has host push!/pop! calls
# (a ragged loop nest inside another ragged loop, HANDOFF 4k). The note is an @info and changes no generated code.
#   1. WITNESS: phc_mg_solve (adjoint and HVP) gives a note that names the loop bound and counts the calls. Its counts are read from the generated function.
#   2. NESTED: `nest_in_lv` (bound assigned in the level loop, below an outer loop) gives a note; `nest_in_kk` (same value assigned in the outer loop) does not,
#      and `mg_vcycle` (a resolved ragged kernel) does not. The pair differs only in where `n = nfine - 1` stands.
#   3. FLAG: keep_push_pop = true (stacks are push!/pop! by design) never gives the note.
#   4. SAME FILE: the written file is identical with and without the note path (the note is read-only).
#   5. SABOTAGE: with the push!/pop! calls renamed in the generated function, the helper returns nothing; with them present it does not. The helper reads the code.
#
#   julia validate_host_stack_note.jl

isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
using Test

const HSN_SRC = """
# nest_%NAME%
function nest_%NAME%(u, f, r, nfine, nlev, maxit)
    for kk = 1:maxit
%OUT%        for lv = 1:nlev
%IN%            for jr = 1:n
                left = 0.0
                if jr > 1
                    left = u[jr - 1, lv]
                end
                right = 0.0
                if jr < n
                    right = u[jr + 1, lv]
                end
                r[jr, lv] = f[jr, lv] - ((2.0 * u[jr, lv] - left) - right)
            end
            for jc = 1:n
                u[jc, lv] = u[jc, lv] + 0.5 * r[jc, lv]
            end
        end
    end
    return nothing
end
"""

function hsn_source(name, where_)
    s = replace(HSN_SRC, "%NAME%" => name)
    s = replace(s, "%OUT%" => where_ == :out ? "        n = nfine - 1\n" : "", "%IN%" => where_ == :in ? "            n = nfine - 1\n" : "")
    return s
end

function hsn_note(path, mode; keep_push_pop = false)
    out = tempname() * ".jl"
    logs, _ = Test.collect_test_logs() do
        (mode == :hvp ? STADE.stade_hvp_file : STADE.stade_adjoint_file)(path, out; keep_push_pop = keep_push_pop, fuse_ii_loops = true)
    end
    notes = [string(l.message) for l in logs if occursin("push!", string(l.message))]
    return (note = isempty(notes) ? nothing : notes[1], text = read(out, String))
end

function validate_host_stack_note()
    ok = true
    check(c, msg) = (c || (println("FAIL ", msg); ok = false); c && println("ok   ", msg))
    dir = mktempdir()
    write(joinpath(dir, "nest_in_lv.jl"), hsn_source("in_lv", :in))   # the file name matches the function name
    pkk = joinpath(dir, "nest_in_kk.jl"); write(pkk, hsn_source("in_kk", :out))
    pmg = joinpath(@__DIR__, "counted-corpus", "phc_mg_solve.jl")
    pvc = joinpath(@__DIR__, "val-corpus", "mg_vcycle.jl")
    # 1. WITNESS
    for mode in (:adjoint, :hvp)
        r = hsn_note(pmg, mode)
        check(r.note !== nothing && occursin("`n`", r.note) && occursin("phc_mg_solve", r.note), "phc_mg_solve $mode: a note naming the bound `n`")
        if r.note !== nothing
            np = parse(Int, match(r"has (\d+) push!", r.note).captures[1]); nq = parse(Int, match(r"and (\d+) pop!", r.note).captures[1])
            nt = length(collect(eachmatch(r"push!\(", r.text))); nu = length(collect(eachmatch(r"pop!\(", r.text)))
            check(np > 0 && nq > 0 && np <= nt && nq <= nu, "phc_mg_solve $mode: the counts ($np, $nq) come from the generated function (file has $nt, $nu with the stack setup)")
        end
    end
    # 2. NESTED
    nlv = hsn_note(joinpath(dir, "nest_in_lv.jl"), :adjoint).note
    check(nlv !== nothing, "assignment in the level loop: a note")
    check(nlv !== nothing && occursin("assigning it once before the loops", nlv) && occursin("`if X <= bound` mask", nlv), "the note names both remedies: assign the bound once before the loops, or the constant bound with a mask")
    check(hsn_note(pkk, :adjoint).note === nothing, "same assignment in the outer loop: no note")
    check(hsn_note(pvc, :adjoint).note === nothing, "mg_vcycle (resolved ragged nest): no note")
    # 3. FLAG
    check(hsn_note(pmg, :adjoint; keep_push_pop = true).note === nothing, "keep_push_pop = true: no note")
    # 5. SABOTAGE
    primal = STADE.io_read_corpus_entry(pmg)
    kernel = STADE.parse_kernel(primal)
    gen = STADE.stade_adjoint(primal; keep_push_pop = false, fuse_ii_loops = true)
    check(STADE.agen_host_stack_note(kernel, gen.adjoint) !== nothing, "helper reports the generated adjoint")
    function rename(x)
        x isa Expr || return x
        args = Any[rename(a) for a in x.args]
        x.head === :call && !isempty(args) && args[1] in (:push!, :pop!) && (args[1] = :stack_op_renamed)
        return Expr(x.head, args...)
    end
    check(STADE.agen_host_stack_note(kernel, rename(gen.adjoint)) === nothing, "SABOTAGE: with the push!/pop! calls renamed the helper reports nothing")
    # 4. SAME FILE (note path is read-only): generate twice, the file text is the same
    out2 = tempname() * ".jl"; STADE.stade_adjoint_file(pmg, out2; keep_push_pop = false, fuse_ii_loops = true)
    check(read(out2, String) == hsn_note(pmg, :adjoint).text, "the written file does not depend on the note")
    println(ok ? "ALL OK" : "FAILED")
    return ok
end

abspath(PROGRAM_FILE) == @__FILE__() && (validate_host_stack_note() || exit(1))
