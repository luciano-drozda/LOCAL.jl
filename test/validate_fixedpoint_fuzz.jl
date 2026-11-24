# The entry-value rule of the fixed-point adjoint against generated kernels (fuzz_fp_entry.jl): random statements around random solve loops, in a time loop, a
# branch or neither, one solve or two, plus one directed kernel per kind of read of the state before a solve (the state alone, a product, a call, a quotient, a
# condition). A fixed list of kernels (the generator is seeded), two draws of the inputs each. Checks:
#   no kernel that the dispatch accepts has a fixed-point adjoint that differs from a central difference of the primal (1e-4), or from the unrolled adjoint where the
#   guard is closed on entry
#   the list holds both outcomes: kernels accepted, and kernels refused for the entry value (a generator that refuses or accepts everything checks nothing)
# --sabotage: each rule of the scan (the scan, a product, a call, a quotient) is turned off in a copy of STADE.jl and the fuzz must find a wrong adjoint.
#   julia validate_fixedpoint_fuzz.jl [--sabotage]
isdefined(@__MODULE__, :STADE) || include(joinpath(dirname(@__DIR__), "src", "STADE.jl"))
isdefined(@__MODULE__, :sab_read) || include(joinpath(@__DIR__, "sabotage_run.jl"))
isdefined(@__MODULE__, :fzz_main) || include(joinpath(@__DIR__, "fuzz_fp_entry.jl"))

function validate_fixedpoint_fuzz(; verbose::Bool = true, sabotage::Bool = false, count::Int = 10, stop_on_defect::Bool = false)
    bad = 0
    draws = (20261122, 11)
    acc = 0; needed = 0
    for (fam, seed) in (("scalar", 601), ("array", 602))
        r = fzz_main(count, seed, fam; verbose = false, draws = draws, stop_on_defect = stop_on_defect, directed = true)
        bad += length(r.bad); acc += get(r.tally, :accept, 0); needed += r.needed
        stop_on_defect && !isempty(r.bad) && return false
    end
    if !sabotage
        acc >= 8 || (bad += 1; verbose && println("NOT OK only $acc kernels accepted"))
        needed >= 3 || (bad += 1; verbose && println("NOT OK only $needed kernels refused for the entry value and wrong when forced"))
        verbose && println(bad == 0 ? "ok     " : "NOT OK ", "fuzz: $acc accepted, $needed refused for the entry value and needed, $bad problems")
    end
    return bad == 0
end

function validate_fixedpoint_fuzz_sabotage(; verbose::Bool = true)
    bad = 0
    text = read(joinpath(dirname(@__DIR__), "src", "STADE.jl"), String)
    cases = [
        ("the scan finds nothing", "    agen_fp_stale_scan!(agen_fp_stmts(expr.args[2]), Set{Symbol}(), ctx, hits, true)\n    return sort(collect(hits))", "    agen_fp_stale_scan!(agen_fp_stmts(expr.args[2]), Set{Symbol}(), ctx, hits, true)\n    return Symbol[]"),
        ("a product reads no value", "                has(a) && any(j -> j != i && isact(args[j]), eachindex(args)) && return true", "                false && return true"),
        ("a call reads no value", "            return any(has, args) && any(isact, args)", "            return false"),
        ("a quotient reads no value", "            has(args[2]) && (isact(args[1]) || isact(args[2])) && return true\n            has(args[1]) && isact(args[2]) && return true", "            false && return true"),
    ]
    mktempdir() do tmp
        for (label, old, new) in cases
            n = length(findall(old, text))
            if n != 1
                bad += 1; verbose && println("NOT OK sabotage \"$label\": anchor found $n times"); continue
            end
            get(ENV, "SAB_ANCHORS_ONLY", "") == "1" && continue
            path = joinpath(tmp, "STADE_sab.jl"); write(path, replace(text, old => new; count = 1))
            script = """
                include($(repr(path)))
                cd($(repr(@__DIR__)))
                include($(repr(joinpath(@__DIR__, "validate_fixedpoint_fuzz.jl"))))
                println("RESULT ", validate_fixedpoint_fuzz(; verbose = false, sabotage = true, stop_on_defect = true) ? 0 : 1)
            """
            out = sab_read(script)
            m = match(r"RESULT (\d)", out)
            caught = out == "ERR" || m === nothing || m.captures[1] == "1"
            caught || (bad += 1)
            verbose && println(caught ? "ok     " : "NOT OK ", "sabotage \"$label\": ", caught ? "a wrong adjoint is found" : "NOT CAUGHT")
        end
    end
    return bad == 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    ok = validate_fixedpoint_fuzz()
    "--sabotage" in ARGS && (ok = validate_fixedpoint_fuzz_sabotage() && ok)
    exit(ok ? 0 : 1)
end
