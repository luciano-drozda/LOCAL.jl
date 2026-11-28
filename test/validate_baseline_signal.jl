# ==================== validate_baseline_signal.jl =========================
# Checks that val_generate_baseline never returns a draw that the primal leaves unchanged (v0.6.49).
# clamped_sumsq adds u[i]^2 for u[i] > 0 to loss[1]. With i_n = 3 and every u[i] <= 0 it adds 0.0, the output is not zero (loss starts nonzero),
# and the old guard (output identically zero) kept the draw. The GPU corpus then reported four VACUOUS checks (job CORP4).
#
#   julia validate_baseline_signal.jl

isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
using Random

function validate_baseline_signal()
    path = joinpath(@__DIR__, "val-corpus", "clamped_sumsq.jl")
    primal = STADE.io_read_corpus_entry(path)
    kernel = STADE.parse_kernel(primal)
    fn = STADE.val_compile(primal)
    ok = true
    # the test is stale unless the plain draw does give a draw without signal
    nvac = 0
    ndraw = 400
    for s in 1:ndraw
        Random.seed!(s)
        ints = Dict{Symbol,Int}(:i_n => 3)
        vals = STADE.val_random_values(kernel, Dict(:loss => (1,), :u => (3,)), ints; scale = 1.0)
        x0 = STADE.val_flatten(kernel, vals)
        cp = deepcopy(vals)
        Base.invokelatest(fn, cp[:loss], cp[:u], 3)
        nvac += STADE.val_flatten(kernel, cp) == x0
    end
    println("plain draws with no signal: ", nvac, " of ", ndraw)
    nvac > 0 || (println("no plain draw lacks signal: the test is stale"); ok = false)
    # the generator must never return such a draw
    nbad = 0
    for s in 1:ndraw
        Random.seed!(s)
        r = STADE.val_generate_baseline(kernel, primal; int_lo = 3, int_hi = 3)
        cp = deepcopy(r.values)
        x0 = STADE.val_flatten(kernel, r.values)
        Base.invokelatest(fn, cp[:loss], cp[:u], r.int_args[:i_n])
        nbad += STADE.val_flatten(kernel, cp) == x0
    end
    println("generated draws with no signal: ", nbad, " of ", ndraw)
    ok &= nbad == 0
    println(ok ? "PASS validate_baseline_signal" : "FAIL validate_baseline_signal")
    return ok
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    exit(validate_baseline_signal() ? 0 : 1)
end
