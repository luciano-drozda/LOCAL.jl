# ==================== validate_int_pin.jl =======================
# Checks `int_pin` of val_generate_baseline (v0.5.24). int_lo and int_hi only bound a draw: after a divergent candidate the upper end is halved, so a
# range such as 100 to 200 on br_cg1d2 (a conjugate gradient) ends at a few points (4 to 13 on the seeds below), and a GPU parity run could not be
# pointed at a size. `int_pin = Dict(:n => 9)` fixes the named argument in every draw.
#   1. PIN: the pinned values come back, on several seeds.
#   2. UNKNOWN: a name that is not an integer scalar argument raises an error at once.
#   3. DEFAULT: an empty pin and no pin give the same baseline from the same seed, so the pin path leaves the default draw alone.
#   4. HOLD: a pin that the generator cannot satisfy fails; it is never narrowed to another size.
#   5. WITNESS: the collapse that the pin exists for still happens without it (when it stops, the test is stale).
#
#   julia validate_int_pin.jl

isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
using Random

function validate_int_pin()
    path = joinpath(@__DIR__, "boxed-reset-corpus", "br_cg1d2.jl")
    primal = STADE.io_read_corpus_entry(path)
    kernel = STADE.parse_kernel(primal)
    ok = true
    check(c, msg) = (c || (println("FAIL ", msg); ok = false); c && println("ok   ", msg))
    for s in (1, 2, 3)
        Random.seed!(s)
        r = STADE.val_generate_baseline(kernel, primal; int_lo = 3, int_hi = 5, int_pin = Dict(:n => 9, :maxit => 3))
        check(r.int_args[:n] == 9 && r.int_args[:maxit] == 3, "seed $s: n = 9 and maxit = 3 are kept (got n = $(r.int_args[:n]), maxit = $(r.int_args[:maxit]))")
    end
    for bad in (:b, :nope)
        e = try STADE.val_generate_baseline(kernel, primal; int_pin = Dict(bad => 3)); nothing catch err err end
        check(e !== nothing && occursin("int_pin names `$bad`", sprint(showerror, e)), "pin on `$bad` raises an error that names it")
    end
    for s in (4, 5)
        Random.seed!(s); r0 = STADE.val_generate_baseline(kernel, primal; int_lo = 3, int_hi = 5)
        Random.seed!(s); r1 = STADE.val_generate_baseline(kernel, primal; int_lo = 3, int_hi = 5, int_pin = Dict{Symbol,Int}())
        check(r0.int_args == r1.int_args && r0.values == r1.values, "seed $s: an empty pin leaves the draw as it was")
    end
    for s in (6, 7)
        Random.seed!(s)
        r = try STADE.val_generate_baseline(kernel, primal; int_lo = 3, int_hi = 5, grow_max = 20000, attempts = 8, int_pin = Dict(:n => 100, :maxit => 100)) catch e e end
        held = !(r isa NamedTuple) || (r.int_args[:n] == 100 && r.int_args[:maxit] == 100)
        check(held, "seed $s: n = 100, maxit = 100 is kept or the draw fails, never another size")
    end
    nwit = 0
    for s in (2, 3)
        Random.seed!(s)
        r = STADE.val_generate_baseline(kernel, primal; int_lo = 100, int_hi = 200, grow_max = 20000)
        nwit += r.int_args[:n] < 100
    end
    check(nwit > 0, "witness: 100 to 200 without a pin collapses below 100 on $nwit of 2 seeds (0 means the test is stale)")
    println(ok ? "PASS validate_int_pin" : "FAIL validate_int_pin")
    return ok
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    exit(validate_int_pin() ? 0 : 1)
end
