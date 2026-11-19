# ===================== validate_large_draw.jl =====================
# Large draws of the baseline generator (v0.5.26). CPU only, no device. Two kernels of the corpus could not get a large draw from `val_generate_baseline`,
# for two different reasons, and neither was a defect of the kernel:
#   asnap_slice   indexes `(e - 1) * i_nfeat + f`: every array gets one length N, so N must reach i_nedge * i_nfeat (68 * 67 = 4556 at a draw of 33 to 70);
#                 grow_max = 256 or 512 cannot, 8192 can.
#   remat_shapes  sums squares of sums: the loss grows like i_n * i_nc^2 with unit-normal inputs and passes `max_output_magnitude = 1e6` from i_nc = 10; each
#                 such candidate halves the upper end of the integer range, so a run asking for 33 to 70 came back with i_n = i_nc = 9 and said nothing.
# This test shows each cause, shows the knob that lifts it, and that a run that asked for large integers and got small ones now says so.
#   julia validate_large_draw.jl
isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
# validate_corpus_gpu.jl runs on include, so the helper is read out of its text
let src = read(joinpath(@__DIR__, "validate_corpus_gpu.jl"), String)
    a = findfirst("# ---- begin gval_narrowed_note", src); b = findfirst("# ---- end gval_narrowed_note ----", src)
    (a === nothing || b === nothing) && error("validate_large_draw: gval_narrowed_note block not found in validate_corpus_gpu.jl")
    include_string(Main, src[first(a):last(b)])
end

function validate_large_draw()
    ok = true
    check(c, msg) = (println(c ? "ok   " : "FAIL ", msg); ok &= c)
    dir = joinpath(@__DIR__, "val-corpus")
    gen(name; kw...) = (p = joinpath(dir, name * ".jl"); e = STADE.io_read_corpus_entry(p); STADE.val_generate_baseline(STADE.parse_kernel(e), e; kw...))
    fails(name; kw...) = try; gen(name; kw...); nothing; catch e; sprint(showerror, e); end

    # asnap_slice: the array size
    pin = Dict(:i_nedge => 68, :i_nfeat => 67)
    msg = fails("asnap_slice"; int_lo = 33, int_hi = 70, grow_max = 1024, attempts = 3, int_pin = pin)
    check(msg !== nothing && occursin("grow_max", msg), "asnap_slice, 68 edges of 67 features, grow_max = 1024: refused, and the message says to raise grow_max")
    b = gen("asnap_slice"; int_lo = 33, int_hi = 70, grow_max = 8192, attempts = 64, int_pin = pin)
    check(b.int_args[:i_nedge] == 68 && b.int_args[:i_nfeat] == 67 && length(b.values[:agg]) >= 68 * 67, "asnap_slice at grow_max = 8192: the draw is 68 x 67, arrays of $(length(b.values[:agg]))")

    # remat_shapes: the magnitude of the output
    pin = Dict(:i_n => 40, :i_nc => 33)
    msg = fails("remat_shapes"; int_lo = 33, int_hi = 70, grow_max = 1024, attempts = 3, int_pin = pin)
    check(msg !== nothing && occursin("max_output_magnitude", msg), "remat_shapes, i_n = 40, i_nc = 33, default magnitude 1e6: refused, and the message names max_output_magnitude")
    b = gen("remat_shapes"; int_lo = 33, int_hi = 70, grow_max = 1024, attempts = 64, int_pin = pin, max_output_magnitude = 1e12)
    check(b.int_args[:i_n] == 40 && b.int_args[:i_nc] == 33, "remat_shapes with max_output_magnitude = 1e12: the draw is i_n = 40, i_nc = 33")
    # unpinned: the requested range holds (it used to collapse to 9)
    b = gen("remat_shapes"; int_lo = 33, int_hi = 70, grow_max = 1024, attempts = 64, max_output_magnitude = 1e12)
    check(all(v >= 33 for v in values(b.int_args)), "remat_shapes unpinned, 33 to 70, magnitude 1e12: both integers stay in the range ($(sort(collect(b.int_args))))")

    # the note
    check(gval_narrowed_note(Dict(:i_n => 9, :i_nc => 40), Dict{Symbol,Int}(), 33) == " narrowed=i_n<33", "narrowed note: an integer below int_lo is named")
    check(gval_narrowed_note(Dict(:i_n => 40, :i_nc => 40), Dict{Symbol,Int}(), 33) == "", "narrowed note: a draw inside the range has no note")
    check(gval_narrowed_note(Dict(:i_n => 9), Dict(:i_n => 9), 33) == "", "narrowed note: a pinned integer is the caller's choice, no note")
    check(gval_narrowed_note(Dict(:i_n => 2), Dict{Symbol,Int}(), 3) == " narrowed=i_n<3", "narrowed note: the default int_lo = 3 flags a range that fell to 2")
    println(ok ? "ALL OK" : "FAILED")
    return ok
end

abspath(PROGRAM_FILE) == (@__FILE__) && (validate_large_draw() || exit(1))
