# ===================== validate_jacc_async.jl ======================
# Guard of `jacc_async = true` (v0.5.26, HANDOFF item 4g). Plan only, CPU only. With asynchronous launches the generated host function returns before the last
# kernel ends, and every host read or write of device data in between is safe only because it runs on the stream of the launches. Which host forms exist, and which of
# them were tried on a device, is the list `async_audited_forms()` of census_jacc_async_host.jl. This test fails when the host code holds a device-touching form that is
# not on that list (a new copy, fill, memcopy, library call, another macro), and shows that the check can fail.
#   1. unit: audited forms pass; an unaudited call or macro is reported; an audited form removed from the list is reported
#   2. plans: the async plans of a few corpus kernels (adjoint and HVP, glue off and on, production flags) hold only audited forms, and launch asynchronously
#   julia validate_jacc_async.jl
isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))
include(joinpath(@__DIR__, "census_jacc_async_host.jl"))

function validate_jacc_async()
    ok = true
    check(c, msg) = (println(c ? "ok   " : "FAIL ", msg); ok &= c)
    forms_of(e) = (f = Dict{String,Int}(); a = Ref(0); s = Ref(0); census_walk!(e, f; async_launches = a, sync_launches = s); (f, a[], s[]))

    # 1. unit
    good = Meta.parse("begin\n x = JACC.array(h)\n JACC.@parallel_for range = n sync = false k!(x)\n if CUDA.@allowscalar(x[1] > 0.0)\n y = JACC.to_host(x)\n end\n CUDA.unsafe_free!(x)\nend")
    (f, na, ns) = forms_of(good)
    check(isempty(async_unaudited(f)) && na == 1, "unit: JACC.array, an async launch, an allowscalar read, to_host and unsafe_free! are audited")
    for (src, what) in (("JACC.memcopy!(a, b)", "a dotted call"), ("copyto!(a, b)", "copyto!"), ("fill!(a, 0.0)", "fill!"), ("CUDA.synchronize()", "a CUDA call"),
                        ("CUDA.@sync k!(x)", "a CUDA macro"), ("MPI.Allreduce!(a, +, comm)", "an MPI call"), ("JACC.@parallel_for range = n k!(x); Base.@nospecialize x", "another macro"))
        (f, _, _) = forms_of(Meta.parse("begin\n" * src * "\nend"))
        check(!isempty(async_unaudited(f)), "unit: $what is reported as unaudited")
    end
    (f, _, _) = forms_of(good)
    aud = async_audited_forms(); delete!(aud, "call:CUDA.unsafe_free!")
    check(async_unaudited(f; aud) == ["call:CUDA.unsafe_free!"], "unit sabotage: with unsafe_free! taken off the list, exactly that form is reported")

    # 2. plans
    paths = corpus_paths()
    for n in ("advection", "br_cg1d2", "pv_a11")
        haskey(paths, n) || (println("FAIL corpus entry $n missing"); ok = false; continue)
        (forms, na, ns) = census_kernel(n, STADE.io_read_corpus_entry(paths[n]))
        ua = async_unaudited(forms)
        check(isempty(ua), "plans $n: every host form is audited" * (isempty(ua) ? "" : " ($(join(ua, ", ")))"))
        check(na > 0, "plans $n: $na async launches, $ns waiting (reductions)")
        aud = async_audited_forms(); delete!(aud, "macro:@allowscalar")
        haskey(forms, "macro:@allowscalar") && check(async_unaudited(forms; aud) == ["macro:@allowscalar"], "plans $n sabotage: without @allowscalar on the list, that form is reported")
    end
    # the default is untouched: a plan without jacc_async has no `sync = false`
    gen = STADE.stade_adjoint(STADE.io_read_corpus_entry(paths["br_cg1d2"]); keep_push_pop = false, fuse_ii_loops = true)
    pd = STADE.stade_jacc(gen.adjoint)
    (fd, nad, _) = forms_of(pd.host)
    check(nad == 0, "default: stade_jacc without jacc_async has no asynchronous launch")
    println(ok ? "ALL OK" : "FAILED")
    return ok
end

abspath(PROGRAM_FILE) == (@__FILE__) && (validate_jacc_async() || exit(1))
