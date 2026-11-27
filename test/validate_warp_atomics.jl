include(joinpath(@__DIR__, "sabotage_run.jl"))
include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

"""
    validate_warp_atomics(dir="val-corpus"; only=nothing, sabotage=true)

Structural gate for the warp-aggregated atomics of the CUDA backend. No GPU needed; `validate_corpus_gpu.jl` checks the numbers on a device.

For every corpus kernel, adjoint and HVP (`keep_push_pop = false`, `fuse_ii_loops = true`), three groups of checks:

  1. EQUIVALENCE: replacing each `__stade_wa_add!(A, i, v)` call of the default CUDA output by `CUDA.@atomic A[i] += v` gives exactly the output of
     `warp_atomics = false`. The pass changes nothing but the atomic sites.
  2. HELPERS: the helper definitions are present exactly when a call is; `warp_atomics = false`, AMDGPU and Metal output have none.
  3. SABOTAGE: with the emission function swapped to put the index and the value in the wrong order, check 1 must fail on a kernel with an atomic.
  4. GATE (v0.6.33, three-way since v0.6.36): no warp call has an index that differs in adjacent lanes (class :D), a `__stade_wa_add!` call has an index equal in every lane (:U),
     a `__stade_wa_runs!` call has any other index, hand-written kernels send each index shape to the right side, the classes equal
     those of the JACC analysis under the renamed thread variable, and the helper definitions leave with the last call. Sabotage: a gate that calls everything uniform (:U)
     leaves :D calls, and one that calls everything :D removes every call.
"""
function validate_warp_atomics(dir::String = joinpath(@__DIR__, "val-corpus"); only::Union{Vector{String},Nothing} = nothing, sabotage::Bool = true)
    bad = 0
    check(ok, msg) = (println(rpad(ok ? "ok" : "FAIL", 6), msg); ok || (bad += 1))
    cuda = STADE.cgen_backend_cuda()
    wa = STADE.cgen_warp_add_name()
    wr = STADE.cgen_warp_runs_name()
    names = sort([splitext(f)[1] for f in readdir(dir) if endswith(f, ".jl") &&
                  !any(endswith(f, s) for s in ("_b.jl", "_d.jl", "_hv.jl", "_cuda.jl", "_jacc.jl"))])
    only !== nothing && (names = filter(n -> n in only, names))
    function unwa(e)
        e isa Expr || return e
        args = Any[unwa(a) for a in e.args]
        if e.head == :call && length(args) == 4 && (args[1] === wa || args[1] === wr)
            return Expr(:macrocall, cuda.atomic_macro, nothing, Expr(:(+=), Expr(:ref, args[2], args[3]), args[4]))
        end
        return Expr(e.head, args...)
    end
    ncallsof(e, f) = e isa Expr ? (e.head == :call && !isempty(e.args) && e.args[1] === f ? 1 : 0) + sum(ncallsof(a, f) for a in e.args; init = 0) : 0
    ncalls(e) = ncallsof(e, wa) + ncallsof(e, wr)
    src(e) = STADE.io_expr_to_source(e)
    total = 0; kernels_with = 0
    for name in names
        pe = STADE.io_read_corpus_entry(joinpath(dir, name * ".jl"))
        for (mode, gen) in (("adjoint", STADE.stade_adjoint), ("hvp", STADE.stade_hvp))
            out = try gen(pe; keep_push_pop = false, fuse_ii_loops = true) catch; continue end
            e = mode == "hvp" ? out.hvp : out.adjoint
            on = STADE.stade_gpu(e, cuda; serial_loops = false, glue_kernels = false, reduce_block = false, inbounds = false, dead_scalars = false, split_reduce = 0, contraction = 0)
            off = STADE.stade_gpu(e, cuda; serial_loops = false, glue_kernels = false, warp_atomics = false, reduce_block = false, inbounds = false, dead_scalars = false, split_reduce = 0, contraction = 0)
            n = sum(ncalls(k) for k in on.kernels; init = 0)
            total += n; n > 0 && (kernels_with += 1)
            same = length(on.kernels) == length(off.kernels) && all(src(unwa(a)) == src(b) for (a, b) in zip(on.kernels, off.kernels)) && src(on.host) == src(off.host)
            check(same, "$name [$mode]: $n warp call(s), output equals the plain atomics after substitution")
            check((n > 0) == !isempty(on.helpers), "$name [$mode]: helpers present iff a call is ($(length(on.helpers)) definition(s))")
            check(isempty(off.helpers) && !any(k -> STADE.cgen_expr_calls(k, wa) || STADE.cgen_expr_calls(k, wr), off.kernels), "$name [$mode]: warp_atomics = false has no helper and no call")
            gate_ok = true
            for k in on.kernels
                k.head === :function || continue
                cls = STADE.cgen_warp_classes(k)
                function walk(x)
                    x isa Expr || return
                    if x.head == :call && length(x.args) == 4 && (x.args[1] === wa || x.args[1] === wr)
                        c = STADE.cgen_warp_class(x.args[3], cls)
                        (c === :D || (x.args[1] === wa) != (c === :U)) && (gate_ok = false)
                    end
                    foreach(walk, x.args)
                end
                walk(k)
            end
            check(gate_ok, "$name [$mode]: every call sits on its side of the gate (add: uniform index, runs: other, none: lane-distinct)")
            for (label, be) in (("amdgpu", STADE.cgen_backend_amdgpu()), ("metal", STADE.cgen_backend_metal()))
                pl = try STADE.stade_gpu(e, be; inbounds = false, dead_scalars = false, split_reduce = 0, contraction = 0) catch; continue end
                check(isempty(pl.helpers) && !any(k -> STADE.cgen_expr_calls(k, wa), pl.kernels), "$name [$mode]: $label output has no warp call")
            end
        end
    end
    println("\n$total warp call(s) in $kernels_with kernel plan(s)")
    check(total > 0, "at least one corpus kernel emits a warp call")
    # the gate on hand-written kernels: which indices reach the helper
    tidline = "__tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x"
    function gate_kernel(body; tid = "__tid")
        return Meta.parse("function k_g(A, B, idx, n, v)\n$(replace(tidline, "__tid" => tid))\n$body\nreturn nothing\nend")
    end
    function gate_case(label, body, want_add, want_runs, want_plain)
        k = gate_kernel(body)
        t = STADE.cgen_warp_gate(k, cuda)
        na = ncallsof(t, wa); nr = ncallsof(t, wr)
        np = length(findall("CUDA.@atomic", src(t)))
        check(na == want_add && nr == want_runs && np == want_plain, "gate: $label ($na add, $nr runs, $np plain; want $want_add, $want_runs, $want_plain)")
        # with the run helper off an S or X site keeps the full-warp helper
        t0 = STADE.cgen_warp_gate(k, merge(cuda, (warp_runs = false,)))
        check(ncallsof(t0, wa) == want_add + want_runs && ncallsof(t0, wr) == 0, "gate: $label: warp_runs = false keeps the full-warp helper")
        # the classes agree with the JACC analysis on the same body with the launch index renamed
        kj = gate_kernel(replace(body, "__tid" => "__jacc_i"); tid = "__jacc_i")
        cc = STADE.cgen_warp_classes(k); cj = STADE.jgen_warp_classes(kj)
        for v in (:j, :m, :c, :r, :q, :cur)
            haskey(cc, v) || haskey(cj, v) || continue
            check(get(cc, v, :U) === get(cj, v, :U), "gate: $label: class of `$v` equals the JACC class")
        end
    end
    W = "__stade_wa_add!"
    gate_case("a fixed address", "$W(A, 3, v)", 1, 0, 0)
    gate_case("the thread index", "$W(A, __tid, v)", 0, 0, 1)
    gate_case("a scalar assigned from the thread index", "j = __tid + 1\n$W(A, j, v)", 0, 0, 1)
    gate_case("a chain of two scalars", "j = __tid\nm = 2 * j\n$W(A, m, v)", 0, 0, 1)
    gate_case("a loop variable with fixed bounds", "for k = 1:n\n$W(A, k, v)\nend", 1, 0, 0)
    gate_case("a loop variable bounded by the thread index", "for k = 1:__tid\n$W(A, k, v)\nend", 0, 1, 0)
    gate_case("an index read at the thread index", "q = idx[__tid]\n$W(A, q, v)", 0, 1, 0)
    gate_case("an index read at a fixed position", "q = idx[2]\n$W(A, q, v)", 1, 0, 0)
    gate_case("one of each", "$W(A, 1, v)\n$W(B, __tid, v)", 1, 0, 1)
    gate_case("div of the thread index by a uniform value (lanes agree in runs)", "r = div(__tid - 1, n) + 1\n$W(A, r, v)", 0, 1, 0)
    gate_case("mod of the thread index (distinct lanes)", "c = mod(__tid - 1, n) + 1\n$W(A, c, v)", 0, 0, 1)
    gate_case("a flattened 2-D position", "c = mod(__tid - 1, n) + 1\nr = div(__tid - 1, n) + 1\n$W(A, c + (r - 1) * n, v)", 0, 0, 1)
    gate_case("the same position shifted by a neighbour offset", "c = mod(__tid - 1, n) + 1\nr = div(__tid - 1, n) + 1\n$W(A, c + (r - 1) * n + 1, v)", 0, 0, 1)
    gate_case("a row index times a width plus an inner loop variable", "r = div(__tid - 1, n) + 1\nfor k = 1:n\n$W(A, (r - 1) * n + k, v)\nend", 0, 1, 0)
    gate_case("the thread index scaled by a literal", "$W(A, 3 * __tid, v)", 0, 0, 1)
    gate_case("the sum of two lane-distinct values", "$W(A, __tid + mod(__tid, 7), v)", 0, 1, 0)
    gate_case("a scalar assigned twice, once from the thread index", "j = 1\nj = __tid\n$W(A, j, v)", 0, 1, 0)
    gate_case("the jam kernel index (a div part times a span plus a mod part)", "jb = div(__tid - 1, 32) * n + mod(__tid - 1, 32)\n$W(A, jb + 1, v)", 0, 0, 1)
    # an index that mentions another launch intrinsic is never :D: that kernel keeps the helper
    let k = gate_kernel("jx = threadIdx().y\n$W(A, jx, v)")
        t = STADE.cgen_warp_gate(k, cuda)
        check(ncallsof(t, wr) == 1 && ncallsof(t, wa) == 0, "gate: an index from threadIdx().y goes to the run helper, never plain")
    end
    let k = gate_kernel("for k = threadIdx().x:blockDim().x:n\n$W(A, k, v)\nend")
        t = STADE.cgen_warp_gate(k, cuda)
        check(ncallsof(t, wr) == 1 && ncallsof(t, wa) == 0, "gate: a block-strided loop variable goes to the run helper, never plain")
    end
    # a kernel with no call, and a kernel list that loses its last call, lose the definitions
    let ks = Expr[gate_kernel("$W(A, __tid, v)")], hs = STADE.cgen_warp_helper_exprs()
        gk = Expr[STADE.cgen_warp_gate(k, cuda) for k in ks]
        check(isempty(STADE.cgen_warp_helpers_prune(hs, gk)), "gate: the definitions leave with the last call")
        check(STADE.cgen_warp_helpers_prune(hs, Expr[gate_kernel("$W(A, 3, v)")]) == hs, "gate: the definitions stay while a call is left")
        rk = Expr[STADE.cgen_warp_gate(gate_kernel("q = idx[__tid]\n$W(A, q, v)"), cuda)]
        check(STADE.cgen_warp_helpers_prune(hs, rk) == STADE.cgen_warp_runs_helper_exprs(), "gate: a kernel with only a run call carries only the run definitions")
        both = vcat(rk, Expr[gate_kernel("$W(A, 3, v)")])
        check(STADE.cgen_warp_helpers_prune(hs, both) == vcat(hs, STADE.cgen_warp_runs_helper_exprs()), "gate: both kinds of call carry both groups of definitions")
    end
    if sabotage
        for (label, sabfn, only) in (
                ("every index calls itself uniform", "function cgen_warp_class(e, cls::Dict{Symbol,Symbol})\n    return :U\nend", ["idm_tile"]),
                ("every index calls itself lane-distinct", "function cgen_warp_class(e, cls::Dict{Symbol,Symbol})\n    return :D\nend", ["idm_tile"]))
            script = """
                include($(repr(joinpath(@__DIR__, "validate_warp_atomics.jl"))))
                Core.eval(STADE, Meta.parse($(repr(sabfn))))
                println("RESULT ", validate_warp_atomics($(repr(dir)); only = $(repr(only)), sabotage = false))
            """
            out = sab_read(script)
            m = match(r"RESULT (\d+)", out)
            check(m !== nothing && parse(Int, m.captures[1]) > 0, "sabotage \"$label\": the gate checks fail")
        end
    end
    if sabotage
        sab = "function cgen_atomic_add(backend, target, other)\n" *
              "    if cgen_warp_atomics_on(backend) && target isa Expr && target.head == :ref && length(target.args) == 2 && target.args[1] isa Symbol\n" *
              "        return Expr(:call, cgen_warp_add_name(), target.args[1], other, target.args[2])\n    end\n" *
              "    return Expr(:macrocall, backend.atomic_macro, nothing, Expr(:(+=), target, other))\nend"
        script = """
            include($(repr(joinpath(@__DIR__, "validate_warp_atomics.jl"))))
            Core.eval(STADE, Meta.parse($(repr(sab))))
            println("RESULT ", validate_warp_atomics($(repr(dir)); only = ["idm_tile"], sabotage = false))
        """
        out = sab_read(script)
        m = match(r"RESULT (\d+)", out)
        check(m !== nothing && parse(Int, m.captures[1]) > 0, "sabotage \"index and value swapped\": idm_tile fails the equivalence check")
    end
    println("\n", bad == 0 ? "all warp atomics checks passed" : "*** $bad warp atomics check(s) failed ***")
    return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_warp_atomics(; only = isempty(ARGS) ? nothing : ARGS) == 0 ? 0 : 1)
end
