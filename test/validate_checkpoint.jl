# The checkpointed adjoint (stade_adjoint with checkpoint = true, agen_ckpt_adjoint) against the adjoint without checkpoints, bit for bit.
# For each kernel of ckpt_shapes.jl, each number of steps nt in (0, 1, 2, 10) and each segment length ck in (1, 2, nt - 1, nt, nt + 1, 3) (those that are at least 1):
#   exact      every argument after the call (the primal arrays, every shadow, the integer arrays) and every returned scalar equal, bit for bit
#   vacuous    the gradient is not zero (a comparison of zeros proves nothing) and the region wrote something
#   memory     for ck < nt the tape (the stacks without the checkpoint stacks) is smaller than the tape without checkpoints, and does not depend on nt
# and the refusals: a loop that is not a region (independent iterations, no tape, a bound that changes, a small literal trip count) leaves the adjoint as it was,
# with its reason in the @info line. The dispatch: checkpoint = false (the default) is the adjoint of before, byte for byte.
# usage: julia validate_checkpoint.jl [kernel ...] [--sabotage]   (CKV_SAB=text runs the sabotage cases whose label holds the text)
isdefined(@__MODULE__, :STADE) || include(joinpath(dirname(@__DIR__), "src", "STADE.jl"))
isdefined(@__MODULE__, :ckp_shapes) || include(joinpath(@__DIR__, "ckpt_shapes.jl"))
isdefined(@__MODULE__, :sg_rename) || include(joinpath(@__DIR__, "validate_adjoint_split_gpu.jl"))
isdefined(@__MODULE__, :sab_read) || include(joinpath(@__DIR__, "sabotage_run.jl"))
using Random, Logging

# the output of `f` and the text the logger got, as in fpv_logged
function ckv_logged(f)
    buf = IOBuffer()
    v = with_logger(SimpleLogger(buf, Logging.Info)) do; f() end
    return (value = v, log = String(take!(buf)))
end

ckv_same(a, b) = a isa AbstractArray ? (b isa AbstractArray && size(a) == size(b) && all(isequal.(a, b))) : a isa Tuple ? (b isa Tuple && length(a) == length(b) && all(ckv_same.(a, b))) : isequal(a, b)

# one call of a generated adjoint: (returned scalars, the arguments after the call without the stacks, the stacks)
function ckv_call(adj_fn, init_fn, sig, int_args, vals, seed, names)
    extra = Any[haskey(int_args, n) ? int_args[n] : deepcopy(vals[n]) for n in names]
    stacks = STADE.val_init_stacks(init_fn, extra)
    args = STADE.val_adjoint_call_args(sig, int_args, vals, seed, stacks)
    ret = Base.invokelatest(adj_fn, args...)
    return (ret = ret, args = args[1:length(args) - length(stacks)], stacks = stacks)
end

# the integer pins and the float values a shape needs beyond the common ones (a solve loop needs a cap that is never reached and a tight tolerance)
ckv_options(name) = name in ("ck_solve", "ck_solve_steps", "ck_solve_array", "ck_solve_refused") ? (pins = Dict(:maxit => 600), floats = Dict(:tol => 1e-26, :a => 0.6, :lam => 0.7)) :
                    name == "ck_cov_run" ? (pins = Dict(:m => 3), floats = Dict{Symbol,Float64}()) :
                    name == "ck_cov_maynot" ? (pins = Dict(:m => 0), floats = Dict{Symbol,Float64}()) : (pins = Dict{Symbol,Int}(), floats = Dict{Symbol,Float64}())

# the `fp_info` of a call (the stack of that name), or nothing
function ckv_fpinfo(init, stacks)
    ret = STADE.agen_fp_stmts(init.args[2])[end].args[1]
    names = ret isa Expr && ret.head === :tuple ? ret.args : Any[ret]
    k = findfirst(==(:fp_info), names)
    return k === nothing ? nothing : copy(stacks[k])
end

ckv_ints(nt) = sort(unique(filter(c -> c >= 1, [1, 2, nt - 1, nt, nt + 1, 3])))
ckv_nsteps() = (0, 1, 2, 10)

# the kernel `src` for nt steps: the worst case of every ck, the sizes of the tapes. `errors` collects what differs.
function ckv_kernel(say, name::String, primal::Expr; nt::Int, seed::Int = 20261122, verbose::Bool = true)
    kernel = STADE.parse_kernel(primal); sig = kernel.sig
    pins = Dict{Symbol,Int}()
    for a in sig.args
        sig.kinds[a] == :scalar_int || continue
        pins[a] = a === :n ? 6 : a === :nt2 ? nt + 2 : nt
    end
    opts = ckv_options(name)
    merge!(pins, opts.pins)
    Random.seed!(seed)
    bl = STADE.val_generate_baseline(kernel, primal; self_check = false, int_pin = copy(pins), int_lo = 2, int_hi = 4, grow_start = 8, grow_max = 64)
    STADE.val_coerce_int_arrays!(kernel, bl.values)
    int_args = merge(bl.int_args, pins); vals = bl.values
    for (k, v) in vals
        v isa AbstractVector{Float64} && (vals[k] = 0.3 .* v)
    end
    vals[:a] = 0.9
    for (k, v) in opts.floats; haskey(vals, k) && (vals[k] = v); end
    seedv = STADE.val_random_values_like(kernel, vals)
    ref = with_logger(NullLogger()) do; STADE.stade_adjoint(primal; keep_push_pop = false) end
    rn = Symbol[a for a in ref.initstacks.args[1].args[2:end]]
    rfn = STADE.val_compile(sg_rename(ref.adjoint, :ckv_ref)); rin = STADE.val_compile(sg_rename(ref.initstacks, :ckv_ref_init))
    r0 = ckv_call(rfn, rin, sig, int_args, vals, seedv, rn)
    ck = STADE.agen_ckpt_adjoint(primal)
    ck === nothing && (say(false, "$name: no region was found"); return nothing)
    ksig = STADE.parse_kernel(ck.primal).sig
    cn = Symbol[a for a in ck.initstacks.args[1].args[2:end]]
    cfn = STADE.val_compile(sg_rename(ck.adjoint, :ckv_ck)); cin = STADE.val_compile(sg_rename(ck.initstacks, :ckv_ck_init))
    tape0 = sum(length, r0.stacks; init = 0)
    nz = count(!=(0.0), vcat([vec(x) for x in r0.args if x isa AbstractArray{Float64}]...))
    say(nz > 0, "$name nt=$nt: the reference arguments are all zero (vacuous)")
    worst = 0
    tapes = Dict{Int,Int}()
    for c in ckv_ints(nt)
        ia = merge(int_args, Dict(:ck => c))
        r = try ckv_call(cfn, cin, ksig, ia, vals, seedv, cn)
            catch e
                say(false, "$name nt=$nt ck=$c: raised ($(first(split(sprint(showerror, e), "\n"))))"); continue
            end
        # the stacks that hold the checkpoints and the end state end in `_ck_stack`, `_ckf_stack` (`_ck2_stack` for a second region)
        tape = 0
        ret = STADE.agen_fp_stmts(ck.initstacks.args[2])[end].args[1]
        retn = ret isa Expr && ret.head === :tuple ? ret.args : Any[ret]
        for (nm, st) in zip(retn, r.stacks); occursin(r"_ck[f0-9]*_stack$", string(nm)) || (tape += length(st)); end
        same = ckv_same(r0.ret, r.ret) && length(r.args) == length(r0.args) + 1 && all(ckv_same(x, y) for (x, y) in zip(r0.args, r.args[1:end - 1]))
        # a solve loop reports the same iterations, sweeps, residual ratio and status
        f0 = ckv_fpinfo(ref.initstacks, r0.stacks); f1 = ckv_fpinfo(ck.initstacks, r.stacks)
        same &= (f0 === nothing) == (f1 === nothing) && (f0 === nothing || ckv_same(f0, f1))
        say(same, "$name nt=$nt ck=$c: the checkpointed adjoint differs from the adjoint without checkpoints")
        tapes[c] = tape
        worst += same ? 0 : 1
        # a segment of at most a quarter of the steps must hold less than all of them (each step also holds the guard flag and the end of the segment)
        if 4c <= nt && nt >= 8
            say(tape < tape0, "$name nt=$nt ck=$c: the tape has $tape elements, without checkpoints $tape0")
        end
    end
    verbose && println(rpad(name, 14), " nt=", lpad(nt, 2), "  tape without checkpoints ", lpad(tape0, 6), "  with: ", join(["ck=$c:$(tapes[c])" for c in sort(collect(keys(tapes)))], " "), worst == 0 ? "  exact" : "  DIFFERS")
    return (tape0 = tape0, tapes = tapes)
end

# the loops that are not regions, and the default
function ckv_dispatch(say)
    for (name, src, expect) in ckp_shapes()
        expect === :region && continue
        primal = Meta.parse(src)
        if name == "ck_no_bound"
            (c, why) = STADE.agen_ckpt_candidates(primal, STADE.parse_kernel(primal))
            say(isempty(c) && any(w -> occursin(expect, w), why), "$name: the candidate test refuses the loop with `$expect` (got $why)")
            continue
        end
        r = ckv_logged(() -> STADE.stade_adjoint(primal; keep_push_pop = false, checkpoint = true))
        plain = STADE.stade_adjoint(primal; keep_push_pop = false)
        say(string(r.value.adjoint) == string(plain.adjoint) && string(r.value.initstacks) == string(plain.initstacks), "$name: a loop that is not a region keeps the adjoint without checkpoints")
        say(occursin(expect, r.log), "$name: the log gives the reason `$expect` (got: $(first(r.log, 200)))")
    end
    for (name, src, expect) in ckp_shapes()
        expect === :region || continue
        primal = Meta.parse(src)
        a = STADE.stade_adjoint(primal; keep_push_pop = false)
        b = STADE.stade_adjoint(primal; keep_push_pop = false, checkpoint = false)
        say(string(a.adjoint) == string(b.adjoint) && string(a.initstacks) == string(b.initstacks), "$name: checkpoint = false (the default) is the adjoint without checkpoints")
        r1 = ckv_logged(() -> STADE.stade_adjoint(primal; keep_push_pop = true, checkpoint = true))
        say(string(r1.value.adjoint) == string(STADE.stade_adjoint(primal; keep_push_pop = true, checkpoint = false).adjoint) && occursin("keep_push_pop = true", r1.log), "$name: checkpoint with keep_push_pop = true keeps the adjoint and says why")
        r2 = ckv_logged(() -> STADE.stade_adjoint(primal; keep_push_pop = false, fuse_ii_loops = true, checkpoint = true))
        say(occursin("fuse_ii_loops = true", r2.log), "$name: checkpoint with fuse_ii_loops = true says why")
    end
    return nothing
end

# the arrays that no segment saves are the ones listed in ckp_scratch
function ckv_sets(say)
    for (name, src, expect) in ckp_shapes()
        haskey(ckp_scratch(), name) || continue
        r = with_logger(NullLogger()) do; STADE.agen_ckpt_adjoint(Meta.parse(src); min_trip = 0) end
        got = r === nothing ? nothing : sort(vcat([s.scratch for s in r.sets]...); by = string)
        say(got == sort(ckp_scratch()[name]; by = string), "$name: the arrays no segment saves are $(ckp_scratch()[name]) (got $got)")
    end
    # a subscript with a symbol that the body assigns is no form, unless the symbol is the variable of an enclosing loop of the analysis (no kernel shows it: the adjoint without checkpoints refuses an integer that the step assigns twice)
    say(STADE.agen_ckpt_lin_ok(:(j + 1), Any[], Set([:j])) === nothing, "a subscript with a symbol that the body assigns is no linear form")
    say(STADE.agen_ckpt_lin_ok(:(j + 1), Any[(var = :j, lo = nothing, hi = nothing)], Set([:j])) !== nothing, "the variable of an enclosing loop is a symbol of the form")
    say(STADE.agen_ckpt_lin_ok(:(n + 1), Any[], Set([:j])) !== nothing, "an invariant symbol is a symbol of the form")
    return nothing
end

# the sizing function refuses a segment length below 1
function ckv_ckguard(say)
    src = first(s for (n, s, e) in ckp_shapes() if n == "ck_array")
    ck = STADE.agen_ckpt_adjoint(Meta.parse(src))
    init = STADE.val_compile(sg_rename(ck.initstacks, :ckv_guard_init))
    for c in (0, -3)
        msg = try Base.invokelatest(init, c, 3, 1, 4); "" catch e; sprint(showerror, e) end
        say(occursin("at least 1", msg), "initstacks with ck = $c raises the message about the segment length (got `$msg`)")
    end
    return nothing
end

function validate_checkpoint(; verbose = true, only = String[], sabotage = false)
    checks = 0; bad = 0
    say(ok, msg) = (checks += 1; ok || (bad += 1; verbose && println("FAIL ", msg)))
    (isempty(only) || "dispatch" in only) && (ckv_dispatch(say); ckv_ckguard(say); ckv_sets(say))
    for (name, src, expect) in ckp_shapes()
        expect === :region || continue
        isempty(only) || name in only || continue
        primal = Meta.parse(src)
        seen = Dict{Int,Int}()
        for nt in ckv_nsteps()
            r = try ckv_kernel(say, name, primal; nt = nt, verbose = verbose)
                catch e; say(false, "$name nt=$nt: raised ($(first(split(sprint(showerror, e), "\n"))))"); nothing end
            r === nothing && continue
            # the tape of a segment does not depend on the number of steps
            # a kernel with a loop outside the region (a tape that grows with `nt`, or sizes that all follow `nt`) has no tape of a segment alone
            for (c, t) in r.tapes
                name in ("ck_oldscalar", "ck_snapshot") && break
                haskey(seen, c) ? say(seen[c] == t, "$name ck=$c: the tape has $t elements for nt=$nt and $(seen[c]) for another number of steps") : (seen[c] = t)
            end
        end
    end
    verbose && println("validate_checkpoint: $(checks - bad)/$checks ok")
    sabotage && (bad += ckv_sabotage(verbose))
    return bad == 0
end

# each guard of the generator, taken out of a copy of STADE.jl: some check of the kernels in `kset` must fail
function ckv_sabotage(verbose::Bool)
    bad = 0
    text = read(joinpath(dirname(@__DIR__), "src", "STADE.jl"), String)
    cases = [
        ("an array of the written set is not saved", "    return (arrays = A, scalars = S, in_arrays = IA, in_scalars = IS, scratch = scratch)\n", "    return (arrays = A[1:max(0, length(A) - 1)], scalars = S, in_arrays = IA, in_scalars = IS, scratch = scratch)\n", ["ck_array", "ck_wave"]),
        ("a scalar of the written set is not saved", "    return (arrays = A, scalars = S, in_arrays = IA, in_scalars = IS, scratch = scratch)\n", "    return (arrays = A, scalars = Symbol[], in_arrays = IA, in_scalars = IS, scratch = scratch)\n", ["ck_scalar"]),
        ("the last segment runs past the last step", "guard = Expr(:if, :(\$(c.var) <= \$hi), Expr(:block,", "guard = Expr(:if, true, Expr(:block,", ["ck_array"]),
        ("the restore reads the wrong segment", "slot = :((\$s - 1) * \$n + \$i)", "slot = :((\$s) * \$n + \$i)", ["ck_array"]),
        ("the forward tape keeps the segment counter", "sf = Any[agen_ckpt_slot_tape(x, defs, trips, r.s) for x in fwd]", "sf = fwd", ["ck_array"]),
        ("the reverse tape keeps the segment counter", "sr = Any[agen_ckpt_slot_tape(x, defs, trips, r.s) for x in rev]", "sr = rev", ["ck_array"]),
        ("the tape outside the regions keeps the segment count", "stmts = Any[agen_ckpt_slot_tape(x, defs, trips, :__ck_none) for x in agen_fp_stmts(body)]", "stmts = agen_fp_stmts(body)", ["ck_scalar"]),
        ("the end state of an accumulator is not put back", "agen_ckpt_final(kp_.arrays, kp_.scalars, r.number, true)...])", "])", ["ck_accum"]),
        ("a restored array counts as never restored", "        l isa Expr && l.head === :ref && l.args[1] isa Symbol && push!(out, l.args[1])\n", "", ["ck_array"]),
        ("two regions share the names of their stacks", "k == 1 ? \"_ck_stack\" : \"_ck\$(k)_stack\"", "\"_ck_stack\"", ["ck_two"]),
        ("an input that a later statement writes is not saved", "for a in arrs2 if a in used && !(a in arrs)", "for a in Symbol[] if a in used && !(a in arrs)", ["ck_input"]),
        ("the inputs are not put back before the reverse loop", "agen_ckpt_inputs(set, r.number, true)..., newr,", "newr,", ["ck_input"]),
        ("a scalar input that a later statement writes is not saved", "for v in scal2 if v in used && !(v in scal)", "for v in Symbol[] if v in used && !(v in scal)", ["ck_input_s"]),
        ("a float scalar that every step assigns first is not saved", "(get(kinds, v, :none) === :scalar_float || (get(kinds, v, :none) === :scalar_int && (v in before || v in uev)))", "(get(kinds, v, :none) === :scalar_float && (v in before || v in uev) || (get(kinds, v, :none) === :scalar_int && (v in before || v in uev)))", ["ck_oldscalar"]),
        ("the index of an array snapshot stays in the forward loop", " || startswith(string(s.args[1]), \"__sbase_\")", "", ["ck_snapshot"]),
        ("a loop with independent iterations is a region", "elseif !agen_ckpt_carries(st)\n", "elseif false\n", ["dispatch"]),
        ("a loop without tape is a region", "(length(fl) == 1 && agen_ckpt_has_tape(fl[1], r.var)) || push!(notape, r.number)", "nothing", ["dispatch"]),
        ("the guard flag counts as a tape", "agen_ckpt_is_guard_flag(st) || agen_ckpt_has_tape(st, v) && return true", "agen_ckpt_has_tape(st, v) && return true", ["dispatch"]),
        ("a loop with a short literal trip count is a region", "elseif h.lo isa Integer && h.hi isa Integer && h.hi - h.lo + 1 <= min_trip", "elseif false", ["dispatch"]),
        ("a loop whose bounds change is a region", "if !isempty(intersect(bound, union(arrs, scal, lvars))) || h.var in bound", "if false", ["dispatch"]),
        ("a solve loop that the fixed-point reverse refuses is checkpointed", "            isempty(why) || error(\"agen_ckpt: the kernel has a solve loop", "            true || error(\"agen_ckpt: the kernel has a solve loop", ["dispatch"]),
        ("a solve loop is a region", "        if any(d -> d.loop === st, solves)\n", "        if false\n", ["ck_solve"]),
        ("the converged field is not read through an if", "    for (blk, i) in c.lchain\n        append!(pre,", "    for (blk, i) in c.lchain[1:1]\n        append!(pre,", ["ck_solve_array"]),
        ("a gap between two covered ranges is a cover", "agen_ckpt_lin_eq(q.lo, agen_ckpt_lin_add(p.hi, one, 1)) || continue", "true || continue", ["dispatch", "ck_cov_gap"]),
        ("a read past the end of a cover is covered", "agen_ckpt_le(p.lo, e, ranges) && agen_ckpt_le(e, p.hi, ranges)", "agen_ckpt_le(p.lo, e, ranges)", ["dispatch", "ck_cov_shift"]),
        ("a read before the start of a cover is covered", "agen_ckpt_le(p.lo, e, ranges) && agen_ckpt_le(e, p.hi, ranges)", "agen_ckpt_le(e, p.hi, ranges)", ["dispatch", "ck_cov_below"]),
        ("a write under an if covers what follows the if", "agen_ckpt_exposed!(body, Dict{Symbol,Vector{Any}}(k => copy(x) for (k, x) in cov), ranges, bad, exposed)", "agen_ckpt_exposed!(body, cov, ranges, bad, exposed)", ["dispatch", "ck_cov_if"]),
        ("a write in a loop that may not run covers", "return runs ? piece : nothing", "return piece", ["dispatch", "ck_cov_maynot"]),
        ("a window of writes in a loop that may not run covers", "(get(width, nothing, 0) == 0 || runs) || return nothing", "nothing", ["dispatch", "ck_cov_window"]),
        ("a symbol that the body assigns is in a subscript form", "(k in bad && !any(r -> r.var === k, ranges)) && return nothing", "nothing", ["dispatch"]),
        ("a write with two subscripts covers", "if length(lhs.args) == 2\n", "if true\n", ["dispatch", "ck_cov_linear"]),
        ("the cover of an array is not widened by adjacent pieces", "while changed && length(pieces) < 24", "while false", ["dispatch", "ck_cov_union"]),
        ("the entry state of a scratch array is not put back", "return restore && !isempty(out) ? Any[Expr(:if, :(\$s == 1), Expr(:block, out...))] : out", "return restore ? Any[] : out", ["ck_solve_array"]),
        ("a read of a written array at another subscript is no carry", "        all(sb -> sb == subs[1], subs) || return true\n", "", ["ck_sweep"]),
        ("a write without the loop variable is no carry", "        any(x -> agen_fp_count_sym(x, t) > 0, subs[1]) || return true\n", "", ["ck_array"]),
        ("a scalar read before its assignment is no carry", "    any(v -> v in uev && v !== t && !(v in lvars), scal) && return true\n", "", ["ck_scalar"]),
        ("a segment length below 1 is accepted", "    guard = :(ck < 1 && error(", "    guard = :(false && error(", ["dispatch"]),
        ("the default is the checkpointed adjoint", "primal_outputs::Bool=true, checkpoint::Bool=false)\n    if checkpoint", "primal_outputs::Bool=true, checkpoint::Bool=true)\n    if checkpoint", ["dispatch"]),
        ("keep_push_pop = true is checkpointed", "    why = keep_push_pop ? \"keep_push_pop = true", "    why = false ? \"keep_push_pop = true", ["dispatch"]),
        ("fuse_ii_loops = true is checkpointed", "          fuse_ii_loops ? \"fuse_ii_loops = true (the checkpointed", "          false ? \"fuse_ii_loops = true (the checkpointed", ["dispatch"]),
    ]
    mktempdir() do tmp
        for (label, old, new, kset) in cases
            occursin(get(ENV, "CKV_SAB", ""), label) || continue
            n = length(findall(old, text))
            if n != 1
                bad += 1; verbose && println("FAIL sabotage \"$label\": anchor found $n times"); continue
            end
            path = joinpath(tmp, "STADE_sab.jl"); write(path, replace(text, old => new; count = 1))
            script = """
                include($(repr(path)))
                cd($(repr(@__DIR__)))
                include($(repr(joinpath(@__DIR__, "validate_checkpoint.jl"))))
                println("RESULT ", validate_checkpoint(; verbose = false, only = $(repr(kset))) ? 0 : 1)
            """
            out = sab_read(script)
            m = match(r"RESULT (\d)", out)
            caught = out == "ERR" || m === nothing || m.captures[1] == "1"
            caught || (bad += 1)
            verbose && println(caught ? "ok     " : "NOT OK ", "sabotage \"$label\": ", caught ? "a check fails" : "NOT CAUGHT")
        end
    end
    return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_checkpoint(; only = String[a for a in ARGS if !startswith(a, "--")], sabotage = "--sabotage" in ARGS) ? 0 : 1)
end
