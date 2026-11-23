# Regression for v0.5.35: stade_adjoint_split, the adjoint as a forward function that records the tape and a backward function that only reads it.
#   The use: a fixed-point adjoint runs the backward sweep many times on one recording, with a new seed each time (Christianson's two-step method, Tapenade's
#   FP-LOOP). The forward sweep does not depend on the seed, so it runs once.
#   For every kernel of every corpus under test/ whose full adjoint exists with keep_push_pop = false, on a random draw:
#   A  forward then backward equals the full adjoint in every argument, every stack and the returned scalars, bit for bit;
#   B  the backward sweep leaves every indexed tape array unchanged, and changes only the host stacks listed in `consume`;
#   C  every primal array the backward sweep changes is in `restore`;
#   D  a second backward sweep with a second seed, on the restored arrays and fresh copies of the `consume` stacks, equals a full adjoint call with that seed, bit for bit.
#   Plus unit checks of the guards that no corpus kernel reaches, and the sabotage cases.
isdefined(@__MODULE__, :STADE) || include(joinpath(dirname(@__DIR__), "src", "STADE.jl"))
isdefined(@__MODULE__, :sab_read) || include(joinpath(@__DIR__, "sabotage_run.jl"))

sp_same(a, b) = a isa AbstractArray ? (b isa AbstractArray && size(a) == size(b) && all(isequal.(a, b))) : isequal(a, b)
sp_finite(x) = x isa AbstractArray ? all(isfinite, x) : (x isa Number ? isfinite(x) : true)

# position of every argument in the positional call of a generated adjoint: a float argument is followed by its adjoint
function sp_positions(sig)
    pos = Dict{Symbol,Int}(); shpos = Dict{Symbol,Int}(); p = 1
    for a in sig.args
        pos[a] = p
        if sig.kinds[a] in (:scalar_float, :array_float)
            shpos[a] = p + 1; p += 2
        else
            p += 1
        end
    end
    return pos, shpos, p - 1
end

# one kernel; returns (status, detail). status: :ok, :refused (the full adjoint is refused too), :split_refused, :nodraw, :skipped, :fail
function sp_kernel(path; verbose = false)
    # a generated sweep sitting in a corpus is not a kernel
    primal = try STADE.io_read_corpus_entry(path) catch; return (:skipped, "") end
    kernel = try STADE.parse_kernel(primal) catch; return (:skipped, "") end
    sig = kernel.sig
    full = try STADE.agen_adjoint_unrolled(primal; keep_push_pop = false) catch; return (:refused, "") end
    sp = try STADE.stade_adjoint_split(primal) catch e; return (:split_refused, first(split(sprint(showerror, e), "\n"))) end
    pos, shpos, nsig = sp_positions(sig)
    comp(e) = STADE.val_compile(e)
    fullfn = comp(full.adjoint); fullinit = comp(full.initstacks)
    spinit = comp(sp.initstacks); fwd = comp(sp.forward); bwd = comp(sp.backward)
    carryfn = sp.carry_init === nothing ? nothing : comp(sp.carry_init)
    # a draw with a finite result of the full adjoint, for two seeds
    for attempt in 1:8
        baseline = try STADE.val_generate_baseline(kernel, primal; self_check = false) catch; return (:nodraw, "") end
        STADE.val_coerce_int_arrays!(kernel, baseline.values)
        int_args = baseline.int_args; vals = baseline.values
        seed1 = STADE.val_random_values_like(kernel, vals); seed2 = STADE.val_random_values_like(kernel, vals)
        names_full = Symbol[a for a in full.initstacks.args[1].args[2:end]]
        names_sp = Symbol[a for a in sp.initstacks.args[1].args[2:end]]
        extra(names) = [haskey(int_args, n) ? int_args[n] : deepcopy(vals[n]) for n in names]
        # a tape is allocated without initial contents: fill it, so the slots a run never writes compare equal
        zeroed(t) = (foreach(x -> x isa AbstractArray && fill!(x, zero(eltype(x))), t); t)
        runfull(seed) = begin
            stk = zeroed(STADE.val_init_stacks(fullinit, extra(names_full)))
            call = STADE.val_adjoint_call_args(sig, int_args, vals, seed, stk)
            ret = Base.invokelatest(fullfn, call...)
            (call = call, ret = ret, stk = stk)
        end
        local f1, f2
        try
            f1 = runfull(seed1); f2 = runfull(seed2)
        catch
            continue
        end
        all(sp_finite, f1.call) && all(sp_finite, f2.call) && sp_finite(f1.ret) || continue
        # split: forward, then backward on the same arguments
        stk = zeroed(STADE.val_init_stacks(spinit, extra(names_sp)))
        call = STADE.val_adjoint_call_args(sig, int_args, vals, seed1, stk)
        carry = carryfn === nothing ? () : Base.invokelatest(carryfn)
        try
            Base.invokelatest(fwd, call..., carry...)
        catch e
            return (:fail, "forward sweep raised " * first(split(sprint(showerror, e), "\n")))
        end
        tape = deepcopy(stk)
        names_call = Symbol[x for x in sp.forward.args[1].args[2:end] if x !== :__carry_f && x !== :__carry_i][nsig + 1:end]
        stkpos = Dict(nm => i for (i, nm) in enumerate(names_call))
        arrays = [a for a in sig.args if sig.kinds[a] in (:array_float, :array_int)]
        after_fwd = Dict(a => copy(call[pos[a]]) for a in arrays)
        ret = try Base.invokelatest(bwd, call..., carry...) catch e
            return (:fail, "backward sweep raised " * first(split(sprint(showerror, e), "\n")))
        end
        # A: the composition is the full adjoint
        for i in 1:nsig
            sp_same(call[i], f1.call[i]) || return (:fail, "A: argument $i differs from the full adjoint")
        end
        sp_same(ret, f1.ret) || return (:fail, "A: the returned adjoints of the scalars differ")
        for (i, nm) in enumerate(names_call)
            nm in sp.consume || sp_same(stk[i], f1.stk[i]) || return (:fail, "A: the stack `$nm` differs from the full adjoint's")
        end
        # B: the tape is unchanged by the backward sweep, but for the host stacks it pops
        for (i, nm) in enumerate(names_call)
            nm in sp.consume || sp_same(stk[i], tape[i]) || return (:fail, "B: the backward sweep changed `$nm`, which `consume` does not list")
        end
        # C: the arrays the backward sweep changed are listed
        for a in arrays
            a in sp.restore || sp_same(call[pos[a]], after_fwd[a]) || return (:fail, "C: the backward sweep changed `$a`, which `restore` does not list")
        end
        # D: a second backward sweep with a second seed
        rcall = Any[]
        for (i, x) in enumerate(call)
            push!(rcall, x)
        end
        for a in sig.args
            k = sig.kinds[a]
            if k in (:array_float, :array_int)
                rcall[pos[a]] = a in sp.restore ? copy(after_fwd[a]) : call[pos[a]]
            end
            if k in (:scalar_float, :array_float)
                rcall[shpos[a]] = deepcopy(seed2[a])
            end
            k == :scalar_float && (rcall[pos[a]] = vals[a])
        end
        for nm in sp.consume
            rcall[nsig + stkpos[nm]] = copy(tape[stkpos[nm]])
        end
        ret2 = try Base.invokelatest(bwd, rcall..., carry...) catch e
            return (:fail, "second backward sweep raised " * first(split(sprint(showerror, e), "\n")))
        end
        for i in 1:nsig
            sp_same(rcall[i], f2.call[i]) || return (:fail, "D: argument $i of the second sweep differs from a full adjoint with the second seed")
        end
        sp_same(ret2, f2.ret) || return (:fail, "D: the returned adjoints of the second sweep differ")
        # signal: the two seeds must give different adjoints, or D compared the same numbers twice
        differs = any(a -> !sp_same(f1.call[shpos[a]], f2.call[shpos[a]]), keys(shpos))
        differs || continue
        return (:ok, "carry $(length(sp.carry_f) + length(sp.carry_i)) restore $(length(sp.restore)) consume $(length(sp.consume))")
    end
    return (:nodraw, "")
end

# the guards that no corpus kernel reaches
function sp_unit_checks(say)
    k = STADE.parse_kernel(:(function sp_u(x, n)
        x[1] = 1.0
        return nothing
    end))
    # a forward half that reads an adjoint would depend on the seed
    ok = try
        STADE.agen_split_emit(k, [:x, :xb, :n], Any[], Any[], Any[:(xb[1] = 1.0)], Any[], Any[], Symbol[], Symbol[]); false
    catch e
        occursin("reads the adjoint", sprint(showerror, e))
    end
    say(ok, "a forward sweep that reads an adjoint is refused")
    # a host stack the backward half pops is listed, and an indexed tape array is not
    r = STADE.agen_split_emit(k, [:x, :xb, :n, :t, :u], Any[], Any[], Any[:(push!(t, 1.0))], Any[:(pop!(t)), :(w = u[1])], Any[], Symbol[:t, :u], Symbol[])
    say(r.consume == [:t], "a host stack the backward half pops is in `consume`, an array it only reads is not (got $(r.consume))")
    # the carry holds only Float64 and Int64 scalars
    kk = STADE.parse_kernel(:(function sp_v(x, n)
        for i = 1:n
            x[i] = 2.0 * x[i]
        end
        return nothing
    end))
    kk = (sig = (name = kk.sig.name, args = kk.sig.args, kinds = merge(kk.sig.kinds, Dict(:zz => :array_float)), independents = kk.sig.independents, dependents = kk.sig.dependents), body = kk.body)
    ok = try
        STADE.agen_split_emit(kk, [:x, :xb, :n], Any[:(zz = 1.0)], Any[], Any[], Any[:(xb[1] = zz[1])], Any[], Symbol[], Symbol[]); false
    catch e
        occursin("carries only Float64 and Int64 scalars", sprint(showerror, e))
    end
    say(ok, "a carried name that is not a scalar is refused")
    # the analyses on a hand-built sweep
    exp = Set{Symbol}()
    STADE.agen_split_uev!(:(begin
        a = 1.0
        for i = 1:n
            b = a + c
            d = b
        end
        if p > 0
            e = 1.0
        else
            f = 1.0
        end
        g = d + e + f + a
    end), Set{Symbol}(), exp)
    say(exp == Set([:n, :c, :p, :d, :e, :f]), "upward-exposed uses: a name a loop assigns is not defined after it, and an if defines what both arms assign (got $(sort(collect(exp))))")
    top = Set{Symbol}()
    STADE.agen_split_top_assigned!(Any[:(a = 1.0), :(for i = 1:n; b = 2.0; end), :(if p > 0; c = 1.0; end)], top)
    say(top == Set([:a, :c]), "function-scope names: a loop body's assignment is not at function scope (got $(sort(collect(top))))")
end

function validate_adjoint_split(; verbose = true, only = String[], sabotage = false)
    bad = 0; checks = 0
    say(ok, msg) = (checks += 1; ok || (bad += 1; verbose && println("NOT OK ", msg)))
    root = @__DIR__
    sp_unit_checks(say)
    counts = Dict(:ok => 0, :refused => 0, :split_refused => 0, :nodraw => 0, :fail => 0, :skipped => 0)
    carried = 0; restored = 0; consumed = 0; split_refusals = String[]
    for d in sort(filter(d -> endswith(d, "-corpus") && isdir(joinpath(root, d)), readdir(root))), f in sort(readdir(joinpath(root, d)))
        endswith(f, ".jl") || continue
        nm = f[1:end-3]
        isempty(only) || nm in only || continue
        (st, detail) = try sp_kernel(joinpath(root, d, f)) catch e; (:fail, "raised " * first(split(sprint(showerror, e), "\n"))) end
        counts[st] += 1
        st == :split_refused && push!(split_refusals, "$d/$nm: $detail")
        if st == :ok
            m = match(r"carry (\d+) restore (\d+) consume (\d+)", detail)
            parse(Int, m.captures[1]) > 0 && (carried += 1)
            parse(Int, m.captures[2]) > 0 && (restored += 1)
            parse(Int, m.captures[3]) > 0 && (consumed += 1)
        end
        say(st != :fail, "$d/$nm: $detail")
    end
    verbose && println("validate_adjoint_split: ", join(("$k=$v" for (k, v) in sort(collect(counts))), " "), "  carried=$carried restored=$restored consumed=$consumed")
    isempty(only) && say(carried > 0 && restored > 0 && consumed > 0, "no kernel needed a carry, a restore and a consumed stack: the checks above compared nothing that matters")
    verbose && for r in split_refusals; println("  split refused: ", r); end
    if sabotage
        bad += sp_sabotage(; verbose = verbose) ? 0 : 1
    end
    verbose && println("validate_adjoint_split: $(checks - bad)/$checks ok")
    return bad == 0
end

function sp_sabotage(; verbose = true)
    bad = 0
    text = read(joinpath(dirname(@__DIR__), "src", "STADE.jl"), String)
    cases = [
        ("no scalar is carried", [("carry = sort(collect(setdiff(intersect(exposed, fwd_scope), skip)))", "carry = Symbol[]")], ["sp_carry"]),
        ("the forward half does not store its carried floats", [("push!(fwd_body, :(__carry_f[\$k] = \$v))", "nothing")], ["sp_carry"]),
        ("the backward half does not load its carried floats", [("push!(bwd_body, :(\$v = __carry_f[\$k]))", "nothing")], ["sp_carry"]),
        ("a carried name starts undefined", [("v in pinit_names || push!(fwd_body, Expr(:(=), v, v in carry_f ? 0.0 : 0))", "nothing")], ["sp_branch_off"]),
        ("an if defines what either arm assigns", [("union!(defd, intersect(d1, d2))", "union!(defd, union(d1, d2))")], ["sp_carry"]),
        ("what a loop body assigns counts after the loop", [("inner = copy(defd)\n            push!(inner, spec.args[1])", "inner = defd\n            push!(inner, spec.args[1])")], ["sp_carry"]),
        ("no array is restored", [("restore = Symbol[a for a in sig.args if a in written && sig.kinds[a] in (:array_float, :array_int)]", "restore = Symbol[]")], ["sp_restore"]),
        ("the shadow init stays in the forward half", [("    append!(bwd_body, sinit)\n    append!(bwd_body, bwd)", "    append!(bwd_body, bwd)"), ("    append!(fwd_body, pinit)\n    append!(fwd_body, fwd)", "    append!(fwd_body, pinit)\n    append!(fwd_body, sinit)\n    append!(fwd_body, fwd)")], ["sp_carry", "sp_restore"]),
        ("the forward half may read an adjoint", [("isempty(leaked) || error(", "true || error(")], ["sp_carry"]),
        ("a popped host stack is not listed", [("consume = Symbol[nm for nm in vcat(stack_names, tier_b_extra_args) if nm in popped]", "consume = Symbol[]")], ["phc_mg_solve"]),
        ("a non-scalar carry is allowed", [("isempty(odd) || error(", "true || error(")], ["sp_carry"]),
    ]
    mktempdir() do tmp
        for (label, edits, kset) in cases
            isempty(get(ENV, "SP_SAB_ONLY", "")) || any(t -> occursin(t, label), split(ENV["SP_SAB_ONLY"], "|")) || continue
            t = text; okanchor = true
            for (old, new) in edits
                n = length(findall(old, t))
                n == 1 || (okanchor = false; verbose && println("NOT OK sabotage \"$label\": anchor found $n times"); bad += 1)
                t = replace(t, old => new; count = 1)
            end
            okanchor || continue
            get(ENV, "SAB_ANCHORS_ONLY", "") == "1" && continue
            path = joinpath(tmp, "STADE_sab.jl"); write(path, t)
            script = """
                include($(repr(path)))
                cd($(repr(@__DIR__)))
                include($(repr(joinpath(@__DIR__, "validate_adjoint_split.jl"))))
                println("RESULT ", validate_adjoint_split(; verbose = false, only = $(repr(kset))) ? 0 : 1)
            """
            out = sab_read(script)
            m = match(r"RESULT (\d)", out)
            caught = out == "ERR" || m === nothing || m.captures[1] == "1"
            caught || (bad += 1)
            verbose && println(caught ? "ok     " : "NOT OK ", "sabotage \"$label\": ", caught ? "a check fails" : "NOT CAUGHT")
        end
    end
    return bad == 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(validate_adjoint_split(; only = String[a for a in ARGS if !startswith(a, "--")], sabotage = "--sabotage" in ARGS) ? 0 : 1)
end
