# ===================== validate_baseline_yaml.jl =====================
# The baseline file of the GPU parity script holds scalars, 1-D lists, 2-D block rows and, since v0.5.26, arrays of rank 3 or more (`shape:` and `data:` in column-major order).
# Before, io_write_baseline_yaml stopped with "unsupported array rank" for the 45 `f3_nest_*` kernels of the battle corpus, so they had no GPU parity row.
#   1. ROUND TRIP: write, read, compare, for rank 0 to 6, an uneven shape and a shape with a 1.
#   2. FILE FORM: a hand-written rank-3 file (the documented form) reads as the array whose first index runs fastest.
#   3. REFUSALS: a shape without a data list, a data list that does not fit the shape, matrix rows mixed with a shape.
#   4. CORPUS: the baseline of every battle kernel of rank 3 or more is generated, written and read back equal (the kernels the old writer refused).
#   5. SABOTAGE: a reader that fills the array row-major, and a writer that drops the shape, each fail a check above.
#   julia validate_baseline_yaml.jl
isdefined(Main, :STADE) || include(joinpath(@__DIR__, "..", "src", "STADE.jl"))

function yb_roundtrip(S, vals)
    p = tempname() * ".yaml"
    S.io_write_baseline_yaml(p, :k, Dict(:n => 3), vals)
    b = S.io_read_baseline_yaml(p)
    rm(p; force = true)
    return b
end

function yb_values()
    return Dict{Symbol,Any}(:s => 2.5, :a => [1.0, -2.0, 3.5], :m => reshape(collect(1.0:6.0), 2, 3), :t => reshape(collect(1.0:24.0), 2, 3, 4),
                            :u => reshape(collect(1.0:6.0), 1, 2, 3), :q => reshape(collect(1.0:16.0), 2, 2, 2, 2), :r6 => reshape(collect(1.0:64.0), 2, 2, 2, 2, 2, 2))
end

# the checks 1 to 3 against one implementation; returns the list of results
function yb_checks(S)
    r = Bool[]
    v = yb_values()
    b = try yb_roundtrip(S, v) catch; nothing end
    push!(r, b !== nothing && all(haskey(b.values, k) && b.values[k] == v[k] && size(b.values[k]) == size(v[k]) for k in keys(v)))
    p = tempname() * ".yaml"
    write(p, "int_args:\n  n: 2\nvalues:\n  t:\n    shape: [2, 3, 1]\n    data: [1, 2, 3, 4, 5, 6]\n  z: 1.0\n")
    t = S.io_read_baseline_yaml(p).values[:t]
    push!(r, size(t) == (2, 3, 1) && t[2, 1, 1] == 2.0 && t[1, 2, 1] == 3.0 && t[2, 3, 1] == 6.0)
    function refused(txt, frag)
        write(p, txt)
        try S.io_read_baseline_yaml(p); return false catch e; return occursin(frag, sprint(showerror, e)) end
    end
    push!(r, refused("int_args:\n  n: 2\nvalues:\n  t:\n    shape: [2, 3]\n", "needs both a shape and a data list"))
    push!(r, refused("int_args:\n  n: 2\nvalues:\n  t:\n    shape: [2, 3, 2]\n    data: [1, 2, 3]\n", "3 data values for the shape [2, 3, 2]"))
    push!(r, refused("int_args:\n  n: 2\nvalues:\n  t:\n    - [1, 2]\n    shape: [1, 2]\n    data: [1, 2]\n", "no matrix rows"))
    rm(p; force = true)
    return r
end

# phc_colebrook baseline with float_range, against one implementation of STADE
function yb_range_ok(S)
    cp = joinpath(@__DIR__, "counted-corpus", "phc_colebrook.jl")
    fr = Dict{Symbol,Tuple{Float64,Float64}}(:re => (1e4, 1e6), :rough => (1e-5, 1e-3))
    yp = tempname() * ".gpu.yaml"
    got = try
        S.stade_generate_baseline_file(cp; yaml_path = yp, int_lo = 3, int_hi = 4, float_range = fr)
        b = S.io_read_baseline_yaml(yp)
        all(1e4 <= x <= 1e6 for x in b.values[:re]) && all(1e-5 <= x <= 1e-3 for x in b.values[:rough])
    catch e
        false
    end
    rm(yp; force = true)
    return got
end

function validate_baseline_yaml()
    ok = true
    check(c, msg) = (println(c ? "ok   " : "FAIL ", msg); ok &= c)
    labels = ["round trip: scalar, 1-D, 2-D, 3-D (uneven), a shape with a 1, 4-D, 6-D", "a hand-written rank-3 file: the first index runs fastest",
              "refused: a shape without data", "refused: data that does not fit the shape", "refused: matrix rows mixed with a shape"]
    base = yb_checks(STADE)
    for (k, l) in enumerate(labels); check(base[k], l); end
    # the battle kernels of rank 3 or more
    dir = joinpath(@__DIR__, "battle-corpus")
    names = sort([splitext(f)[1] for f in readdir(dir) if startswith(f, "f3_nest_") && endswith(f, ".jl") && !occursin(r"_(b|d|hv)\.jl$", f)])
    nok = 0; nbad = String[]
    for nm in names
        yp = tempname() * ".gpu.yaml"
        try
            STADE.stade_generate_baseline_file(joinpath(dir, nm * ".jl"); yaml_path = yp, int_lo = 3, int_hi = 4)
            b = STADE.io_read_baseline_yaml(yp)
            b2 = yb_roundtrip(STADE, b.values)
            all(b2.values[k] == b.values[k] for k in keys(b.values)) && any(v isa AbstractArray && ndims(v) >= 3 for v in values(b.values)) ? (nok += 1) : push!(nbad, nm)
        catch e
            push!(nbad, nm * ": " * first(sprint(showerror, e), 80))
        end
        rm(yp; force = true)
    end
    check(length(names) >= 40 && isempty(nbad), "the f3_nest battle kernels ($(length(names))): baseline generated, written and read back equal ($(nok) ok$(isempty(nbad) ? "" : "; bad: " * join(nbad[1:min(3, end)], " | ")))")
    # float_range: phc_colebrook takes a logarithm of a positive quantity that a normal draw makes negative
    cp = joinpath(@__DIR__, "counted-corpus", "phc_colebrook.jl")
    yp = tempname() * ".gpu.yaml"
    got = yb_range_ok(STADE)
    plain = try STADE.stade_generate_baseline_file(cp; yaml_path = yp, int_lo = 3, int_hi = 4); false catch e; occursin("DomainError", sprint(showerror, e)) end
    check(plain, "without float_range phc_colebrook still fails with the domain error")
    badname = try STADE.stade_generate_baseline_file(cp; yaml_path = yp, float_range = Dict{Symbol,Tuple{Float64,Float64}}(:n => (1.0, 2.0))); false catch e; occursin("not a float argument", sprint(showerror, e)) end
    check(badname, "float_range: a name that is not a float argument is refused")
    rm(yp; force = true)
    # sabotage: a modified copy of STADE.jl, loaded into a scratch module, must fail the checks
    text = read(joinpath(@__DIR__, "..", "src", "STADE.jl"), String)
    function sabotaged(old, new)
        count(old, text) == 1 || return nothing
        m = Module(:YBSab)
        Base.include_string(m, replace(text, old => new; count = 1), "STADE_sab.jl")
        return m
    end
    for (label, old, new, idx) in (("a reader that fills the array row-major", "values[pending_key] = reshape(pending_data, pending_shape...)", "values[pending_key] = permutedims(reshape(pending_data, reverse(pending_shape)...), reverse(1:length(pending_shape)))", (1, 2)),
                                   ("a writer that drops the shape line", "push!(lines, \"    shape: [\" * join(size(v), \", \") * \"]\")", "nothing", (1,)))
        m = sabotaged(old, new)
        if m === nothing
            check(false, "sabotage: $label: anchor not found once"); continue
        end
        res = try Base.invokelatest(Core.eval, m, :(Main.yb_checks(STADE))) catch e; nothing end
        # an exception is no evidence: the checks must run to the end and report the failure themselves
        check(res !== nothing && length(res) == 5 && any(i -> !res[i], idx) && all(res[i] for i in 3:5), "sabotage: $label fails check(s) $(join(idx, ", ")) and the refusals still hold")
    end
    m = sabotaged("lo + (hi - lo) * rand() : lo .+ (hi - lo) .* rand(shapes[a]...)", "scale * val_randn() : scale .* val_randn(shapes[a]...)")
    if m === nothing
        check(false, "sabotage: a draw that ignores float_range: anchor not found once")
    else
        res = try Base.invokelatest(Core.eval, m, :(Main.yb_range_ok(STADE))) catch e; nothing end
        check(res === false, "sabotage: a draw that ignores float_range fails the range check")
    end
    println(ok ? "ALL OK" : "FAILED")
    return ok
end
get(ENV, "YB_NORUN", "") == "1" || (validate_baseline_yaml() || exit(1))
