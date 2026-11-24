# Fuzz of the entry-value rule of the fixed-point adjoint (agen_fp_stale).
# A generated kernel holds one time loop (or none) with one or two solve loops and random statements around them: updates, kills, reads, branches on the state,
# a second state that reads the first as a parameter. The dispatch (agen_fp_adjoint_or_nothing) decides each kernel. For every kernel:
#   accepted  the fixed-point adjoint must equal a central difference of the primal (to 1e-4), and the unrolled adjoint where the guard is closed on entry in every shadow, in the converged regime and in the regime where the guard
#             is closed on entry (an accepted kernel with a different result is a defect)
#   refused   for the entry-value reason only: the fixed-point adjoint is forced, and its result against the unrolled adjoint says whether the refusal was needed
#             (different: needed; equal: the rule is conservative on this shape)
#   julia fuzz_fp_entry.jl [count] [seed] [family: scalar|array|both]
isdefined(@__MODULE__, :STADE) || include(joinpath(dirname(@__DIR__), "src", "STADE.jl"))
isdefined(@__MODULE__, :fpv_run_expr) || include(joinpath(@__DIR__, "validate_fixedpoint.jl"))
using Random, Logging

# one solve loop of a scalar state `S`: contraction `S = c cos(S) + p + extra`, cap `cap`, local names suffixed by `sfx`
fzz_solve_scalar(S, sfx, cap, extra) = """
rn$sfx = 1.0
for kk$sfx = 1:$cap
    if rn$sfx > tol
        xn$sfx = c * cos($S) + p$extra
        rn$sfx = (xn$sfx - $S) * (xn$sfx - $S)
        $S = xn$sfx
    end
end
"""

# one solve loop of an array state `U` (buffer `W`, forcing `F`): `U[j] = c tanh(U[j-1] + U[j+1]) + F[j] + extra`, interior j = 2:n+1
fzz_solve_array(U, W, F, sfx, cap, extra) = """
rn$sfx = 1.0
for kk$sfx = 1:$cap
    if rn$sfx > tol
        for i$sfx = 1:n
            j$sfx = i$sfx + 1
            $W[j$sfx] = c * tanh($U[j$sfx - 1] + $U[j$sfx + 1]) + $F[j$sfx]$extra
        end
        rn$sfx = 0.0
        for i2$sfx = 1:n
            j2$sfx = i2$sfx + 1
            rn$sfx = rn$sfx + ($W[j2$sfx] - $U[j2$sfx]) * ($W[j2$sfx] - $U[j2$sfx])
        end
        for i3$sfx = 1:n
            j3$sfx = i3$sfx + 1
            $U[j3$sfx] = $W[j3$sfx]
        end
    end
end
"""

# the statements that can sit around a solve of the scalar state `A` (the other scalar state is `B`)
fzz_pool_scalar(A, B) = String[
    "$A = $A + 0.1 * pa[t]\n",
    "$A = $A * (1.0 + 0.1 * pa[t])\n",
    "$A = 0.5 + 0.1 * pa[t]\n",
    "$A = $(A)0 + 0.1 * pa[t]\n",
    "ys[t] = ys[t] + $A\n",
    "ys[t] = ys[t] + $A * $A\n",
    "yt = $A * $A\nys[t] = ys[t] + yt * yt\n",
    "if $A > 0.5\n    ys[t] = ys[t] + $A\nend\n",
    "if $A > 0.5\n    $A = $A + 0.2\nend\n",
    "$B = $B + 0.5 * $A\n",
    "$B = $B * $A\n",
    "$A = $A + $B\n",
    "$A = $A * $B\n",
    "ys[t] = ys[t] + cos($A)\n",
    "ys[t] = ys[t] + pa[t] / (5.0 + $A)\n",
]

# the same for the array state `A` (the other is `B`), interior 2:n+1, boundary 1 and n+2 (never written by a solve)
fzz_pool_array(A, B) = String[
    "$A[3] = $A[3] + 0.1 * pa[t]\n",
    "$A[3] = $A[3] * (1.0 + 0.1 * pa[t])\n",
    "$A[3] = 0.4 + 0.1 * pa[t]\n",
    "$A[1] = 0.4 * pa[t]\n",
    "for i = 1:n + 2\n    $A[i] = 0.2 * pa[t] + 0.1 * f[i]\nend\n",
    "for i = 2:n + 1\n    $A[i] = $A[i] * 0.9\nend\n",
    "for i = 1:n + 2\n    $A[i] = $A[i] + 0.1 * pa[t]\nend\n",
    "ys[t] = ys[t] + $A[2] * $A[3]\n",
    "ys[t] = ys[t] + $A[1] * $A[2]\n",
    "for i = 1:n + 2\n    ys[t] = ys[t] + $A[i] * $A[i]\nend\n",
    "yt = $A[2] * $A[3]\nys[t] = ys[t] + yt * yt\n",
    "if $A[2] > 0.5\n    ys[t] = ys[t] + $A[3]\nend\n",
    "$B[3] = $B[3] + 0.5 * $A[2]\n",
    "for i = 1:n + 2\n    $B[i] = $B[i] + 0.5 * $A[i] * $A[i]\nend\n",
    "ys[t] = ys[t] + tanh($A[2])\n",
    "ys[t] = ys[t] + pa[t] / (5.0 + $A[2])\n",
]

fzz_ind(txt, k) = join([isempty(l) ? l : "    "^k * l for l in split(chomp(txt), "\n")], "\n") * "\n"

# a random kernel: (text, description). `rng` decides everything.
function fzz_kernel(rng, family::Symbol, id::Int)
    scalar = family === :scalar
    nest = rand(rng, [:none, :loop1, :loop1, :loop1, :ifbranch])
    nsolve = rand(rng, [1, 2, 2])
    # the solves: (state, other state, cap, extra parameter)
    states = scalar ? [("x", "z"), ("z", "x")] : [("u", "v"), ("v", "u")]
    items = Any[]
    for s in 1:nsolve
        st, ot = nsolve == 1 ? states[1] : (s == 1 ? states[1] : rand(rng, [states[1], states[2]]))
        extra = rand(rng, ["", " + 0.3 * pa[t]", scalar ? " + 0.2 * $ot" : " + 0.2 * $(ot)[j$(s == 1 ? "a" : "b")]"])
        cap = s == 1 ? "maxit" : rand(rng, ["maxit", "maxq"])
        sfx = s == 1 ? "a" : "b"
        txt = scalar ? fzz_solve_scalar(st, sfx, cap, extra) :
              fzz_solve_array(st, st == "u" ? "w" : "w2", st == "u" ? "f" : "f2", sfx, cap, extra)
        push!(items, (:solve, txt, st))
    end
    nstmt = rand(rng, 2:5)
    for _ in 1:nstmt
        a, b = rand(rng, states)
        pool = scalar ? fzz_pool_scalar(a, b) : fzz_pool_array(a, b)
        push!(items, (:stmt, rand(rng, pool), a))
    end
    STADE.val_shuffle!(rng, items)
    # a copy of a state taken early and used late: its use needs the copy, not the state
    if rand(rng) < 0.5
        a = rand(rng, states)[1]
        i1 = rand(rng, 1:length(items) + 1); insert!(items, i1, (:stmt, scalar ? "sv = $a\n" : "sv = $a[2]\n", a))
        i2 = rand(rng, i1 + 1:length(items) + 1); insert!(items, i2, (:stmt, "ys[t] = ys[t] + sv * sv\n", a))
    end
    # a solve of the second state needs the second state to exist: the pool above only uses states of the family, fine
    body = "ys[t] = 0.0\n" * join([it[2] for it in items])
    if nest === :ifbranch
        # the whole body under a branch on the input: the solves then run in some instances only
        b2 = "ys[t] = 0.0\nif pa[t] > 0.5\n" * fzz_ind(join([it[2] for it in items]), 1) * "end\n"
        body = b2
    end
    head = scalar ? "x = x0\nz = z0\n" : ""
    tail = scalar ? "out[1] = x * z + x\n" : "out[1] = u[2] * u[3] + v[2]\n"
    b = nest === :none ? head * replace(body, "[t]" => "[1]") * tail : head * "for t = 1:nt\n" * fzz_ind(body, 1) * "end\n" * tail
    order = ["x0", "z0", "u", "v", "w", "w2", "f", "f2", "c", "p", "pa", "ys", "out", "tol", "n", "nt", "maxit", "maxq"]
    used = [a for a in order if occursin(Regex("\\b" * a * "\\b"), b)]
    # an array argument that no statement reads cannot be a kernel argument that is also state: keep what the text names
    name = "fzz_$(family)_$id"
    txt = "function $name($(join(used, ", ")))\n" * fzz_ind(b, 1) * "    return nothing\nend\n"
    return txt
end

# (verdict, err) of one kernel text: verdict :accept, :refuse_stale (the entry-value reason only), :refuse_other, :none (no solve found), :error
function fzz_one(txt::String; forced::Bool = true, draws = (20261122, 11, 22, 33))
    primal = Meta.parse(txt)
    kernel = STADE.parse_kernel(primal)
    args = Set(kernel.sig.args)
    scalar = :x0 in args
    pins = Dict(:maxit => 500, :maxq => 500, :nt => 3, :n => 6)
    floats = Dict(:c => 0.5, :tol => 1e-30)
    pins = Dict(k => v for (k, v) in pins if k in args)
    floats = Dict(k => v for (k, v) in floats if k in args)
    # distinct forcing per step: with equal values two instances of a solve reach the same fixed point, the second starts on it, and the unrolled reverse of that one
    # iteration is not the derivative of the fixed point (a finite difference agrees with the fixed-point adjoint there, not with the unrolled one)
    prep = :pa in args ? (v -> (v[:pa] .= [(0.9, 0.3, 0.7)[mod1(i, 3)] for i in eachindex(v[:pa])])) : nothing
    dets = STADE.agen_fp_detect(primal, kernel.sig.kinds; kernel = kernel)
    isempty(dets) && return (verdict = :none, why = "", errs = Float64[])
    why = unique(vcat([STADE.agen_fp_refusal(d) for d in dets]...))
    nstale = count(w -> startswith(w, "the state"), why)
    verdict = isempty(why) ? :accept : (nstale > 0 && nstale == length(why) ? :refuse_stale : :refuse_other)
    verdict === :refuse_other && return (verdict = verdict, why = join(why, " | "), errs = Float64[])
    verdict === :refuse_stale && !forced && return (verdict = verdict, why = join(why, " | "), errs = Float64[])
    # the judge in the converged regime is the central difference of the primal: where an instance starts on the fixed point (two solves of one state with the same
    # parameters, or equal forcing at two steps) the unrolled reverse of its one iteration is not the derivative of the fixed point, and the fixed-point adjoint is.
    # Several draws of the inputs: a branch on the state, or a restore the reverse happens to get right, can hide a defect in one draw.
    errs = Float64[]
    for reg in (:converged, :zero)
        worst = 0.0
        for sd in draws
            r = fpv_run_expr(primal, pins, floats, scalar ? 4 : 8, prep, reg; fd = reg === :converged, seed = sd)
            e = isempty(r.raised) ? (reg === :converged ? r.fderr * 1e-2 : r.err) : NaN   # a finite-difference error of 1e-4 counts like 1e-6, the bar below
            worst = isnan(e) || isnan(worst) ? NaN : max(worst, e)
        end
        push!(errs, worst)
    end
    return (verdict = verdict, why = join(why, " | "), errs = errs)
end

# the kernels that read the state in one way each, before a solve of it in a time loop (the solve's fixed point changes with t): one per kind of read
function fzz_directed(family::Symbol)
    scalar = family === :scalar
    reads = scalar ? ["ys[t] = ys[t] + x\n", "ys[t] = ys[t] + x * x\n", "ys[t] = ys[t] + cos(x)\n", "ys[t] = ys[t] + pa[t] / (5.0 + x)\n", "if x > 0.5\n    ys[t] = ys[t] + x\nend\n"] :
                     ["ys[t] = ys[t] + u[2]\n", "ys[t] = ys[t] + u[2] * u[3]\n", "ys[t] = ys[t] + tanh(u[2])\n", "ys[t] = ys[t] + pa[t] / (5.0 + u[2])\n", "if u[2] > 0.5\n    ys[t] = ys[t] + u[3]\nend\n"]
    out = String[]
    for (k, r) in enumerate(reads)
        solve = scalar ? fzz_solve_scalar("x", "a", "maxit", " + 0.3 * pa[t]") : fzz_solve_array("u", "w", "f", "a", "maxit", " + 0.3 * pa[t]")
        head = scalar ? "x = x0\nz = z0\n" : ""
        tail = scalar ? "out[1] = x * z + x\n" : "out[1] = u[2] * u[3] + v[2]\n"
        b = head * "for t = 1:nt\n" * fzz_ind("ys[t] = 0.0\n" * r * solve, 1) * "end\n" * tail
        order = ["x0", "z0", "u", "v", "w", "f", "c", "p", "pa", "ys", "out", "tol", "n", "nt", "maxit"]
        used = [a for a in order if occursin(Regex("\\b" * a * "\\b"), b)]
        push!(out, "function fzz_dir_$(family)_$k($(join(used, ", ")))\n" * fzz_ind(b, 1) * "    return nothing\nend\n")
    end
    return out
end

function fzz_main(count::Int, seed::Int, family_arg::String; verbose::Bool = true, draws = (20261122, 11, 22, 33), stop_on_defect::Bool = false, directed::Bool = false)
    rng = Xoshiro(seed)
    fams = family_arg == "both" ? [:scalar, :array] : [Symbol(family_arg)]
    tally = Dict{Symbol,Int}(); bad = Any[]; stale_needed = 0; stale_spare = 0
    seen = Set{String}()
    dir = directed ? vcat([fzz_directed(f) for f in fams]...) : String[]
    for id in 1:count + length(dir)
        fam = fams[mod1(id, length(fams))]
        txt = id <= count ? fzz_kernel(rng, fam, id) : dir[id - count]
        h = replace(txt, r"function fzz_\w+" => "function f")
        h in seen && continue; push!(seen, h)
        res = try
            with_logger(NullLogger()) do; fzz_one(txt; draws = draws); end
        catch e
            (verdict = :error, why = first(split(sprint(showerror, e), "\n")), errs = Float64[])
        end
        tally[res.verdict] = get(tally, res.verdict, 0) + 1
        worst = isempty(res.errs) ? 0.0 : maximum(res.errs)
        if res.verdict === :accept && !(worst < 1e-6)
            push!(bad, (id, txt, res)); println("DEFECT (accepted, wrong): id=$id errs=$(res.errs)\n", txt)
            stop_on_defect && (println("FAIL fuzz defect id=$id"); break)
        elseif res.verdict === :error
            push!(bad, (id, txt, res)); println("ERROR: id=$id ", res.why, "\n", txt)
        elseif res.verdict === :refuse_stale
            worst < 1e-6 ? (stale_spare += 1) : (stale_needed += 1)
            verbose && println("stale id=$id ", worst < 1e-6 ? "SPARE   " : "NEEDED  ", res.errs)
        end
    end
    println("tally ", tally, "  stale refusals needed=", stale_needed, " spare=", stale_spare, "  defects=", length(bad))
    return (bad = bad, tally = tally, needed = stale_needed, spare = stale_spare)
end

if abspath(PROGRAM_FILE) == @__FILE__
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 40
    sd = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
    fam = length(ARGS) >= 3 ? ARGS[3] : "both"
    r = fzz_main(n, sd, fam)
    exit(isempty(r.bad) ? 0 : 1)
end
