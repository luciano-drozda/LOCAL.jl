# Kernels for the checkpointed adjoint: one loop over time steps (a region) in each shape the transformation can meet, and the loops it must refuse.
# Each entry is (name, source, expectation) where the expectation is :region or the text of the reason the loop is not a region (a part of the @info line).
# Tests build the kernels from here, nothing in STADE.jl reads this file.
isdefined(@__MODULE__, :STADE) || include(joinpath(dirname(@__DIR__), "src", "STADE.jl"))

ckp_shapes() = Any[
    # a carried array, in place
    ("ck_array", """
function ck_array(u, a, out, n, nt)
    for t = 1:nt
        for i = 1:n
            u[i] = u[i] * a + sin(u[i])
        end
    end
    s = 0.0
    for i = 1:n
        s = s + u[i] * u[i]
    end
    out[1] = s
    return nothing
end""", :region),
    # a scratch array that a covering write precedes
    ("ck_scratch", """
function ck_scratch(u, w, a, out, n, nt)
    for t = 1:nt
        for i = 1:n
            w[i] = u[i] * a
        end
        for i = 1:n
            u[i] = sin(w[i]) + u[i]
        end
    end
    s = 0.0
    for i = 1:n
        s = s + u[i] * u[i] + w[i]
    end
    out[1] = s
    return nothing
end""", :region),
    # an array that the body overwrites in part only, a region that starts at 2
    ("ck_partial", """
function ck_partial(u, a, out, n, nt)
    for t = 2:nt
        for i = 2:n - 1
            u[i] = 0.5 * (u[i - 1] + u[i + 1]) * a + 0.1 * u[i] * u[i]
        end
    end
    s = 0.0
    for i = 1:n
        s = s + u[i] * u[i]
    end
    out[1] = s
    return nothing
end""", :region),
    # a carried scalar
    ("ck_scalar", """
function ck_scalar(u, a, p, out, n, nt)
    x = p
    for t = 1:nt
        x = x * a + sin(x)
    end
    s = x
    for i = 1:n
        s = s + u[i] * x
    end
    out[1] = s
    return nothing
end""", :region),
    # an index derived from the step number (a carried integer is refused by the adjoint without checkpoints)
    ("ck_index", """
function ck_index(u, a, out, n, nt)
    for t = 1:nt
        j = mod(3 * t, n) + 1
        u[j] = u[j] * a + sin(u[j])
    end
    s = 0.0
    for i = 1:n
        s = s + u[i] * u[i]
    end
    out[1] = s
    return nothing
end""", :region),
    # an accumulator at a fixed index, and a counter-dependent forcing
    ("ck_accum", """
function ck_accum(u, a, loss, n, nt)
    for t = 1:nt
        for i = 1:n
            u[i] = u[i] * a + sin(0.1 * t) * u[i]
        end
        loss[1] = loss[1] + u[1] * u[1]
    end
    return nothing
end""", :region),
    # a write under a data-dependent branch
    ("ck_branch", """
function ck_branch(u, a, out, n, nt)
    for t = 1:nt
        if u[1] > 0.0
            u[2] = u[2] * a + u[1]
        else
            u[1] = u[1] * a - u[2]
        end
        for i = 1:n
            u[i] = u[i] + 0.1 * sin(u[i])
        end
    end
    s = 0.0
    for i = 1:n
        s = s + u[i]
    end
    out[1] = s
    return nothing
end""", :region),
    # the state is read after the region, by loops that write another array
    ("ck_after", """
function ck_after(u, v, a, out, n, nt)
    for t = 1:nt
        for i = 1:n
            u[i] = u[i] * a + sin(u[i])
        end
    end
    for i = 1:n
        v[i] = u[i] * u[i]
    end
    for i = 1:n
        u[i] = v[i] + u[i]
    end
    s = 0.0
    for i = 1:n
        s = s + u[i] * v[i]
    end
    out[1] = s
    return nothing
end""", :region),
    # a loop inside the step: the region holds a nest
    ("ck_nest", """
function ck_nest(u, a, out, n, nt)
    for t = 1:nt
        for r = 1:3
            for i = 1:n
                u[i] = u[i] * a + 0.1 * sin(u[i]) * r
            end
        end
    end
    s = 0.0
    for i = 1:n
        s = s + u[i]
    end
    out[1] = s
    return nothing
end""", :region),
    # two coupled arrays (a wave step)
    ("ck_wave", """
function ck_wave(u, v, a, out, n, nt)
    for t = 1:nt
        for i = 2:n - 1
            v[i] = v[i] + a * (u[i - 1] - 2.0 * u[i] + u[i + 1])
        end
        for i = 2:n - 1
            u[i] = u[i] + 0.1 * v[i] * v[i]
        end
    end
    s = 0.0
    for i = 1:n
        s = s + u[i] * u[i] + v[i]
    end
    out[1] = s
    return nothing
end""", :region),
    # two regions in sequence with different step counts
    ("ck_two", """
function ck_two(u, a, out, n, nt, nt2)
    for t = 1:nt
        for i = 1:n
            u[i] = u[i] * a + sin(u[i])
        end
    end
    for i = 1:n
        u[i] = u[i] + 1.0
    end
    for t = 1:nt2
        for i = 1:n
            u[i] = u[i] * a - 0.1 * u[i] * u[i]
        end
    end
    s = 0.0
    for i = 1:n
        s = s + u[i]
    end
    out[1] = s
    return nothing
end""", :region),
    # the bound is an expression of the arguments
    ("ck_bound", """
function ck_bound(u, a, out, n, nt)
    for t = 1:nt - 1
        for i = 1:n
            u[i] = u[i] * a + sin(u[i])
        end
    end
    s = 0.0
    for i = 1:n
        s = s + u[i]
    end
    out[1] = s
    return nothing
end""", :region),
    # one array holds both a field that is overwritten and an accumulator
    ("ck_mixed", """
function ck_mixed(u, a, out, n, nt)
    for t = 1:nt
        for i = 2:n
            u[i] = u[i] * a + sin(u[i])
        end
        u[1] = u[1] + a * u[2]
    end
    s = 0.0
    for i = 1:n
        s = s + u[i]
    end
    out[1] = s
    return nothing
end""", :region),
    # a sequential sweep (a scan) is a region too: each step reads what the one before wrote
    ("ck_scan", """
function ck_scan(u, a, out, n, nt)
    for t = 1:nt
        for i = 2:n
            u[i] = u[i - 1] * a + sin(u[i])
        end
    end
    out[1] = u[n]
    return nothing
end""", :region),
    # the loop itself is the scan: it reads the element the step before wrote, at a subscript that holds the loop variable
    ("ck_sweep", """
function ck_sweep(u, a, out, n, nt)
    for i = 2:n
        u[i] = u[i - 1] * a + sin(u[i - 1])
    end
    out[1] = u[n]
    return nothing
end""", :region),
    # test 1: the iterations are independent
    ("ck_no_indep", """
function ck_no_indep(u, a, out, n, nt)
    for i = 1:n
        u[i] = u[i] * a + sin(u[i])
    end
    s = 0.0
    for i = 1:n
        s = s + u[i]
    end
    out[1] = s
    return nothing
end""", "its iterations are independent"),
    # test 2: the loop holds no tape (a linear scalar recurrence)
    ("ck_no_tape", """
function ck_no_tape(u, a, out, n, nt)
    x = a
    for t = 1:nt
        x = 2.0 * x + a
    end
    out[1] = x + u[1]
    return nothing
end""", "holds no tape"),
    # test 4: the bound changes inside the loop
    ("ck_no_bound", """
function ck_no_bound(u, a, out, n, nt)
    m = nt
    for t = 1:m
        m = m - 1
        for i = 1:n
            u[i] = u[i] * a + sin(u[i])
        end
    end
    out[1] = u[1]
    return nothing
end""", "its bounds change"),   # its adjoint cannot be sized with keep_push_pop = false: only the candidate test sees it
    # the literal trip count is too small for a segment to save memory
    ("ck_literal", """
function ck_literal(u, a, out, n, nt)
    for t = 1:5
        for i = 1:n
            u[i] = u[i] * a + sin(u[i])
        end
    end
    out[1] = u[1]
    return nothing
end""", "literal trip count"),
    # a kernel with a solve loop is not supported yet
    ("ck_solve", """
function ck_solve(a, x0, tol, maxit, u, y, n, nt)
    x = x0
    rn = 1.0
    for kk = 1:maxit
        if rn > tol
            xn = a * cos(x)
            rn = (xn - x) * (xn - x)
            x = xn
        end
    end
    for t = 1:nt
        for i = 1:n
            u[i] = u[i] * x + sin(u[i])
        end
    end
    y[1] = x + u[1]
    return nothing
end""", "has a solve loop"),
]
