# Generator of kernels for the fixed-point adjoint of a solve loop at any depth: one solve body wrapped in an enclosing nest.
# A solve loop is a counted loop with one guard (`for kk = 1:maxit`, `if rn > tol`). The nests are the shapes the detector can meet; no kernel is written by hand.
#   fps_kernel(body, nest) -> Expr of one function named fps_<body>_<nest>
#   body: :scalar (a scalar contraction, state x), :array (an array contraction in place, state u), :buf (a pointwise contraction into a buffer and a copy back, state w,
#         the shape of a time-stepping solver: the sweeps can read the state as the loop left it) or :mid (the same with a scratch array written, read and written again)
#   nest: see fps_nests()
# Tests build the kernels from here, nothing in STADE.jl reads this file.
isdefined(@__MODULE__, :STADE) || include(joinpath(dirname(@__DIR__), "src", "STADE.jl"))

fps_nests() = [:none, :loop1, :loop2, :ifbranch, :solve_first, :solve_last, :two_in_body, :two_loops, :literal, :cold,
               :indep, :solve_in_solve, :hi_assigned, :tol_assigned, :ragged, :triangle, :two_caps, :tdep_first, :read_before, :read_top, :read_between, :pre_nl, :tdep_nl, :read_nl, :bound_self,
               :cover, :cover_partial, :cover_acc, :cover_post, :cover_postacc, :cover_two, :cover_if, :cover_ifpre, :cover_cond, :cover_loop2_inner, :cover_loop2_outer, :cover_nn]

# the lines of one solve loop, with `@` replaced by the suffix (so two solves in one body use two sets of names)
function fps_solve(body::Symbol, sfx::String)
    body === :buf && return replace("""
rn@ = 1.0
for kk@ = 1:maxit
    if rn@ > tol
        for i@ = 1:n
            wn[i@] = c * tanh(w[i@]) + u[i@]
        end
        rn@ = 0.0
        for i2@ = 1:n
            rn@ = rn@ + (wn[i2@] - w[i2@]) * (wn[i2@] - w[i2@])
        end
        for i3@ = 1:n
            w[i3@] = wn[i3@]
        end
    end
end
""", "@" => sfx)
    body === :mid && return replace("""
rn@ = 1.0
for kk@ = 1:maxit
    if rn@ > tol
        for i0@ = 1:n
            w[i0@] = 0.5 * wo[i0@]
        end
        for i@ = 1:n
            wn[i@] = c * tanh(w[i@]) + u[i@]
        end
        for i3@ = 1:n
            w[i3@] = 2.0 * wn[i3@]
        end
        rn@ = 0.0
        for i2@ = 1:n
            rn@ = rn@ + (w[i2@] - wo[i2@]) * (w[i2@] - wo[i2@])
        end
        for i4@ = 1:n
            wo[i4@] = w[i4@]
        end
    end
end
""", "@" => sfx)
    t = body === :scalar ? """
rn@ = 1.0
for kk@ = 1:maxit
    if rn@ > tol
        xn@ = c * cos(x@) + p
        rn@ = (xn@ - x@) * (xn@ - x@)
        x@ = xn@
    end
end
""" : """
rn@ = 1.0
for kk@ = 1:maxit
    if rn@ > tol
        for i@ = 1:n
            j@ = i@ + 1
            w[j@] = c * tanh(u[j@ - 1] + u[j@ + 1]) + f[j@]
        end
        rn@ = 0.0
        for i2@ = 1:n
            j2@ = i2@ + 1
            rn@ = rn@ + (w[j2@] - u[j2@]) * (w[j2@] - u[j2@])
        end
        for i3@ = 1:n
            j3@ = i3@ + 1
            u[j3@] = w[j3@]
        end
    end
end
"""
    return replace(t, "@" => sfx)
end

fps_ind(txt, k) = join([isempty(l) ? l : "    "^k * l for l in split(chomp(txt), "\n")], "\n") * "\n"

# the statement that moves the state between two solves (a forcing), and the one that reads it after a solve
fps_state(body) = body === :scalar ? :x : body === :mid ? :wo : body === :buf ? :w : :u
fps_pre(body, sfx) = body === :scalar ? "x$sfx = x$sfx + 0.1 * pa[t]\n" : "$(fps_state(body))[3] = $(fps_state(body))[3] + 0.1 * pa[t]\n"
fps_post(body, sfx) = body === :scalar ? "ys[t] = x$sfx * x$sfx\n" : "ys[t] = $(fps_state(body))[2] + $(fps_state(body))[3]\n"
fps_out(body) = body === :scalar ? "out[1] = x\n" : "out[1] = $(fps_state(body))[2] * $(fps_state(body))[3]\n"
fps_start(body) = body === :scalar ? "x = x0\n" : ""

function fps_text(body::Symbol, nest::Symbol)
    S(sfx = "") = fps_solve(body, sfx)
    # a solve whose fixed point changes with t
    T(sfx = "") = (r = replace(replace(replace(fps_solve(body, sfx), "+ p\n" => "+ p + 0.3 * pa[t]\n"), "+ f[j$sfx]\n" => "+ f[j$sfx] + 0.3 * pa[t]\n"), "+ u[i$sfx]\n" => "+ u[i$sfx] + 0.3 * pa[t]\n"); body === :scalar && !isempty(sfx) ? replace(r, "xq" => "x") : r)
    b = ""
    if nest === :none
        b = fps_start(body) * S()
    elseif nest === :loop1
        b = fps_start(body) * "for t = 1:nt\n" * fps_ind(fps_pre(body, "") * S() * fps_post(body, ""), 1) * "end\n"
    elseif nest === :loop2
        b = fps_start(body) * "for s = 1:ns\n    for t = 1:nt\n" * fps_ind(fps_pre(body, "") * S() * fps_post(body, ""), 2) * "    end\nend\n"
    elseif nest === :ifbranch
        b = fps_start(body) * "for t = 1:nt\n" * fps_ind(fps_pre(body, "") * "if pa[t] > 0.5\n" * fps_ind(S(), 1) * "end\n" * fps_post(body, ""), 1) * "end\n"
    elseif nest === :solve_first
        b = fps_start(body) * "for t = 1:nt\n" * fps_ind(S() * fps_post(body, ""), 1) * "end\n"
    elseif nest === :solve_last
        b = fps_start(body) * "for t = 1:nt\n" * fps_ind(fps_pre(body, "") * S(), 1) * "end\n"
    elseif nest === :two_in_body
        b = body === :scalar ? "x = x0\nz = x0\nfor t = 1:nt\n" * fps_ind("x = x + 0.1 * pa[t]\n" * S() * "z = z + 0.5 * x\n" * replace(fps_solve(:scalar, "q"), "xq" => "z") * "ys[t] = x * z\n", 1) * "end\n" : ""
    elseif nest === :two_loops
        b = fps_start(body) * "for t = 1:nt\n" * fps_ind(fps_pre(body, "") * S() * fps_post(body, ""), 1) * "end\nfor t2 = 1:nt\n" *
            fps_ind(replace(fps_pre(body, "") * replace(fps_solve(body, "q"), "xq" => "x") * fps_post(body, ""), "[t]" => "[t2]"), 1) * "end\n"
    elseif nest === :literal
        b = fps_start(body) * "for t = 1:3\n" * fps_ind(fps_pre(body, "") * S() * fps_post(body, ""), 1) * "end\n"
    elseif nest === :cold
        b = body === :scalar ? "for t = 1:nt\n" * fps_ind("x = x0 + 0.1 * pa[t]\n" * S() * fps_post(body, ""), 1) * "end\n" : ""
    elseif nest === :indep
        # every iteration starts from its own state and writes its own result: independent iterations (a GPU stage offloads such a loop)
        b = body === :scalar ? "for t = 1:nt\n" * fps_ind("x = x0 + 0.1 * pa[t]\n" * S() * "ys[t] = x * x\n", 1) * "end\nx = x0\n" : ""
    elseif nest === :tdep_first
        # the fixed point changes with t, and the state is read after each solve: the reverse needs the value each solve left
        b = fps_start(body) * "for t = 1:nt\n" * fps_ind(T() * fps_post(body, ""), 1) * "end\n"
    elseif nest === :read_before
        # the state is read before each solve, the solve then overwrites it: the reverse of the read needs the value the solve found on entry
        b = fps_start(body) * "for t = 1:nt\n" * fps_ind(fps_post(body, "") * T(), 1) * "end\n"
    elseif nest === :pre_nl
        # a self-update that is nonlinear in the state, before the solve: its reverse needs the state it found
        pre = body === :scalar ? "x = x * (1.0 + 0.1 * pa[t])\n" : "u[3] = u[3] * (1.0 + 0.1 * pa[t])\n"
        b = fps_start(body) * "for t = 1:nt\n" * fps_ind(pre * T() * fps_post(body, ""), 1) * "end\n"
    elseif nest === :tdep_nl
        # a read after the solve that is nonlinear in the state; the fixed point changes with t
        post = body === :scalar ? fps_post(body, "") : "ys[t] = u[2] * u[3]\n"
        b = fps_start(body) * "for t = 1:nt\n" * fps_ind(T() * post, 1) * "end\n"
    elseif nest === :read_nl
        # a read before the solve that is nonlinear in the state
        post = body === :scalar ? fps_post(body, "") : "ys[t] = u[2] * u[3]\n"
        b = fps_start(body) * "for t = 1:nt\n" * fps_ind(post * T(), 1) * "end\n"
    elseif nest === :bound_self
        # the bound of the enclosing loop is assigned inside that loop
        b = body === :scalar ? "x = x0\nmm = nt\nfor t = 1:mm\n" * fps_ind(fps_pre(body, "") * S() * fps_post(body, "") * "mm = nt\n", 1) * "end\n" : ""
    elseif nest === :read_top
        b = replace(fps_start(body) * fps_post(body, "") * T(), "[t]" => "[1]")
    elseif nest === :read_between
        # a read between two solves of the same state
        b = fps_start(body) * "for t = 1:nt\n" * fps_ind(T() * fps_post(body, "") * T("q"), 1) * "end\n"
    elseif nest === :two_caps
        # two solve loops with two different caps in one body
        b = fps_start(body) * "for t = 1:nt\n" * fps_ind(fps_pre(body, "") * S() * replace(replace(fps_solve(body, "q"), "xq" => "x"), "1:maxit" => "1:maxq") * fps_post(body, ""), 1) * "end\n"
    elseif nest === :ragged
        # the bound of the inner enclosing loop is assigned inside the outer one
        b = body === :scalar ? "x = x0\nfor t = 1:nt\n    mm = nt - t + 1\n    for s = 1:mm\n" * fps_ind(fps_pre(body, "") * S() * fps_post(body, ""), 2) * "    end\nend\n" : ""
    elseif nest === :triangle
        # the bound of the inner enclosing loop is the counter of the outer one
        b = body === :scalar ? "x = x0\nfor t = 1:nt\n    for s = 1:t\n" * fps_ind(fps_pre(body, "") * S() * fps_post(body, ""), 2) * "    end\nend\n" : ""
    elseif nest === :solve_in_solve
        b = body === :scalar ? "x = x0\nrq = 1.0\nfor kq = 1:maxit\n    if rq > tol\n" * fps_ind(fps_ind(S(), 1) * "rq = rn\n", 1) * "    end\nend\n" : ""
    elseif nest === :hi_assigned
        b = body === :scalar ? "x = x0\nfor t = 1:nt\n    mi = maxit - t\n" * fps_ind(replace(S(), "1:maxit" => "1:mi"), 1) * "    ys[t] = x\nend\n" : ""
    elseif nest === :tol_assigned
        b = body === :scalar ? "x = x0\nfor t = 1:nt\n    tt = tol * pa[t]\n" * fps_ind(replace(S(), "rn > tol" => "rn > tt"), 1) * "    ys[t] = x\nend\n" : ""
    elseif nest in (:cover, :cover_partial, :cover_acc, :cover_post, :cover_postacc, :cover_two, :cover_if, :cover_ifpre, :cover_cond, :cover_loop2_inner, :cover_loop2_outer, :cover_nn)
        # a time loop around a solver of the :buf or :mid shape; the variants differ in what is written to the state before and after the solve
        if body === :buf || body === :mid
            st = String(fps_state(body))
            pre(lo = "1", hi = "n") = "for ip = $lo:$hi\n    $st[ip] = u[ip] + 0.3 * pa[t]\nend\n"
            copyout = "for io = 1:n\n    u[io] = $st[io]\nend\n"
            tail = copyout * "ys[t] = u[2]\n"
            loop(inner) = "for t = 1:nt\n" * fps_ind(inner, 1) * "end\n"
            if nest === :cover
                b = loop(pre() * T() * tail)
            elseif nest === :cover_partial
                b = loop(pre("2") * T() * tail)
            elseif nest === :cover_acc
                b = loop("for ip = 1:n\n    $st[ip] = $st[ip] + u[ip]\nend\n" * T() * tail)
            elseif nest === :cover_post
                b = loop(pre() * T() * "for iq = 1:n\n    $st[iq] = $st[iq] * $st[iq]\nend\n" * tail)
            elseif nest === :cover_postacc
                b = loop(pre() * T() * "for iq = 1:n\n    $st[iq] = $st[iq] + 0.5 * u[iq]\nend\n" * tail)
            elseif nest === :cover_two
                # the second solver has its own fixed point (twice the forcing): it starts from the first one's
                b = loop(pre() * T() * replace(T("q"), "+ u[iq]" => "+ 2.0 * u[iq]") * tail)
            elseif nest === :cover_if
                b = loop(pre() * "if pa[t] > 0.5\n" * fps_ind(T(), 1) * "end\n" * tail)
            elseif nest === :cover_ifpre
                b = loop("if pa[t] > 0.5\n" * fps_ind(pre() * T(), 1) * "end\n" * tail)
            elseif nest === :cover_cond
                b = loop("if pa[t] > 0.5\n" * fps_ind(pre(), 1) * "end\n" * T() * tail)
            elseif nest === :cover_loop2_inner
                b = "for s = 1:ns\n" * fps_ind(loop(pre() * T() * tail), 1) * "end\n"
            elseif nest === :cover_loop2_outer
                b = "for s = 1:ns\n" * fps_ind(replace(pre(), "pa[t]" => "pa[s]") * loop(T() * tail), 1) * "end\n"
            else
                # the bound of every loop of the iteration is a scalar assigned at its top: the bound is the same text for the pre-write and the solver, and not a value fixed before the loop
                b = loop("nn = n\n" * replace(pre(), "1:n" => "1:nn") * replace(T(), "1:n" => "1:nn") * replace(tail, "1:n" => "1:nn"))
            end
        end
    else
        error("fps_text: unknown nest $nest")
    end
    isempty(b) && return nothing
    b *= fps_out(body)
    # the arguments are the names the text reads, in a fixed order
    order = ["x0", "u", "w", "wo", "wn", "f", "c", "p", "pa", "ys", "out", "tol", "n", "nt", "ns", "maxit", "maxq"]
    used = [a for a in order if occursin(Regex("\\b" * a * "\\b"), b)]
    name = "fps_$(body)_$(nest)"
    return "function $name($(join(used, ", ")))\n" * fps_ind(b, 1) * "    return nothing\nend\n"
end

function fps_kernel(body::Symbol, nest::Symbol)
    t = fps_text(body, nest)
    t === nothing && return nothing
    return Meta.parse(t)
end
