# Generator of kernels for the fixed-point adjoint of a solve loop at any depth: one solve body wrapped in an enclosing nest.
# A solve loop is a counted loop with one guard (`for kk = 1:maxit`, `if rn > tol`). The nests are the shapes the detector can meet; no kernel is written by hand.
#   fps_kernel(body, nest) -> Expr of one function named fps_<body>_<nest>
#   body: :scalar (a scalar contraction, state x) or :array (an array contraction in place, state u)
#   nest: see fps_nests()
# Tests build the kernels from here, nothing in STADE.jl reads this file.
isdefined(@__MODULE__, :STADE) || include(joinpath(dirname(@__DIR__), "src", "STADE.jl"))

fps_nests() = [:none, :loop1, :loop2, :ifbranch, :solve_first, :solve_last, :two_in_body, :two_loops, :literal, :cold,
               :indep, :solve_in_solve, :hi_assigned, :tol_assigned, :ragged, :triangle, :two_caps]

# the lines of one solve loop, with `@` replaced by the suffix (so two solves in one body use two sets of names)
function fps_solve(body::Symbol, sfx::String)
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
fps_pre(body, sfx) = body === :scalar ? "x$sfx = x$sfx + 0.1 * pa[t]\n" : "u[3] = u[3] + 0.1 * pa[t]\n"
fps_post(body, sfx) = body === :scalar ? "ys[t] = x$sfx * x$sfx\n" : "ys[t] = u[2] + u[3]\n"
fps_out(body) = body === :scalar ? "out[1] = x\n" : "out[1] = u[2] * u[3]\n"
fps_start(body) = body === :scalar ? "x = x0\n" : ""

function fps_text(body::Symbol, nest::Symbol)
    S(sfx = "") = fps_solve(body, sfx)
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
    elseif nest === :two_caps
        # two solve loops with two different caps in one body
        b = fps_start(body) * "for t = 1:nt\n" * fps_ind(fps_pre(body, "") * S() * fps_post(body, "") * replace(replace(fps_solve(body, "q"), "xq" => "x"), "1:maxit" => "1:maxq"), 1) * "end\n"
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
    else
        error("fps_text: unknown nest $nest")
    end
    isempty(b) && return nothing
    b *= fps_out(body)
    # the arguments are the names the text reads, in a fixed order
    order = ["x0", "u", "w", "f", "c", "p", "pa", "ys", "out", "tol", "n", "nt", "ns", "maxit", "maxq"]
    used = [a for a in order if occursin(Regex("\\b" * a * "\\b"), b)]
    name = "fps_$(body)_$(nest)"
    return "function $name($(join(used, ", ")))\n" * fps_ind(b, 1) * "    return nothing\nend\n"
end

function fps_kernel(body::Symbol, nest::Symbol)
    t = fps_text(body, nest)
    t === nothing && return nothing
    return Meta.parse(t)
end
