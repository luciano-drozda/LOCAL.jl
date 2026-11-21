# ir_prod(u, f, r, w, n, ns)
#
# Damped Jacobi-like sweeps: the nonlinear product w * r is the only reader of r.
#
# u: iterate of length n, updated in place
# f: right-hand side of length n
# r: residual of length n, rewritten every sweep
# w: damping factor; inactive when only u and f are independent
# n: length
# ns: number of sweeps
function ir_prod(u, f, r, w, n, ns)
    for s = 1:ns
        # residual: linear in u and f
        for i = 1:n
            r[i] = f[i] - 2.0 * u[i]
        end
        # update: w * r is nonlinear, but the partial of w is the only one that needs r
        for i = 1:n
            u[i] = u[i] + w * r[i]
        end
    end
    return nothing
end
