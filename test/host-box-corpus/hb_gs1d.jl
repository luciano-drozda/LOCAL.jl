# hb_gs1d
# Gauss-Seidel iteration for -u'' = f on n points with zero boundary values, at most maxit sweeps, each run only while the squared residual is above tol; the final squared residual is an output. The sweep carries a value between iterations and stays on the host, and the adjoint of the scalar input h2 is boxed for the kernels of the reverse sweep.
# u: solution of length n, updated in place
# f: right-hand side of length n
# res: length-1 output, the squared residual after the last sweep
# h2: squared mesh width
# tol: stop when the squared residual is at or below tol
# n: number of points
# maxit: cap on the number of sweeps
function hb_gs1d(u, f, res, h2, tol, n, maxit)
    rs = 1.0
    for kk = 1:maxit
        if rs > tol
            # in-place sweep: u[i-1] was written by the previous iteration
            for i1 = 2:n - 1
                u[i1] = 0.5 * (u[i1 - 1] + u[i1 + 1] + h2 * f[i1])
            end
            # residual of the discrete equation
            rs = 0.0
            for i2 = 2:n - 1
                rs = rs + (f[i2] - (2.0 * u[i2] - u[i2 - 1] - u[i2 + 1]) / h2) * (f[i2] - (2.0 * u[i2] - u[i2 - 1] - u[i2 + 1]) / h2)
            end
        end
    end
    # the squared residual reaches an output, so its chain is useful
    res[1] = rs
    return nothing
end
