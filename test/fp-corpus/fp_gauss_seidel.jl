# fp_gauss_seidel
# Gauss-Seidel sweeps for the tridiagonal system -u_{i-1} + 2.5 u_i - u_{i+1} = f_i (ends fixed at their given values), each sweep in place and in order, until the squared change of a sweep is at most tol, at most maxit sweeps: a counted loop with a guard. Each sweep carries a value from one point to the next. The result is a weighted sum of u, in s[1].
# u: iterate, updated in place; f: right-hand side; wt: weights of the result; tol: tolerance on the squared change; n: interior points; maxit: cap on the sweeps; s: length-1 result
function fp_gauss_seidel(u, f, wt, tol, n, maxit, s)
    rn = 1.0
    for kk = 1:maxit
        if rn > tol
            rn = 0.0
            # in-place sweep: the update of point i reads the new value of point i - 1
            for i = 2:n + 1
                un = (f[i] + u[i - 1] + u[i + 1]) / 2.5
                rn = rn + (un - u[i]) * (un - u[i])
                u[i] = un
            end
        end
    end
    t = 0.0
    for i2 = 1:n + 2
        t = t + wt[i2] * u[i2]
    end
    s[1] = t
    return nothing
end
