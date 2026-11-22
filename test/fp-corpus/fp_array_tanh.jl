# fp_array_tanh
# Fixed point of u_i = c * tanh(u_{i-1} + u_{i+1}) + f_i on the interior of a line (ends fixed at their given values), by plain iteration into a second array and a copy back, until the squared update is at most tol, at most maxit sweeps: a counted loop with a guard. The result is the sum of squares of u, in s[1].
# u: iterate, updated in place (its first and last entries are fixed data); w: second array for the update; f: forcing; c: coupling (below 1/2 in magnitude); tol: tolerance on the squared update; n: number of interior points; maxit: cap on the sweeps; s: length-1 result
function fp_array_tanh(u, w, f, c, tol, n, maxit, s)
    rn = 1.0
    for kk = 1:maxit
        if rn > tol
            # update every interior point from the current iterate
            for i = 1:n
                j = i + 1
                w[j] = c * tanh(u[j - 1] + u[j + 1]) + f[j]
            end
            # squared size of the update
            rn = 0.0
            for i2 = 1:n
                j2 = i2 + 1
                rn = rn + (w[j2] - u[j2]) * (w[j2] - u[j2])
            end
            # copy back
            for i3 = 1:n
                j3 = i3 + 1
                u[j3] = w[j3]
            end
        end
    end
    # a scalar of the converged field
    t = 0.0
    for i4 = 1:n + 2
        t = t + u[i4] * u[i4]
    end
    s[1] = t
    return nothing
end
