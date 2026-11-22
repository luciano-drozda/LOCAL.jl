# fp_accumulate
# Not a fixed-point iteration: a counted loop with a guard whose body only adds to the state (u grows by a fixed amount each step until the guard closes). Its reverse never consumes a state adjoint, so stade_adjoint_fixedpoint refuses it.
# u: running state (updated in place); f: increments; tol: guard tolerance; n: entries; maxit: cap; s: length-1 result
function fp_accumulate(u, f, tol, n, maxit, s)
    rn = 1.0
    for kk = 1:maxit
        if rn > tol
            rn = 0.0
            for i = 1:n
                u[i] = u[i] + f[i]
                rn = rn + 0.5 * f[i] * f[i]
            end
        end
    end
    t = 0.0
    for i2 = 1:n
        t = t + u[i2]
    end
    s[1] = t
    return nothing
end
