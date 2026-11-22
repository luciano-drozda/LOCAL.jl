# fp_mixed_state
# A damped fixed-point iteration with an array state u and a scalar state m (a running average of the entries of u): u_i <- (1 - a) u_i + a (f_i + 0.3 sin(m)), m <- mean of the new u; until the squared change of u is at most tol, at most maxit steps: a counted loop with a guard. The result is m, in s[1].
# u: iterate (updated in place); f: forcing; a: relaxation (between 0 and 1); tol: tolerance on the squared change; n: entries; maxit: cap on the steps; s: length-1 result
function fp_mixed_state(u, f, a, tol, n, maxit, s)
    m = 0.0
    rn = 1.0
    for kk = 1:maxit
        if rn > tol
            rn = 0.0
            tot = 0.0
            # update every entry from the old mean
            for i = 1:n
                un = (1.0 - a) * u[i] + a * (f[i] + 0.3 * sin(m))
                rn = rn + (un - u[i]) * (un - u[i])
                u[i] = un
                tot = tot + un
            end
            m = tot / n
        end
    end
    s[1] = m
    return nothing
end
