# fp_slow_linear
# Damped iteration x <- x + omega * (a - x), until the squared update is at most tol, at most maxit steps: a counted loop with a guard. The contraction is slow, so with omega near 0 and a loose tol the forward loop converges while the reverse sweeps (as many as the loop has iterations) cannot settle. The result goes to y[1].
# a: target; x0: starting value; omega: damping, between 0 and 1; tol: tolerance on the squared update; maxit: cap on the steps; y: length-1 result
function fp_slow_linear(a, x0, omega, tol, maxit, y)
    x = x0
    rn = 1.0
    for kk = 1:maxit
        if rn > tol
            xn = x + omega * (a - x)
            rn = (xn - x) * (xn - x)
            x = xn
        end
    end
    y[1] = x
    return nothing
end
