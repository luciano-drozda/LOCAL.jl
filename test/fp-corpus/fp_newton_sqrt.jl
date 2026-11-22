# fp_newton_sqrt
# Newton iteration for the square root of a, x <- (x + a / x) / 2, until the squared update is at most tol, at most maxit steps: a counted loop with a guard. The square root goes to y[1].
# a: radicand (positive); x0: starting value (positive); tol: tolerance on the squared update; maxit: cap on the steps; y: length-1 result
function fp_newton_sqrt(a, x0, tol, maxit, y)
    x = x0
    rn = 1.0
    for kk = 1:maxit
        if rn > tol
            xn = 0.5 * (x + a / x)
            rn = (xn - x) * (xn - x)
            x = xn
        end
    end
    y[1] = x
    return nothing
end
