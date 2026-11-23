# fp_two_loops
# Two counted loops with a guard at the top level: agen_adjoint_fixedpoint needs exactly one and refuses this.
# x0: start; a, b: coefficients; tol: tolerance on the squared update; maxit: cap; y: length-1 result
function fp_two_loops(x0, a, b, tol, maxit, y)
    x = x0
    rn = 1.0
    for kk = 1:maxit
        if rn > tol
            xn = a * cos(x)
            rn = (xn - x) * (xn - x)
            x = xn
        end
    end
    rm = 1.0
    for k2 = 1:maxit
        if rm > tol
            xm = b * sin(x) + 0.1
            rm = (xm - x) * (xm - x)
            x = xm
        end
    end
    y[1] = x
    return nothing
end
