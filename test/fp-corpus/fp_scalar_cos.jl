# fp_scalar_cos
# Fixed point x = c * cos(x) + p by plain iteration, one scalar state; the answer is scaled into out[1].
# c: contraction coefficient (below 1 in magnitude); p: offset; x0: starting value; tol: tolerance on the squared update; maxit: cap on the iterations; out: length-1 result
function fp_scalar_cos(c, p, x0, tol, maxit, out)
    x = x0
    rn = 1.0
    for kk = 1:maxit
        if rn > tol
            xn = c * cos(x) + p
            rn = (xn - x) * (xn - x)
            x = xn
        end
    end
    out[1] = 2.0 * x
    return nothing
end
