# fp_counter_used
# A counted loop with a guard whose body reads the loop counter (the iteration is not the same map every time): not a fixed-point loop, refused.
# x0: start; a: coefficient; tol: tolerance on the squared update; maxit: cap; y: length-1 result
function fp_counter_used(x0, a, tol, maxit, y)
    x = x0
    rn = 1.0
    for kk = 1:maxit
        if rn > tol
            xn = a * cos(x) + 0.01 * kk
            rn = (xn - x) * (xn - x)
            x = xn
        end
    end
    y[1] = x
    return nothing
end
