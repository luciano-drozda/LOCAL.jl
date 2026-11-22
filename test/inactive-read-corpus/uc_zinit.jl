# uc_zinit(s, w, out, tol, n, maxit)
#
# A bisection whose sign test g is read only by the branch condition: g has no snapshot site, but the sweep over k still saves it at its block boundary.
# The loop over k may run zero times, so g needs an initial value before that save.
#
# s: parameter of length n
# w: initial upper bracket of length n
# out: result of length n
# tol: width below which the bracket stops moving
# n: length
# maxit: number of bisection steps
function uc_zinit(s, w, out, tol, n, maxit)
    for i = 1:n
        lo = 0.0
        hi = w[i]
        for k = 1:maxit
            mid = 0.5 * (lo + hi)
            g = mid * mid - s[i] * (1.0 - mid)
            # the bracket moves only while it is wider than tol
            if hi - lo > tol
                if g > 0.0
                    hi = mid
                else
                    lo = mid
                end
            end
        end
        out[i] = 0.5 * (lo + hi)
    end
    return nothing
end
