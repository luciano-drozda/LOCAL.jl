# uc_recompute(x, w, y, n)
#
# A branch on a scalar inside a loop whose backward pass re-executes the forward half; the scalar is overwritten after the loop.
#
# x: input of length n
# w: scratch of length n, written in the loop and read after it
# y: output of length n
# n: length
function uc_recompute(x, w, y, n)
    # c only selects the branch inside the loop
    c = 0.5 * x[1]
    for i = 1:n
        t = x[i] * x[i]
        if c > 0.1
            w[i] = t * t
        else
            w[i] = t
        end
    end
    for i = 1:n
        y[i] = 2.0 * w[i]
    end
    # c is overwritten and used again after the loop
    c = x[2] * 0.25
    y[1] = y[1] + c
    return nothing
end
