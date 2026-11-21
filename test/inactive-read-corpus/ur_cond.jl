# ur_cond(x, y, n)
#
# A scalar built from the state steers a branch and reaches no output; the state is overwritten after it is read.
#
# x: state of length n, overwritten at the end
# y: scaled copy of x of length n
# n: length
function ur_cond(x, y, n)
    s = 0.0
    for i = 1:n
        s = s + x[i] * x[i]
    end
    # s only selects the branch
    if s > 1.0
        for i = 1:n
            y[i] = 2.0 * x[i]
        end
    else
        for i = 1:n
            y[i] = 0.5 * x[i]
        end
    end
    for i = 1:n
        x[i] = 0.5 * x[i] + 0.1
    end
    return nothing
end
