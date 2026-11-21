# ur_via(x, w, y, n)
#
# Control: a scratch array that is not a requested dependent still reaches y, so it stays active and x keeps its snapshot.
#
# x: state of length n, overwritten at the end
# w: scratch of length n holding the square of x
# y: output of length n built from w
# n: length
function ur_via(x, w, y, n)
    for i = 1:n
        w[i] = x[i] * x[i]
    end
    for i = 1:n
        y[i] = 2.0 * w[i] + 1.0
    end
    for i = 1:n
        x[i] = 0.5 * x[i] + 0.1
    end
    return nothing
end
