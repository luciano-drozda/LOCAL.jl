# ur_live(x, y, n)
#
# Control: a nonlinear accumulation that reaches the output through a scalar chain must keep its snapshot.
#
# x: state of length n, overwritten at the end
# y: y[1] receives the accumulated square sum, length n
# n: length
function ur_live(x, y, n)
    acc = 0.0
    for i = 1:n
        acc = acc + x[i] * x[i]
    end
    b = acc * acc
    y[1] = b
    for i = 1:n
        x[i] = 0.5 * x[i] + 0.1
    end
    return nothing
end
