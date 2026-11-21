# ur_arm(x, y, n)
#
# Control: a scalar assigned in both arms of a branch reaches the output through a product with the state.
#
# x: state of length n, overwritten at the end
# y: output of length n
# n: length
function ur_arm(x, y, n)
    if x[1] > 0.5
        q = x[1] * x[1]
    else
        q = 3.0 * x[1] * x[1]
    end
    for i = 1:n
        y[i] = q * x[i]
    end
    for i = 1:n
        x[i] = 0.5 * x[i] + 0.1
    end
    return nothing
end
