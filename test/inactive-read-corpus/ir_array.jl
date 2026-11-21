# ir_array(x, m, y, n, ns)
#
# An inactive coefficient ARRAY is overwritten every sweep and multiplies an active array.
#
# x: active state of length n, overwritten every sweep
# m: coefficient of length n, inactive when only x is independent; overwritten every sweep
# y: product of length n
# n: length
# ns: number of sweeps
function ir_array(x, m, y, n, ns)
    for s = 1:ns
        for i = 1:n
            y[i] = m[i] * x[i]
        end
        for i = 1:n
            x[i] = 0.5 * y[i] + 0.2
        end
        for i = 1:n
            m[i] = 0.5 * m[i] + 0.1
        end
    end
    return nothing
end
