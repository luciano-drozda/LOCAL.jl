# ir_quot(x, y, c, d, n, ns)
#
# Quotient by a scalar: the partial of the numerator needs d only, the partial of d needs x.
#
# x: state of length n, overwritten every sweep
# y: scaled copy of x of length n
# c: source of length n
# d: divisor; inactive when only x and c are independent
# n: length
# ns: number of sweeps
function ir_quot(x, y, c, d, n, ns)
    for s = 1:ns
        for i = 1:n
            y[i] = x[i] / d
        end
        # x is overwritten from y
        for i = 1:n
            x[i] = 0.5 * y[i] + c[i]
        end
    end
    return nothing
end
