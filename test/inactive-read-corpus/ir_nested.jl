# ir_nested(x, z, y, w, n, ns)
#
# Nested product: the inactive factor w sits inside the first operand of the outer product, whose other operand z is active.
# The partial of z needs the VALUE of w * x, so x must keep its snapshot even though w is inactive.
#
# x: active state of length n, overwritten every sweep
# z: active state of length n, overwritten every sweep
# y: product of length n
# w: inactive scalar
# n: length
# ns: number of sweeps
function ir_nested(x, z, y, w, n, ns)
    for s = 1:ns
        for i = 1:n
            y[i] = (w * x[i]) * z[i]
        end
        for i = 1:n
            x[i] = 0.5 * y[i] + 0.1
            z[i] = 0.5 * z[i] + 0.1 * y[i]
        end
    end
    return nothing
end
