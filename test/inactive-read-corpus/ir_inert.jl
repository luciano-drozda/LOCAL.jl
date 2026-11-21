# ir_inert(x, y, z, w, n, ns)
#
# An inactive scalar overwritten every sweep that only feeds an inactive array: nothing of it is read by a partial, so nothing is taped.
#
# x: active state of length n
# y: scaled copy of x of length n
# z: array built from w alone, inactive
# w: inactive scalar, overwritten every sweep
# n: length
# ns: number of sweeps
function ir_inert(x, y, z, w, n, ns)
    for s = 1:ns
        for i = 1:n
            z[i] = w * w
        end
        for i = 1:n
            y[i] = 2.0 * x[i]
        end
        w = 0.9 * w + 0.1
    end
    return nothing
end
