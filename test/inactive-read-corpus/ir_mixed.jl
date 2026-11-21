# ir_mixed(x, z, a, b, w, n, ns)
#
# One array read by two products: one with an inactive scalar, one with an active array. The second needs the value of x.
#
# x: active state of length n, overwritten every sweep
# z: active array of length n
# a: x scaled by w, length n
# b: product of x and z, length n
# w: inactive scalar
# n: length
# ns: number of sweeps
function ir_mixed(x, z, a, b, w, n, ns)
    for s = 1:ns
        for i = 1:n
            a[i] = x[i] * w
            b[i] = x[i] * z[i]
        end
        for i = 1:n
            x[i] = 0.5 * a[i] + 0.25 * b[i]
        end
    end
    return nothing
end
