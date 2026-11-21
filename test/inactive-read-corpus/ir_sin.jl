# ir_sin(x, y, w, n, ns)
#
# sin(x) * w with w inactive: the partial of sin needs x.
#
# x: active state of length n, overwritten every sweep
# y: result of length n
# w: inactive scalar
# n: length
# ns: number of sweeps
function ir_sin(x, y, w, n, ns)
    for s = 1:ns
        for i = 1:n
            y[i] = sin(x[i]) * w
        end
        for i = 1:n
            x[i] = 0.5 * x[i] + y[i]
        end
    end
    return nothing
end
