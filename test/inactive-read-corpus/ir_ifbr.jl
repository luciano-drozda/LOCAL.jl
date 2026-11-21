# ir_ifbr(x, y, w, n, ns)
#
# An inactive scalar overwritten every sweep and read in one arm of a data-dependent branch on an active array.
#
# x: active state of length n, overwritten every sweep
# y: result of length n
# w: inactive scalar, overwritten every sweep
# n: length
# ns: number of sweeps
function ir_ifbr(x, y, w, n, ns)
    for s = 1:ns
        for i = 1:n
            if x[i] > 0.0
                y[i] = w * x[i]
            else
                y[i] = 0.5 * x[i]
            end
        end
        for i = 1:n
            x[i] = 0.5 * x[i] + 0.5 * y[i] - 0.1
        end
        w = 0.9 * w + 0.1
    end
    return nothing
end
