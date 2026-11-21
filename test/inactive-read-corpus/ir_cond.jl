# ir_cond(x, y, w, n, ns)
#
# A data-dependent branch reads x: a condition always counts as a value read.
#
# x: active state of length n, overwritten every sweep
# y: result of length n
# w: inactive scalar
# n: length
# ns: number of sweeps
function ir_cond(x, y, w, n, ns)
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
    end
    return nothing
end
