# ir_local(x, y, n, ns)
#
# A local scalar built from constants only is overwritten every sweep and multiplies an active array.
#
# x: active state of length n
# y: product of length n
# n: length
# ns: number of sweeps
function ir_local(x, y, n, ns)
    t = 1.0
    for s = 1:ns
        for i = 1:n
            y[i] = t * x[i]
        end
        for i = 1:n
            x[i] = 0.5 * y[i] + 0.2
        end
        t = 0.5 * t + 0.1
    end
    return nothing
end
