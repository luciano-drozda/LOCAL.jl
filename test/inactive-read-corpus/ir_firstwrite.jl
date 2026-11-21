# ir_firstwrite(x, y, n, ns)
#
# A local scalar is first assigned inside the sweep loop, from the sweep counter only, and multiplies an active array.
#
# x: active state of length n, overwritten every sweep
# y: product of length n
# n: length
# ns: number of sweeps
function ir_firstwrite(x, y, n, ns)
    for s = 1:ns
        # factor built from the counter: inactive, and not defined before the first sweep
        t = 0.5 * s
        for i = 1:n
            y[i] = t * x[i]
        end
        for i = 1:n
            x[i] = 0.5 * y[i] + 0.2
        end
    end
    return nothing
end
