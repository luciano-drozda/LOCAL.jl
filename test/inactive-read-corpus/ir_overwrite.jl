# ir_overwrite(x, y, w, n, ns)
#
# The inactive factor w is itself overwritten between sweeps and multiplies an active array.
#
# x: active state of length n, overwritten every sweep
# y: product of length n
# w: scalar updated every sweep; inactive when only x is independent
# n: length
# ns: number of sweeps
function ir_overwrite(x, y, w, n, ns)
    for s = 1:ns
        for i = 1:n
            y[i] = w * x[i]
        end
        for i = 1:n
            x[i] = 0.5 * y[i] + 0.2
        end
        w = 0.9 * w + 0.1
    end
    return nothing
end
