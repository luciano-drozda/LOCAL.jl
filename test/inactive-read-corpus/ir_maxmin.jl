# ir_maxmin(x, y, c, n, ns)
#
# max(x, c) with c inactive: the selection compares x with c, so x keeps its snapshot.
#
# x: active state of length n, overwritten every sweep
# y: result of length n
# c: inactive scalar threshold
# n: length
# ns: number of sweeps
function ir_maxmin(x, y, c, n, ns)
    for s = 1:ns
        for i = 1:n
            y[i] = max(x[i], c)
        end
        for i = 1:n
            x[i] = 0.5 * x[i] + 0.5 * y[i] - 0.1
        end
    end
    return nothing
end
