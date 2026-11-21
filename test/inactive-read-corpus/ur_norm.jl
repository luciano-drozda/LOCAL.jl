# ur_norm(x, r, n, ns)
#
# A guarded sweep loop whose squared residual norm only steers the guard and reaches no output.
#
# x: state of length n, updated every sweep
# r: residual of length n, rewritten every sweep
# n: length
# ns: maximum number of sweeps
function ur_norm(x, r, n, ns)
    rn = 1.0
    for s = 1:ns
        # run the sweep only while the squared norm is above the tolerance
        if rn > 1.0e-20
            for i = 1:n
                r[i] = 0.5 * x[i] + 0.1
            end
            for i = 1:n
                x[i] = 0.9 * x[i] + 0.2 * r[i]
            end
            # squared norm of r: steers the guard only
            rn = 0.0
            for i = 1:n
                rn = rn + r[i] * r[i]
            end
        end
    end
    return nothing
end
