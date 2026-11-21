# ur_dep(x, y, z, n, ns)
#
# The array z is a nonlinear function of the state, and is not a dependent when only y is.
#
# x: state of length n, overwritten every sweep
# y: accumulated output of length n
# z: square of x, length n; an output nobody asks for when only y is a dependent
# n: length
# ns: number of sweeps
function ur_dep(x, y, z, n, ns)
    for s = 1:ns
        for i = 1:n
            z[i] = x[i] * x[i]
        end
        for i = 1:n
            y[i] = y[i] + 0.5 * x[i]
        end
        for i = 1:n
            x[i] = 0.5 * x[i] + 0.1
        end
    end
    return nothing
end
