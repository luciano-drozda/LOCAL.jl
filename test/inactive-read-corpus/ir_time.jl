# ir_time(x, y, t, dt, n, ns)
#
# A time loop: the inactive time t advances every step and enters a forcing sin(t) that multiplies an active array.
#
# x: active state of length n, overwritten every step
# y: forced copy of x of length n
# t: time, inactive when only x is independent
# dt: time step, inactive
# n: length
# ns: number of steps
function ir_time(x, y, t, dt, n, ns)
    for s = 1:ns
        for i = 1:n
            y[i] = sin(t) * x[i]
        end
        for i = 1:n
            x[i] = 0.5 * y[i] + 0.1
        end
        t = t + dt
    end
    return nothing
end
