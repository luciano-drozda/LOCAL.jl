# heat1d_multi(u, v, n, nsteps)
#
# 1D diffusion with the time loop inside the kernel: 2 * nsteps explicit three-point steps, ping-pong between u and v.
# Both arrays are overwritten, so the adjoint stores snapshots, and its stack grows with nsteps.
# The time loop carries the field from one pass to the next, so it stays sequential on the host. Each inner loop is one launch.
# This is the shape that tests the stack layout when the outermost loop is sequential and the thread is an inner loop.
#
# u: field, length n, input and final state (after 2 * nsteps steps)
# v: scratch field, length n, overwritten
# n: number of points
# nsteps: number of (u -> v, v -> u) pairs of steps
function heat1d_multi(u, v, n, nsteps)
    # boundary values are fixed, in both buffers
    v[1] = u[1]
    v[n] = u[n]
    # the pass loop carries the field from one pass to the next
    for t = 1:nsteps
        # u -> v, interior points
        for i = 2:n - 1
            v[i] = u[i] + 0.25 * (u[i - 1] - 2.0 * u[i] + u[i + 1])
        end
        # v -> u, interior points
        for i = 2:n - 1
            u[i] = v[i] + 0.25 * (v[i - 1] - 2.0 * v[i] + v[i + 1])
        end
    end
    return nothing
end
