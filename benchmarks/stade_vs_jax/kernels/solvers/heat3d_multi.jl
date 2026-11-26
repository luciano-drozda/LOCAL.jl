# heat3d_multi(u, v, n, npairs)
#
# Reference version for the time loop inside the kernel: 2 * npairs steps of the same stencil as
# heat3d_step, ping-pong between u and v, both arrays overwritten in place. The adjoint has to
# store one snapshot of each overwritten array per pass, so its stack grows with npairs.
# This is the baseline that shows what the one-step kernel with a host-driven time loop avoids.
#
# u: field, length n^3, input and final state (after 2 * npairs steps)
# v: scratch field, length n^3, overwritten
# n: number of cells per direction
# npairs: number of (u -> v, v -> u) pairs of steps
function heat3d_multi(u, v, n, npairs)
    nn = n * n
    # the pair loop carries the field from one pass to the next
    for t = 1:npairs
        for idx = 1:n * nn
            i = mod(idx - 1, n) + 1
            j = mod(div(idx - 1, n), n) + 1
            k = div(idx - 1, nn) + 1
            im = mod(i - 2, n) + 1
            ip = mod(i, n) + 1
            jm = mod(j - 2, n) + 1
            jp = mod(j, n) + 1
            km = mod(k - 2, n) + 1
            kp = mod(k, n) + 1
            c = u[idx]
            v[idx] = c + 0.1 * (u[(k - 1) * nn + (j - 1) * n + im] + u[(k - 1) * nn + (j - 1) * n + ip] + u[(k - 1) * nn + (jm - 1) * n + i] + u[(k - 1) * nn + (jp - 1) * n + i] + u[(km - 1) * nn + (j - 1) * n + i] + u[(kp - 1) * nn + (j - 1) * n + i] - 6.0 * c)
        end
        for idx = 1:n * nn
            i = mod(idx - 1, n) + 1
            j = mod(div(idx - 1, n), n) + 1
            k = div(idx - 1, nn) + 1
            im = mod(i - 2, n) + 1
            ip = mod(i, n) + 1
            jm = mod(j - 2, n) + 1
            jp = mod(j, n) + 1
            km = mod(k - 2, n) + 1
            kp = mod(k, n) + 1
            c = v[idx]
            u[idx] = c + 0.1 * (v[(k - 1) * nn + (j - 1) * n + im] + v[(k - 1) * nn + (j - 1) * n + ip] + v[(k - 1) * nn + (jm - 1) * n + i] + v[(k - 1) * nn + (jp - 1) * n + i] + v[(km - 1) * nn + (j - 1) * n + i] + v[(kp - 1) * nn + (j - 1) * n + i] - 6.0 * c)
        end
    end
    return nothing
end
