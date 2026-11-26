# heat3d_step(u, un, n)
#
# One explicit step of the 3D heat equation on an n x n x n periodic grid, 7-point stencil.
# The coefficient nu*dt/dx^2 = 0.1 is a literal, so the only differentiated input is the field.
# Written for GPU use: one flattened loop, one thread per cell, gather form (each cell writes
# only itself), neighbors by modular index (no boundary branch), separate output array.
#
# u: field at the old time level, length n^3, x index fastest
# un: field at the new time level, length n^3, filled in place
# n: number of cells per direction
function heat3d_step(u, un, n)
    nn = n * n
    # one loop over all cells; i, j, k recovered with div/mod
    for idx = 1:n * nn
        i = mod(idx - 1, n) + 1
        j = mod(div(idx - 1, n), n) + 1
        k = div(idx - 1, nn) + 1
        # periodic neighbors, no branch
        im = mod(i - 2, n) + 1
        ip = mod(i, n) + 1
        jm = mod(j - 2, n) + 1
        jp = mod(j, n) + 1
        km = mod(k - 2, n) + 1
        kp = mod(k, n) + 1
        c = u[idx]
        un[idx] = c + 0.1 * (u[(k - 1) * nn + (j - 1) * n + im] + u[(k - 1) * nn + (j - 1) * n + ip] + u[(k - 1) * nn + (jm - 1) * n + i] + u[(k - 1) * nn + (jp - 1) * n + i] + u[(km - 1) * nn + (j - 1) * n + i] + u[(kp - 1) * nn + (j - 1) * n + i] - 6.0 * c)
    end
    return nothing
end
