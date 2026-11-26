# wave2d_step(um, u, c2, up, n)
#
# One leapfrog step of the 2D wave equation with a variable coefficient on an n x n periodic grid:
# up = 2 u - um + c2 * (5-point Laplacian of u). c2 holds (c dt / dx)^2 per cell, so the
# coefficient field is a differentiated parameter, as in an inverse problem.
#
# um: field at time level t-1, length n^2, x index fastest
# u: field at time level t, length n^2
# c2: coefficient field, length n^2, values in (0, 0.5) for stability
# up: field at time level t+1, length n^2, filled in place
# n: number of cells per direction
function wave2d_step(um, u, c2, up, n)
    # one loop over all cells; i, j recovered with div/mod
    for idx = 1:n * n
        i = mod(idx - 1, n) + 1
        j = div(idx - 1, n) + 1
        # periodic neighbors, no branch
        im = mod(i - 2, n) + 1
        ip = mod(i, n) + 1
        jm = mod(j - 2, n) + 1
        jp = mod(j, n) + 1
        c = u[idx]
        lap = u[(j - 1) * n + im] + u[(j - 1) * n + ip] + u[(jm - 1) * n + i] + u[(jp - 1) * n + i] - 4.0 * c
        up[idx] = 2.0 * c - um[idx] + c2[idx] * lap
    end
    return nothing
end
