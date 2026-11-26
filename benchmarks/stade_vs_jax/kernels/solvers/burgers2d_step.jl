# burgers2d_step(u, v, un, vn, n)
#
# One explicit step of the 2D viscous Burgers equation (two velocity components) on an n x n
# periodic grid: first-order upwind advection plus 5-point diffusion. The upwind choice is
# written with max/min, so the loop has no branch. dt/dx = 0.1 and nu*dt/dx^2 = 0.05 are literals.
#
# u: x velocity at the old time level, length n^2, x index fastest
# v: y velocity at the old time level, length n^2
# un: x velocity at the new time level, length n^2, filled in place
# vn: y velocity at the new time level, length n^2, filled in place
# n: number of cells per direction
function burgers2d_step(u, v, un, vn, n)
    for idx = 1:n * n
        i = mod(idx - 1, n) + 1
        j = div(idx - 1, n) + 1
        im = mod(i - 2, n) + 1
        ip = mod(i, n) + 1
        jm = mod(j - 2, n) + 1
        jp = mod(j, n) + 1
        iw = (j - 1) * n + im
        ie = (j - 1) * n + ip
        is = (jm - 1) * n + i
        inn = (jp - 1) * n + i
        uc = u[idx]
        vc = v[idx]
        # upwind weights: the positive part of the speed looks west/south, the negative part east/north
        upos = max(uc, 0.0)
        uneg = min(uc, 0.0)
        vpos = max(vc, 0.0)
        vneg = min(vc, 0.0)
        un[idx] = uc - 0.1 * (upos * (uc - u[iw]) + uneg * (u[ie] - uc) + vpos * (uc - u[is]) + vneg * (u[inn] - uc)) + 0.05 * (u[iw] + u[ie] + u[is] + u[inn] - 4.0 * uc)
        vn[idx] = vc - 0.1 * (upos * (vc - v[iw]) + uneg * (v[ie] - vc) + vpos * (vc - v[is]) + vneg * (v[inn] - vc)) + 0.05 * (v[iw] + v[ie] + v[is] + v[inn] - 4.0 * vc)
    end
    return nothing
end
