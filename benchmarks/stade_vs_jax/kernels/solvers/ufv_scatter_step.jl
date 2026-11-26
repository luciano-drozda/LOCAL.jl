# ufv_scatter_step(phi, vel, e1, e2, rhs, phin, nc, ne)
#
# One explicit step of advection-diffusion on an unstructured mesh, written the usual way:
# loop over faces, compute one flux per face, add it to the residual of both cells it touches.
# Two faces can share a cell, so the residual update is a scatter-accumulate, and the adjoint of
# the flux gather is again a scatter. This is the reference form for ufv_gather_step.
# Face flux f = vel * (phi1 + phi2) / 2 + (phi1 - phi2) / 2, leaving cell e1 and entering cell e2.
# dt/dx = 0.1 is a literal.
#
# phi: cell values at the old time level, length nc
# vel: normal velocity of each face, positive from e1 to e2, length ne
# e1: first cell of each face (1-based), length ne
# e2: second cell of each face (1-based), length ne
# rhs: scratch residual, length nc, overwritten
# phin: cell values at the new time level, length nc, filled in place
# nc: number of cells
# ne: number of faces
function ufv_scatter_step(phi, vel, e1, e2, rhs, phin, nc, ne)
    # clear the residual
    for c = 1:nc
        rhs[c] = 0.0
    end
    # one flux per face, added to both cells (scatter-accumulate)
    for e = 1:ne
        a = e1[e]
        b = e2[e]
        f = vel[e] * 0.5 * (phi[a] + phi[b]) + 0.5 * (phi[a] - phi[b])
        rhs[a] = rhs[a] + f
        rhs[b] = rhs[b] - f
    end
    # explicit update
    for c = 1:nc
        phin[c] = phi[c] - 0.1 * rhs[c]
    end
    return nothing
end
