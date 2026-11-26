# ufv_gather_step(phi, velc, nbr, phin, nc, deg)
#
# The same update as ufv_scatter_step in gather form: one loop over cells, each cell visits its
# deg neighbors, recomputes the flux through each face and writes only itself. Every face flux is
# computed twice, once from each side, with opposite sign. In exchange the loop has no
# scatter-accumulate write, and its adjoint is a gather over the same neighbor table plus a
# reduction into the cell's own value.
# Flux out of cell c through the face to cell j: velc * (phi_c + phi_j) / 2 + (phi_c - phi_j) / 2.
# dt/dx = 0.1 is a literal.
#
# phi: cell values at the old time level, length nc
# velc: velocity out of the cell through each of its faces, length nc * deg, cell-major
# nbr: neighbor cell of each face (1-based), length nc * deg, cell-major
# phin: cell values at the new time level, length nc, filled in place
# nc: number of cells
# deg: faces per cell (the same for every cell)
function ufv_gather_step(phi, velc, nbr, phin, nc, deg)
    for c = 1:nc
        pc = phi[c]
        acc = 0.0
        # running sum over the faces of this cell
        for k = 1:deg
            j = nbr[(c - 1) * deg + k]
            acc += velc[(c - 1) * deg + k] * 0.5 * (pc + phi[j]) + 0.5 * (pc - phi[j])
        end
        phin[c] = pc - 0.1 * acc
    end
    return nothing
end
