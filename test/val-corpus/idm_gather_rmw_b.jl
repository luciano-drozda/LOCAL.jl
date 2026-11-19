function initstacks_idm_gather_rmw_b()
    return nothing
end

function idm_gather_rmw_b(m, mb, x, xb, i_cell_to_node, loss, lossb, i_ncell, i_nnode)
    for i_cell = 1:i_ncell
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            m[i_node] = m[i_node] * 0.5 + x[i_cell]
        end
    end
    for i_n = 1:i_nnode
        __cse_0 = m[i_n]
        loss[1] = loss[1] + __cse_0 * __cse_0
    end
    for i_n = i_nnode:-1:1
        __cse_1 = m[i_n] * lossb[1]
        mb[i_n] = mb[i_n] + __cse_1
        mb[i_n] = mb[i_n] + __cse_1
    end
    for i_cell = i_ncell:-1:1
        for i_loc = 4:-1:1
            i_node = i_cell_to_node[i_loc, i_cell]
            __oldb_0 = mb[i_node]
            mb[i_node] = 0.0
            mb[i_node] = mb[i_node] + 0.5__oldb_0
            xb[i_cell] = xb[i_cell] + __oldb_0
        end
    end
    return nothing
end

function idm_gather_rmw(m, x, i_cell_to_node, loss, i_ncell, i_nnode)
    for i_cell = 1:i_ncell
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            m[i_node] = m[i_node] * 0.5 + x[i_cell]
        end
    end
    for i_n = 1:i_nnode
        loss[1] = loss[1] + m[i_n] * m[i_n]
    end
end
