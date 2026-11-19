function initstacks_idm_gather_rmw_b()
    return nothing
end

function idm_gather_rmw_hv(m, mb, x, xb, i_cell_to_node, loss, lossb, i_ncell, i_nnode, md, mbd, xd, xbd, lossd, lossbd)
    for i_cell = 1:i_ncell
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            md[i_node] = 0.5 * md[i_node] + xd[i_cell]
            m[i_node] = m[i_node] * 0.5 + x[i_cell]
        end
    end
    for i_n = 1:i_nnode
        __cse_0d = md[i_n]
        __cse_0 = m[i_n]
        __hcse_0 = __cse_0 * __cse_0d
        lossd[1] = lossd[1] + (__hcse_0 + __hcse_0)
        loss[1] = loss[1] + __cse_0 * __cse_0
    end
    for i_n = i_nnode:-1:1
        __hcse_1 = lossb[1]
        __hcse_2 = m[i_n]
        __cse_1d = __hcse_1 * md[i_n] + __hcse_2 * lossbd[1]
        __cse_1 = __hcse_2 * __hcse_1
        mbd[i_n] = mbd[i_n] + __cse_1d
        mb[i_n] = mb[i_n] + __cse_1
        mbd[i_n] = mbd[i_n] + __cse_1d
        mb[i_n] = mb[i_n] + __cse_1
    end
    for i_cell = i_ncell:-1:1
        for i_loc = 4:-1:1
            i_node = i_cell_to_node[i_loc, i_cell]
            __oldb_0d = mbd[i_node]
            __oldb_0 = mb[i_node]
            mbd[i_node] = 0.0
            mb[i_node] = 0.0
            mbd[i_node] = mbd[i_node] + 0.5__oldb_0d
            mb[i_node] = mb[i_node] + 0.5__oldb_0
            xbd[i_cell] = xbd[i_cell] + __oldb_0d
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
