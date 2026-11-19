function asnap_noscatter_d(x, xd, w, wd, m, md, i_nnode, i_njac)
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            md[i_node] = 0.0
            m[i_node] = 0.0
        end
        for i_node = 1:i_nnode
            __cse_0 = x[i_node]
            __cse_1 = w[i_node]
            md[i_node] = md[i_node] + ((__cse_0 * 0.1) * wd[i_node] + (__cse_1 * 0.1) * xd[i_node])
            m[i_node] = m[i_node] + __cse_1 * __cse_0 * 0.1
        end
        for i_node = 1:i_nnode
            __cse_2 = m[i_node]
            __cse_3 = (__cse_2 * 0.5) * md[i_node]
            xd[i_node] = xd[i_node] + (__cse_3 + __cse_3)
            x[i_node] = x[i_node] + __cse_2 * __cse_2 * 0.5
        end
    end
    return nothing
end

function asnap_noscatter(x, w, m, i_nnode, i_njac)
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            m[i_node] = 0.0
        end
        for i_node = 1:i_nnode
            m[i_node] = m[i_node] + w[i_node] * x[i_node] * 0.1
        end
        for i_node = 1:i_nnode
            x[i_node] = x[i_node] + m[i_node] * m[i_node] * 0.5
        end
    end
end
