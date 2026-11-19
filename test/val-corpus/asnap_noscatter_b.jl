function initstacks_asnap_noscatter_b(i_njac, i_nnode)
    m_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1))
    x_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1))
    return (m_stack, x_stack)
end

function asnap_noscatter_b(x, xb, w, wb, m, mb, i_nnode, i_njac, m_stack, x_stack)
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            __idx_m_stack_0 = ((i_node - 1) + (i_ - 1) * (div(i_nnode - 1, 1) + 1)) + 1
            m_stack[__idx_m_stack_0] = m[i_node]
            m[i_node] = 0.0
        end
        for i_node = 1:i_nnode
            __icse_0 = div(i_nnode - 1, 1) + 1
            __idx_m_stack_0 = max(0, div(i_njac - 1, 1) + 1) * max(0, __icse_0) + (((i_node - 1) + (i_ - 1) * __icse_0) + 1)
            __cse_1 = m[i_node]
            m_stack[__idx_m_stack_0] = __cse_1
            m[i_node] = __cse_1 + w[i_node] * x[i_node] * 0.1
        end
        for i_node = 1:i_nnode
            __idx_x_stack_0 = ((i_node - 1) + (i_ - 1) * (div(i_nnode - 1, 1) + 1)) + 1
            __cse_2 = x[i_node]
            x_stack[__idx_x_stack_0] = __cse_2
            __cse_3 = m[i_node]
            x[i_node] = __cse_2 + __cse_3 * __cse_3 * 0.5
        end
    end
    for i_ = i_njac:-1:1
        for i_node = i_nnode:-1:1
            __idx_x_stack_0 = ((i_node - 1) + (i_ - 1) * (div(i_nnode - 1, 1) + 1)) + 1
            x[i_node] = x_stack[__idx_x_stack_0]
            __cse_4 = (m[i_node] * 0.5) * xb[i_node]
            mb[i_node] = mb[i_node] + __cse_4
            mb[i_node] = mb[i_node] + __cse_4
        end
        for i_node = i_nnode:-1:1
            __icse_5 = div(i_nnode - 1, 1) + 1
            __idx_m_stack_0 = max(0, div(i_njac - 1, 1) + 1) * max(0, __icse_5) + (((i_node - 1) + (i_ - 1) * __icse_5) + 1)
            m[i_node] = m_stack[__idx_m_stack_0]
            __cse_6 = mb[i_node]
            wb[i_node] = wb[i_node] + (x[i_node] * 0.1) * __cse_6
            xb[i_node] = xb[i_node] + (w[i_node] * 0.1) * __cse_6
        end
        for i_node = i_nnode:-1:1
            __idx_m_stack_0 = ((i_node - 1) + (i_ - 1) * (div(i_nnode - 1, 1) + 1)) + 1
            m[i_node] = m_stack[__idx_m_stack_0]
            mb[i_node] = 0.0
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
