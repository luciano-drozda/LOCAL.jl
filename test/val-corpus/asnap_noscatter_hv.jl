function initstacks_asnap_noscatter_b(i_njac, i_nnode)
    m_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1))
    x_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1))
    return (m_stack, x_stack)
end

function asnap_noscatter_hv(x, xb, w, wb, m, mb, i_nnode, i_njac, xd, xbd, wd, wbd, md, mbd, m_stack, x_stack)
    m_stack_d = Vector{Float64}(undef, length(m_stack))
    x_stack_d = Vector{Float64}(undef, length(x_stack))
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            __idx_m_stack_0 = ((i_node - 1) + (i_ - 1) * (div(i_nnode - 1, 1) + 1)) + 1
            m_stack_d[__idx_m_stack_0] = md[i_node]
            m_stack[__idx_m_stack_0] = m[i_node]
            md[i_node] = 0.0
            m[i_node] = 0.0
        end
        for i_node = 1:i_nnode
            __icse_0 = div(i_nnode - 1, 1) + 1
            __idx_m_stack_0 = max(0, div(i_njac - 1, 1) + 1) * max(0, __icse_0) + (((i_node - 1) + (i_ - 1) * __icse_0) + 1)
            __cse_1d = md[i_node]
            __cse_1 = m[i_node]
            m_stack_d[__idx_m_stack_0] = __cse_1d
            m_stack[__idx_m_stack_0] = __cse_1
            __hcse_0 = x[i_node]
            __hcse_1 = w[i_node]
            md[i_node] = __cse_1d + ((__hcse_0 * 0.1) * wd[i_node] + (__hcse_1 * 0.1) * xd[i_node])
            m[i_node] = __cse_1 + __hcse_1 * __hcse_0 * 0.1
        end
        for i_node = 1:i_nnode
            __idx_x_stack_0 = ((i_node - 1) + (i_ - 1) * (div(i_nnode - 1, 1) + 1)) + 1
            __cse_2d = xd[i_node]
            __cse_2 = x[i_node]
            x_stack_d[__idx_x_stack_0] = __cse_2d
            x_stack[__idx_x_stack_0] = __cse_2
            __cse_3d = md[i_node]
            __cse_3 = m[i_node]
            __hcse_2 = (__cse_3 * 0.5) * __cse_3d
            xd[i_node] = __cse_2d + (__hcse_2 + __hcse_2)
            x[i_node] = __cse_2 + __cse_3 * __cse_3 * 0.5
        end
    end
    for i_ = i_njac:-1:1
        for i_node = i_nnode:-1:1
            __idx_x_stack_0 = ((i_node - 1) + (i_ - 1) * (div(i_nnode - 1, 1) + 1)) + 1
            xd[i_node] = x_stack_d[__idx_x_stack_0]
            x[i_node] = x_stack[__idx_x_stack_0]
            __hcse_3 = xb[i_node]
            __hcse_4 = m[i_node] * 0.5
            __cse_4d = __hcse_3 * (0.5 * md[i_node]) + __hcse_4 * xbd[i_node]
            __cse_4 = __hcse_4 * __hcse_3
            mbd[i_node] = mbd[i_node] + __cse_4d
            mb[i_node] = mb[i_node] + __cse_4
            mbd[i_node] = mbd[i_node] + __cse_4d
            mb[i_node] = mb[i_node] + __cse_4
        end
        for i_node = i_nnode:-1:1
            __icse_5 = div(i_nnode - 1, 1) + 1
            __idx_m_stack_0 = max(0, div(i_njac - 1, 1) + 1) * max(0, __icse_5) + (((i_node - 1) + (i_ - 1) * __icse_5) + 1)
            md[i_node] = m_stack_d[__idx_m_stack_0]
            m[i_node] = m_stack[__idx_m_stack_0]
            __cse_6d = mbd[i_node]
            __cse_6 = mb[i_node]
            __hcse_5 = x[i_node] * 0.1
            wbd[i_node] = wbd[i_node] + (__cse_6 * (0.1 * xd[i_node]) + __hcse_5 * __cse_6d)
            wb[i_node] = wb[i_node] + __hcse_5 * __cse_6
            __hcse_6 = w[i_node] * 0.1
            xbd[i_node] = xbd[i_node] + (__cse_6 * (0.1 * wd[i_node]) + __hcse_6 * __cse_6d)
            xb[i_node] = xb[i_node] + __hcse_6 * __cse_6
        end
        for i_node = i_nnode:-1:1
            __idx_m_stack_0 = ((i_node - 1) + (i_ - 1) * (div(i_nnode - 1, 1) + 1)) + 1
            md[i_node] = m_stack_d[__idx_m_stack_0]
            m[i_node] = m_stack[__idx_m_stack_0]
            mbd[i_node] = 0.0
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
