function initstacks_lin_guard_products_b(n, nsteps)
    v_stack = Vector{Float64}(undef, max(0, div(nsteps - 1, 1) + 1) * max(0, div(n - 1, 1) + 1))
    u_stack = Vector{Float64}(undef, max(0, div(nsteps - 1, 1) + 1) * max(0, div(n - 1, 1) + 1))
    return (v_stack, u_stack)
end

function lin_guard_products_hv(u, ub, v, vb, w, wb, n, nsteps, ud, ubd, vd, vbd, wd, wbd, v_stack, u_stack)
    v_stack_d = Vector{Float64}(undef, length(v_stack))
    u_stack_d = Vector{Float64}(undef, length(u_stack))
    for t = 1:nsteps
        for i = 1:n
            __idx_v_stack_0 = ((i - 1) + (t - 1) * (div(n - 1, 1) + 1)) + 1
            v_stack_d[__idx_v_stack_0] = vd[i]
            v_stack[__idx_v_stack_0] = v[i]
            __cse_0d = ud[i]
            __cse_0 = u[i]
            __hcse_0 = (0.5__cse_0) * __cse_0d
            vd[i] = __hcse_0 + __hcse_0
            v[i] = 0.5 * __cse_0 * __cse_0
        end
        for i = 1:n
            __idx_u_stack_0 = ((i - 1) + (t - 1) * (div(n - 1, 1) + 1)) + 1
            u_stack_d[__idx_u_stack_0] = ud[i]
            u_stack[__idx_u_stack_0] = u[i]
            __hcse_1 = 1.0 + v[i]
            ud[i] = -(1.0 / __hcse_1 ^ 2) * vd[i]
            u[i] = 1.0 / __hcse_1
        end
        for i = 1:n
            __hcse_2 = 0.5 * u[i]
            wd[i] = wd[i] + 2.0 * (cos(__hcse_2) * (0.5 * ud[i]))
            w[i] = w[i] + 2.0 * sin(__hcse_2)
        end
    end
    for t = nsteps:-1:1
        for i = n:-1:1
            __hcse_3 = 2.0 * wb[i]
            __hcse_4 = 0.5 * u[i]
            __hcse_5 = cos(__hcse_4)
            ubd[i] = ubd[i] + 0.5 * (__hcse_3 * (-(sin(__hcse_4)) * (0.5 * ud[i])) + __hcse_5 * (2.0 * wbd[i]))
            ub[i] = ub[i] + 0.5 * (__hcse_5 * __hcse_3)
        end
        for i = n:-1:1
            __idx_u_stack_0 = ((i - 1) + (t - 1) * (div(n - 1, 1) + 1)) + 1
            ud[i] = u_stack_d[__idx_u_stack_0]
            u[i] = u_stack[__idx_u_stack_0]
            __oldb_2d = ubd[i]
            __oldb_2 = ub[i]
            ubd[i] = 0.0
            ub[i] = 0.0
            __hcse_6 = 1.0 + v[i]
            __hcse_7 = __hcse_6 ^ 2
            __hcse_8 = -(1.0 / __hcse_7)
            vbd[i] = vbd[i] + (__oldb_2 * -(-(1.0 / __hcse_7 ^ 2) * ((2__hcse_6) * vd[i])) + __hcse_8 * __oldb_2d)
            vb[i] = vb[i] + __hcse_8 * __oldb_2
        end
        for i = n:-1:1
            __idx_v_stack_0 = ((i - 1) + (t - 1) * (div(n - 1, 1) + 1)) + 1
            vd[i] = v_stack_d[__idx_v_stack_0]
            v[i] = v_stack[__idx_v_stack_0]
            __oldb_2d = vbd[i]
            __oldb_2 = vb[i]
            vbd[i] = 0.0
            vb[i] = 0.0
            __hcse_9 = 0.5 * u[i]
            __cse_1d = __oldb_2 * (0.5 * ud[i]) + __hcse_9 * __oldb_2d
            __cse_1 = __hcse_9 * __oldb_2
            ubd[i] = ubd[i] + __cse_1d
            ub[i] = ub[i] + __cse_1
            ubd[i] = ubd[i] + __cse_1d
            ub[i] = ub[i] + __cse_1
        end
    end
    return nothing
end

function lin_guard_products(u, v, w, n, nsteps)
    for t = 1:nsteps
        for i = 1:n
            v[i] = 0.5 * u[i] * u[i]
        end
        for i = 1:n
            u[i] = 1.0 / (1.0 + v[i])
        end
        for i = 1:n
            w[i] = w[i] + 2.0 * sin(0.5 * u[i])
        end
    end
end
