function initstacks_lin_guard_products_b(n, nsteps)
    v_stack = Vector{Float64}(undef, max(0, div(nsteps - 1, 1) + 1) * max(0, div(n - 1, 1) + 1))
    u_stack = Vector{Float64}(undef, max(0, div(nsteps - 1, 1) + 1) * max(0, div(n - 1, 1) + 1))
    return (v_stack, u_stack)
end

function lin_guard_products_b(u, ub, v, vb, w, wb, n, nsteps, v_stack, u_stack)
    for t = 1:nsteps
        for i = 1:n
            __idx_v_stack_0 = ((i - 1) + (t - 1) * (div(n - 1, 1) + 1)) + 1
            v_stack[__idx_v_stack_0] = v[i]
            __cse_0 = u[i]
            v[i] = 0.5 * __cse_0 * __cse_0
        end
        for i = 1:n
            __idx_u_stack_0 = ((i - 1) + (t - 1) * (div(n - 1, 1) + 1)) + 1
            u_stack[__idx_u_stack_0] = u[i]
            u[i] = 1.0 / (1.0 + v[i])
        end
        for i = 1:n
            w[i] = w[i] + 2.0 * sin(0.5 * u[i])
        end
    end
    for t = nsteps:-1:1
        for i = n:-1:1
            ub[i] = ub[i] + 0.5 * (cos(0.5 * u[i]) * (2.0 * wb[i]))
        end
        for i = n:-1:1
            __idx_u_stack_0 = ((i - 1) + (t - 1) * (div(n - 1, 1) + 1)) + 1
            u[i] = u_stack[__idx_u_stack_0]
            __oldb_2 = ub[i]
            ub[i] = 0.0
            vb[i] = vb[i] + -(1.0 / (1.0 + v[i]) ^ 2) * __oldb_2
        end
        for i = n:-1:1
            __idx_v_stack_0 = ((i - 1) + (t - 1) * (div(n - 1, 1) + 1)) + 1
            v[i] = v_stack[__idx_v_stack_0]
            __oldb_2 = vb[i]
            vb[i] = 0.0
            __cse_1 = (0.5 * u[i]) * __oldb_2
            ub[i] = ub[i] + __cse_1
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
