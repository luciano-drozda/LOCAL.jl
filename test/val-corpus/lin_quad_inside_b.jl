function initstacks_lin_quad_inside_b(n, nsteps)
    u_stack = Vector{Float64}(undef, max(0, div(nsteps - 1, 1) + 1) * max(0, div((n - 1) - 2, 1) + 1))
    return u_stack
end

function lin_quad_inside_b(u, ub, v, vb, loss, lossb, n, nsteps, u_stack)
    v[1] = u[1]
    v[n] = u[n]
    for t = 1:nsteps
        for i = 1:n
            __cse_0 = u[i]
            loss[1] = loss[1] + 0.5 * __cse_0 * __cse_0
        end
        for i = 2:n - 1
            __cse_1 = u[i]
            v[i] = __cse_1 + 0.25 * ((u[i - 1] - 2.0__cse_1) + u[i + 1])
        end
        for i = 2:n - 1
            __idx_u_stack_0 = ((i - 2) + (t - 1) * (div((n - 1) - 2, 1) + 1)) + 1
            u_stack[__idx_u_stack_0] = u[i]
            __cse_2 = v[i]
            u[i] = __cse_2 + 0.25 * ((v[i - 1] - 2.0__cse_2) + v[i + 1])
        end
    end
    for t = nsteps:-1:1
        for i = n - 1:-1:2
            __idx_u_stack_0 = ((i - 2) + (t - 1) * (div((n - 1) - 2, 1) + 1)) + 1
            u[i] = u_stack[__idx_u_stack_0]
            __oldb_2 = ub[i]
            ub[i] = 0.0
            vb[i] = vb[i] + __oldb_2
            __cse_3 = 0.25__oldb_2
            vb[i - 1] = vb[i - 1] + __cse_3
            vb[i] = vb[i] + 2.0 * -__cse_3
            vb[i + 1] = vb[i + 1] + __cse_3
        end
        for i = n - 1:-1:2
            __oldb_0 = vb[i]
            vb[i] = 0.0
            ub[i] = ub[i] + __oldb_0
            __cse_4 = 0.25__oldb_0
            ub[i - 1] = ub[i - 1] + __cse_4
            ub[i] = ub[i] + 2.0 * -__cse_4
            ub[i + 1] = ub[i + 1] + __cse_4
        end
        for i = n:-1:1
            __cse_5 = (0.5 * u[i]) * lossb[1]
            ub[i] = ub[i] + __cse_5
            ub[i] = ub[i] + __cse_5
        end
    end
    __oldb_0 = vb[n]
    vb[n] = 0.0
    ub[n] = ub[n] + __oldb_0
    __oldb_0 = vb[1]
    vb[1] = 0.0
    ub[1] = ub[1] + __oldb_0
    return nothing
end

function lin_quad_inside(u, v, loss, n, nsteps)
    v[1] = u[1]
    v[n] = u[n]
    for t = 1:nsteps
        for i = 1:n
            loss[1] = loss[1] + 0.5 * u[i] * u[i]
        end
        for i = 2:n - 1
            v[i] = u[i] + 0.25 * ((u[i - 1] - 2.0 * u[i]) + u[i + 1])
        end
        for i = 2:n - 1
            u[i] = v[i] + 0.25 * ((v[i - 1] - 2.0 * v[i]) + v[i + 1])
        end
    end
end
