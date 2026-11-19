function initstacks_lin_quad_inside_b(n, nsteps)
    u_stack = Vector{Float64}(undef, max(0, div(nsteps - 1, 1) + 1) * max(0, div((n - 1) - 2, 1) + 1))
    return u_stack
end

function lin_quad_inside_hv(u, ub, v, vb, loss, lossb, n, nsteps, ud, ubd, vd, vbd, lossd, lossbd, u_stack)
    u_stack_d = Vector{Float64}(undef, length(u_stack))
    vd[1] = ud[1]
    v[1] = u[1]
    vd[n] = ud[n]
    v[n] = u[n]
    for t = 1:nsteps
        for i = 1:n
            __cse_0d = ud[i]
            __cse_0 = u[i]
            __hcse_0 = (0.5__cse_0) * __cse_0d
            lossd[1] = lossd[1] + (__hcse_0 + __hcse_0)
            loss[1] = loss[1] + 0.5 * __cse_0 * __cse_0
        end
        for i = 2:n - 1
            __cse_1d = ud[i]
            __cse_1 = u[i]
            vd[i] = __cse_1d + 0.25 * ((ud[i - 1] + -(2.0__cse_1d)) + ud[i + 1])
            v[i] = __cse_1 + 0.25 * ((u[i - 1] - 2.0__cse_1) + u[i + 1])
        end
        for i = 2:n - 1
            __idx_u_stack_0 = ((i - 2) + (t - 1) * (div((n - 1) - 2, 1) + 1)) + 1
            u_stack_d[__idx_u_stack_0] = ud[i]
            u_stack[__idx_u_stack_0] = u[i]
            __cse_2d = vd[i]
            __cse_2 = v[i]
            ud[i] = __cse_2d + 0.25 * ((vd[i - 1] + -(2.0__cse_2d)) + vd[i + 1])
            u[i] = __cse_2 + 0.25 * ((v[i - 1] - 2.0__cse_2) + v[i + 1])
        end
    end
    for t = nsteps:-1:1
        for i = n - 1:-1:2
            __idx_u_stack_0 = ((i - 2) + (t - 1) * (div((n - 1) - 2, 1) + 1)) + 1
            ud[i] = u_stack_d[__idx_u_stack_0]
            u[i] = u_stack[__idx_u_stack_0]
            __oldb_2d = ubd[i]
            __oldb_2 = ub[i]
            ubd[i] = 0.0
            ub[i] = 0.0
            vbd[i] = vbd[i] + __oldb_2d
            vb[i] = vb[i] + __oldb_2
            __cse_3d = 0.25__oldb_2d
            __cse_3 = 0.25__oldb_2
            vbd[i - 1] = vbd[i - 1] + __cse_3d
            vb[i - 1] = vb[i - 1] + __cse_3
            vbd[i] = vbd[i] + 2.0 * -__cse_3d
            vb[i] = vb[i] + 2.0 * -__cse_3
            vbd[i + 1] = vbd[i + 1] + __cse_3d
            vb[i + 1] = vb[i + 1] + __cse_3
        end
        for i = n - 1:-1:2
            __oldb_0d = vbd[i]
            __oldb_0 = vb[i]
            vbd[i] = 0.0
            vb[i] = 0.0
            ubd[i] = ubd[i] + __oldb_0d
            ub[i] = ub[i] + __oldb_0
            __cse_4d = 0.25__oldb_0d
            __cse_4 = 0.25__oldb_0
            ubd[i - 1] = ubd[i - 1] + __cse_4d
            ub[i - 1] = ub[i - 1] + __cse_4
            ubd[i] = ubd[i] + 2.0 * -__cse_4d
            ub[i] = ub[i] + 2.0 * -__cse_4
            ubd[i + 1] = ubd[i + 1] + __cse_4d
            ub[i + 1] = ub[i + 1] + __cse_4
        end
        for i = n:-1:1
            __hcse_1 = lossb[1]
            __hcse_2 = 0.5 * u[i]
            __cse_5d = __hcse_1 * (0.5 * ud[i]) + __hcse_2 * lossbd[1]
            __cse_5 = __hcse_2 * __hcse_1
            ubd[i] = ubd[i] + __cse_5d
            ub[i] = ub[i] + __cse_5
            ubd[i] = ubd[i] + __cse_5d
            ub[i] = ub[i] + __cse_5
        end
    end
    __oldb_0d = vbd[n]
    __oldb_0 = vb[n]
    vbd[n] = 0.0
    vb[n] = 0.0
    ubd[n] = ubd[n] + __oldb_0d
    ub[n] = ub[n] + __oldb_0
    __oldb_0d = vbd[1]
    __oldb_0 = vb[1]
    vbd[1] = 0.0
    vb[1] = 0.0
    ubd[1] = ubd[1] + __oldb_0d
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
