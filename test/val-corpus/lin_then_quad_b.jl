function initstacks_lin_then_quad_b()
    return nothing
end

function lin_then_quad_b(u, ub, v, vb, loss, lossb, n, nsteps)
    v[1] = u[1]
    v[n] = u[n]
    for t = 1:nsteps
        for i = 2:n - 1
            __cse_0 = u[i]
            v[i] = __cse_0 + 0.25 * ((u[i - 1] - 2.0__cse_0) + u[i + 1])
        end
        for i = 2:n - 1
            __cse_1 = v[i]
            u[i] = __cse_1 + 0.25 * ((v[i - 1] - 2.0__cse_1) + v[i + 1])
        end
    end
    for i = 1:n
        __cse_2 = u[i]
        loss[1] = loss[1] + 0.5 * __cse_2 * __cse_2
    end
    for i = n:-1:1
        __cse_3 = (0.5 * u[i]) * lossb[1]
        ub[i] = ub[i] + __cse_3
        ub[i] = ub[i] + __cse_3
    end
    for t = nsteps:-1:1
        for i = n - 1:-1:2
            __oldb_0 = ub[i]
            ub[i] = 0.0
            vb[i] = vb[i] + __oldb_0
            __cse_4 = 0.25__oldb_0
            vb[i - 1] = vb[i - 1] + __cse_4
            vb[i] = vb[i] + 2.0 * -__cse_4
            vb[i + 1] = vb[i + 1] + __cse_4
        end
        for i = n - 1:-1:2
            __oldb_0 = vb[i]
            vb[i] = 0.0
            ub[i] = ub[i] + __oldb_0
            __cse_5 = 0.25__oldb_0
            ub[i - 1] = ub[i - 1] + __cse_5
            ub[i] = ub[i] + 2.0 * -__cse_5
            ub[i + 1] = ub[i + 1] + __cse_5
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

function lin_then_quad(u, v, loss, n, nsteps)
    v[1] = u[1]
    v[n] = u[n]
    for t = 1:nsteps
        for i = 2:n - 1
            v[i] = u[i] + 0.25 * ((u[i - 1] - 2.0 * u[i]) + u[i + 1])
        end
        for i = 2:n - 1
            u[i] = v[i] + 0.25 * ((v[i - 1] - 2.0 * v[i]) + v[i + 1])
        end
    end
    for i = 1:n
        loss[1] = loss[1] + 0.5 * u[i] * u[i]
    end
end
