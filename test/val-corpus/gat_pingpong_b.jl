function initstacks_gat_pingpong_b()
    return nothing
end

function gat_pingpong_b(u, ub, v, vb, n, nsteps)
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
    for t = nsteps:-1:1
        for i = n - 1:-1:2
            __oldb_0 = ub[i]
            ub[i] = 0.0
            vb[i] = vb[i] + __oldb_0
            __cse_2 = 0.25__oldb_0
            vb[i - 1] = vb[i - 1] + __cse_2
            vb[i] = vb[i] + 2.0 * -__cse_2
            vb[i + 1] = vb[i + 1] + __cse_2
        end
        for i = n - 1:-1:2
            __oldb_0 = vb[i]
            vb[i] = 0.0
            ub[i] = ub[i] + __oldb_0
            __cse_3 = 0.25__oldb_0
            ub[i - 1] = ub[i - 1] + __cse_3
            ub[i] = ub[i] + 2.0 * -__cse_3
            ub[i + 1] = ub[i + 1] + __cse_3
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

function gat_pingpong(u, v, n, nsteps)
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
end
