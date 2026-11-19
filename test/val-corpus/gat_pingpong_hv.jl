function initstacks_gat_pingpong_b()
    return nothing
end

function gat_pingpong_hv(u, ub, v, vb, n, nsteps, ud, ubd, vd, vbd)
    vd[1] = ud[1]
    v[1] = u[1]
    vd[n] = ud[n]
    v[n] = u[n]
    for t = 1:nsteps
        for i = 2:n - 1
            __cse_0d = ud[i]
            __cse_0 = u[i]
            vd[i] = __cse_0d + 0.25 * ((ud[i - 1] + -(2.0__cse_0d)) + ud[i + 1])
            v[i] = __cse_0 + 0.25 * ((u[i - 1] - 2.0__cse_0) + u[i + 1])
        end
        for i = 2:n - 1
            __cse_1d = vd[i]
            __cse_1 = v[i]
            ud[i] = __cse_1d + 0.25 * ((vd[i - 1] + -(2.0__cse_1d)) + vd[i + 1])
            u[i] = __cse_1 + 0.25 * ((v[i - 1] - 2.0__cse_1) + v[i + 1])
        end
    end
    for t = nsteps:-1:1
        for i = n - 1:-1:2
            __oldb_0d = ubd[i]
            __oldb_0 = ub[i]
            ubd[i] = 0.0
            ub[i] = 0.0
            vbd[i] = vbd[i] + __oldb_0d
            vb[i] = vb[i] + __oldb_0
            __cse_2d = 0.25__oldb_0d
            __cse_2 = 0.25__oldb_0
            vbd[i - 1] = vbd[i - 1] + __cse_2d
            vb[i - 1] = vb[i - 1] + __cse_2
            vbd[i] = vbd[i] + 2.0 * -__cse_2d
            vb[i] = vb[i] + 2.0 * -__cse_2
            vbd[i + 1] = vbd[i + 1] + __cse_2d
            vb[i + 1] = vb[i + 1] + __cse_2
        end
        for i = n - 1:-1:2
            __oldb_0d = vbd[i]
            __oldb_0 = vb[i]
            vbd[i] = 0.0
            vb[i] = 0.0
            ubd[i] = ubd[i] + __oldb_0d
            ub[i] = ub[i] + __oldb_0
            __cse_3d = 0.25__oldb_0d
            __cse_3 = 0.25__oldb_0
            ubd[i - 1] = ubd[i - 1] + __cse_3d
            ub[i - 1] = ub[i - 1] + __cse_3
            ubd[i] = ubd[i] + 2.0 * -__cse_3d
            ub[i] = ub[i] + 2.0 * -__cse_3
            ubd[i + 1] = ubd[i + 1] + __cse_3d
            ub[i + 1] = ub[i + 1] + __cse_3
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
