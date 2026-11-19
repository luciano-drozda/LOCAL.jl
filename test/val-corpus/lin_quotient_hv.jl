function initstacks_lin_quotient_b()
    return nothing
end

function lin_quotient_hv(u, ub, v, vb, n, nsteps, ud, ubd, vd, vbd)
    vd[1] = ud[1]
    v[1] = u[1]
    vd[n] = ud[n]
    v[n] = u[n]
    for t = 1:nsteps
        for i = 2:n - 1
            __cse_0d = ud[i]
            __cse_0 = u[i]
            __ihcse_0 = 2.0 * 2.0
            vd[i] = (0.5 * (ud[i - 1] + ud[i + 1]) + (1.0 / __ihcse_0) * __cse_0d) + -(0.5 * (0.5__cse_0d))
            v[i] = ((u[i - 1] + u[i + 1]) / 2.0 + __cse_0 / __ihcse_0) - (0.5__cse_0) / 2.0
        end
        for i = 2:n - 1
            ud[i] = 0.5 * (vd[i - 1] + vd[i + 1]) + 0.5 * vd[i]
            u[i] = (v[i - 1] + v[i + 1]) / 2.0 + 0.5 * v[i]
        end
    end
    for t = nsteps:-1:1
        for i = n - 1:-1:2
            __oldb_0d = ubd[i]
            __oldb_0 = ub[i]
            ubd[i] = 0.0
            ub[i] = 0.0
            __cse_1d = 0.5__oldb_0d
            __cse_1 = 0.5__oldb_0
            vbd[i - 1] = vbd[i - 1] + __cse_1d
            vb[i - 1] = vb[i - 1] + __cse_1
            vbd[i + 1] = vbd[i + 1] + __cse_1d
            vb[i + 1] = vb[i + 1] + __cse_1
            vbd[i] = vbd[i] + __cse_1d
            vb[i] = vb[i] + __cse_1
        end
        for i = n - 1:-1:2
            __oldb_0d = vbd[i]
            __oldb_0 = vb[i]
            vbd[i] = 0.0
            vb[i] = 0.0
            __cse_2d = 0.5__oldb_0d
            __cse_2 = 0.5__oldb_0
            ubd[i - 1] = ubd[i - 1] + __cse_2d
            ub[i - 1] = ub[i - 1] + __cse_2
            ubd[i + 1] = ubd[i + 1] + __cse_2d
            ub[i + 1] = ub[i + 1] + __cse_2
            __ihcse_1 = 1.0 / (2.0 * 2.0)
            ubd[i] = ubd[i] + __ihcse_1 * __oldb_0d
            ub[i] = ub[i] + __ihcse_1 * __oldb_0
            ubd[i] = ubd[i] + 0.5 * (0.5 * -__oldb_0d)
            ub[i] = ub[i] + 0.5 * (0.5 * -__oldb_0)
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

function lin_quotient(u, v, n, nsteps)
    v[1] = u[1]
    v[n] = u[n]
    for t = 1:nsteps
        for i = 2:n - 1
            v[i] = ((u[i - 1] + u[i + 1]) / 2.0 + u[i] / (2.0 * 2.0)) - (0.5 * u[i]) / 2.0
        end
        for i = 2:n - 1
            u[i] = (v[i - 1] + v[i + 1]) / 2.0 + 0.5 * v[i]
        end
    end
end
