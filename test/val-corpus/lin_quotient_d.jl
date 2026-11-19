function lin_quotient_d(u, ud, v, vd, n, nsteps)
    vd[1] = ud[1]
    v[1] = u[1]
    vd[n] = ud[n]
    v[n] = u[n]
    for t = 1:nsteps
        for i = 2:n - 1
            __icse_0 = 2.0 * 2.0
            __cse_1 = ud[i]
            vd[i] = (0.5 * (ud[i - 1] + ud[i + 1]) + (1.0 / __icse_0) * __cse_1) + -(0.5 * (0.5__cse_1))
            __cse_2 = u[i]
            v[i] = ((u[i - 1] + u[i + 1]) / 2.0 + __cse_2 / __icse_0) - (0.5__cse_2) / 2.0
        end
        for i = 2:n - 1
            ud[i] = 0.5 * (vd[i - 1] + vd[i + 1]) + 0.5 * vd[i]
            u[i] = (v[i - 1] + v[i + 1]) / 2.0 + 0.5 * v[i]
        end
    end
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
