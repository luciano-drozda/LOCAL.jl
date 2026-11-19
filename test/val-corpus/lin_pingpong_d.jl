function lin_pingpong_d(u, ud, v, vd, n, nsteps)
    vd[1] = ud[1]
    v[1] = u[1]
    vd[n] = ud[n]
    v[n] = u[n]
    for t = 1:nsteps
        for i = 2:n - 1
            __cse_0 = ud[i]
            vd[i] = __cse_0 + 0.25 * ((ud[i - 1] + -(2.0__cse_0)) + ud[i + 1])
            __cse_1 = u[i]
            v[i] = __cse_1 + 0.25 * ((u[i - 1] - 2.0__cse_1) + u[i + 1])
        end
        for i = 2:n - 1
            __cse_2 = vd[i]
            ud[i] = __cse_2 + 0.25 * ((vd[i - 1] + -(2.0__cse_2)) + vd[i + 1])
            __cse_3 = v[i]
            u[i] = __cse_3 + 0.25 * ((v[i - 1] - 2.0__cse_3) + v[i + 1])
        end
    end
    return nothing
end

function lin_pingpong(u, v, n, nsteps)
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
