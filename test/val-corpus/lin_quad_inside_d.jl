function lin_quad_inside_d(u, ud, v, vd, loss, lossd, n, nsteps)
    vd[1] = ud[1]
    v[1] = u[1]
    vd[n] = ud[n]
    v[n] = u[n]
    for t = 1:nsteps
        for i = 1:n
            __cse_0 = u[i]
            __cse_1 = (0.5__cse_0) * ud[i]
            lossd[1] = lossd[1] + (__cse_1 + __cse_1)
            loss[1] = loss[1] + 0.5 * __cse_0 * __cse_0
        end
        for i = 2:n - 1
            __cse_2 = ud[i]
            vd[i] = __cse_2 + 0.25 * ((ud[i - 1] + -(2.0__cse_2)) + ud[i + 1])
            __cse_3 = u[i]
            v[i] = __cse_3 + 0.25 * ((u[i - 1] - 2.0__cse_3) + u[i + 1])
        end
        for i = 2:n - 1
            __cse_4 = vd[i]
            ud[i] = __cse_4 + 0.25 * ((vd[i - 1] + -(2.0__cse_4)) + vd[i + 1])
            __cse_5 = v[i]
            u[i] = __cse_5 + 0.25 * ((v[i - 1] - 2.0__cse_5) + v[i + 1])
        end
    end
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
