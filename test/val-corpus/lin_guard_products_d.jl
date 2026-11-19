function lin_guard_products_d(u, ud, v, vd, w, wd, n, nsteps)
    for t = 1:nsteps
        for i = 1:n
            __cse_0 = u[i]
            __cse_1 = (0.5__cse_0) * ud[i]
            vd[i] = __cse_1 + __cse_1
            v[i] = 0.5 * __cse_0 * __cse_0
        end
        for i = 1:n
            __cse_2 = 1.0 + v[i]
            ud[i] = -(1.0 / __cse_2 ^ 2) * vd[i]
            u[i] = 1.0 / __cse_2
        end
        for i = 1:n
            __cse_3 = 0.5 * u[i]
            wd[i] = wd[i] + 2.0 * (cos(__cse_3) * (0.5 * ud[i]))
            w[i] = w[i] + 2.0 * sin(__cse_3)
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
