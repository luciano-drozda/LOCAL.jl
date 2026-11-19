function gat_nonlinear_d(x, xd, y, yd, n)
    yd[1] = xd[1]
    y[1] = x[1]
    yd[n] = xd[n]
    y[n] = x[n]
    for i = 2:n - 1
        __cse_0 = x[i + 1]
        __cse_1 = x[i - 1]
        __cse_2 = sin(__cse_1)
        __cse_3 = exp(0.1 * x[i])
        yd[i] = (__cse_0 * (cos(__cse_1) * xd[i - 1]) + __cse_2 * xd[i + 1]) + __cse_3 * (0.1 * xd[i])
        y[i] = __cse_2 * __cse_0 + __cse_3
    end
    return nothing
end

function gat_nonlinear(x, y, n)
    y[1] = x[1]
    y[n] = x[n]
    for i = 2:n - 1
        y[i] = sin(x[i - 1]) * x[i + 1] + exp(0.1 * x[i])
    end
end
