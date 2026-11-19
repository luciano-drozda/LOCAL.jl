function gat_wide_d(x, xd, w, wd, y, yd, n)
    yd[1] = xd[1]
    y[1] = x[1]
    yd[2] = xd[2]
    y[2] = x[2]
    yd[n - 1] = xd[n - 1]
    y[n - 1] = x[n - 1]
    yd[n] = xd[n]
    y[n] = x[n]
    for i = 3:n - 2
        __cse_0 = x[i - 2]
        __cse_1 = w[i]
        __cse_2 = x[i + 1]
        __cse_3 = x[i - 1]
        __cse_4 = x[i + 2]
        __cse_5 = w[i + 1]
        yd[i] = (((__cse_0 * wd[i] + __cse_1 * xd[i - 2]) + (__cse_2 * xd[i - 1] + __cse_3 * xd[i + 1])) + (__cse_4 * wd[i + 1] + __cse_5 * xd[i + 2])) + -(2.0 * xd[i])
        y[i] = (__cse_1 * __cse_0 + __cse_3 * __cse_2 + __cse_5 * __cse_4) - 2.0 * x[i]
    end
    return nothing
end

function gat_wide(x, w, y, n)
    y[1] = x[1]
    y[2] = x[2]
    y[n - 1] = x[n - 1]
    y[n] = x[n]
    for i = 3:n - 2
        y[i] = (w[i] * x[i - 2] + x[i - 1] * x[i + 1] + w[i + 1] * x[i + 2]) - 2.0 * x[i]
    end
end
