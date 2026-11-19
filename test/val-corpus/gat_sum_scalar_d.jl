function gat_sum_scalar_d(x, xd, y, yd, s, sd, n)
    yd[1] = xd[1]
    y[1] = x[1]
    yd[n] = xd[n]
    y[n] = x[n]
    td = 0.0
    t = 0.0
    for i = 2:n - 1
        yd[i] = xd[i - 1] + 2.0 * xd[i + 1]
        y[i] = x[i - 1] + 2.0 * x[i + 1]
        __cse_0 = y[i]
        __cse_1 = __cse_0 * yd[i]
        td = td + (__cse_1 + __cse_1)
        t = t + __cse_0 * __cse_0
    end
    sd[1] = td
    s[1] = t
    return nothing
end

function gat_sum_scalar(x, y, s, n)
    y[1] = x[1]
    y[n] = x[n]
    t = 0.0
    for i = 2:n - 1
        y[i] = x[i - 1] + 2.0 * x[i + 1]
        t = t + y[i] * y[i]
    end
    s[1] = t
end
