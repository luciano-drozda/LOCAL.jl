function gat_reverse_d(x, xd, y, yd, n)
    yd[1] = xd[1]
    y[1] = x[1]
    yd[n] = xd[n]
    y[n] = x[n]
    for i = n - 1:-1:2
        __cse_0 = x[i + 1]
        __cse_1 = x[i - 1]
        yd[i] = (__cse_0 * xd[i - 1] + __cse_1 * xd[i + 1]) + xd[i]
        y[i] = __cse_1 * __cse_0 + x[i]
    end
    return nothing
end

function gat_reverse(x, y, n)
    y[1] = x[1]
    y[n] = x[n]
    for i = n - 1:-1:2
        y[i] = x[i - 1] * x[i + 1] + x[i]
    end
end
