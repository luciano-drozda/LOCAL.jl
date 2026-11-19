function gat_two_targets_d(a, ad, x, xd, y, yd, z, zd, n)
    yd[1] = xd[1]
    y[1] = x[1]
    yd[n] = xd[n]
    y[n] = x[n]
    zd[1] = ad[1]
    z[1] = a[1]
    zd[n] = ad[n]
    z[n] = a[n]
    for i = 2:n - 1
        __cse_0 = x[i]
        __cse_1 = a[i - 1]
        yd[i] = __cse_0 * ad[i - 1] + __cse_1 * xd[i]
        y[i] = __cse_1 * __cse_0
        __cse_2 = a[i]
        __cse_3 = x[i + 1]
        zd[i] = (__cse_2 * xd[i + 1] + __cse_3 * ad[i]) + yd[i]
        z[i] = __cse_3 * __cse_2 + y[i]
    end
    return nothing
end

function gat_two_targets(a, x, y, z, n)
    y[1] = x[1]
    y[n] = x[n]
    z[1] = a[1]
    z[n] = a[n]
    for i = 2:n - 1
        y[i] = a[i - 1] * x[i]
        z[i] = x[i + 1] * a[i] + y[i]
    end
end
