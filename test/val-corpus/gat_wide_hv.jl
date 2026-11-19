function initstacks_gat_wide_b()
    return nothing
end

function gat_wide_hv(x, xb, w, wb, y, yb, n, xd, xbd, wd, wbd, yd, ybd)
    yd[1] = xd[1]
    y[1] = x[1]
    yd[2] = xd[2]
    y[2] = x[2]
    yd[n - 1] = xd[n - 1]
    y[n - 1] = x[n - 1]
    yd[n] = xd[n]
    y[n] = x[n]
    for i = 3:n - 2
        __hcse_0 = x[i - 2]
        __hcse_1 = w[i]
        __hcse_2 = x[i + 1]
        __hcse_3 = x[i - 1]
        __hcse_4 = x[i + 2]
        __hcse_5 = w[i + 1]
        yd[i] = (((__hcse_0 * wd[i] + __hcse_1 * xd[i - 2]) + (__hcse_2 * xd[i - 1] + __hcse_3 * xd[i + 1])) + (__hcse_4 * wd[i + 1] + __hcse_5 * xd[i + 2])) + -(2.0 * xd[i])
        y[i] = (__hcse_1 * __hcse_0 + __hcse_3 * __hcse_2 + __hcse_5 * __hcse_4) - 2.0 * x[i]
    end
    for i = n - 2:-1:3
        __oldb_0d = ybd[i]
        __oldb_0 = yb[i]
        ybd[i] = 0.0
        yb[i] = 0.0
        __hcse_6 = x[i - 2]
        wbd[i] = wbd[i] + (__oldb_0 * xd[i - 2] + __hcse_6 * __oldb_0d)
        wb[i] = wb[i] + __hcse_6 * __oldb_0
        __hcse_7 = w[i]
        xbd[i - 2] = xbd[i - 2] + (__oldb_0 * wd[i] + __hcse_7 * __oldb_0d)
        xb[i - 2] = xb[i - 2] + __hcse_7 * __oldb_0
        __hcse_8 = x[i + 1]
        xbd[i - 1] = xbd[i - 1] + (__oldb_0 * xd[i + 1] + __hcse_8 * __oldb_0d)
        xb[i - 1] = xb[i - 1] + __hcse_8 * __oldb_0
        __hcse_9 = x[i - 1]
        xbd[i + 1] = xbd[i + 1] + (__oldb_0 * xd[i - 1] + __hcse_9 * __oldb_0d)
        xb[i + 1] = xb[i + 1] + __hcse_9 * __oldb_0
        __hcse_10 = x[i + 2]
        wbd[i + 1] = wbd[i + 1] + (__oldb_0 * xd[i + 2] + __hcse_10 * __oldb_0d)
        wb[i + 1] = wb[i + 1] + __hcse_10 * __oldb_0
        __hcse_11 = w[i + 1]
        xbd[i + 2] = xbd[i + 2] + (__oldb_0 * wd[i + 1] + __hcse_11 * __oldb_0d)
        xb[i + 2] = xb[i + 2] + __hcse_11 * __oldb_0
        xbd[i] = xbd[i] + 2.0 * -__oldb_0d
        xb[i] = xb[i] + 2.0 * -__oldb_0
    end
    __oldb_0d = ybd[n]
    __oldb_0 = yb[n]
    ybd[n] = 0.0
    yb[n] = 0.0
    xbd[n] = xbd[n] + __oldb_0d
    xb[n] = xb[n] + __oldb_0
    __oldb_0d = ybd[n - 1]
    __oldb_0 = yb[n - 1]
    ybd[n - 1] = 0.0
    yb[n - 1] = 0.0
    xbd[n - 1] = xbd[n - 1] + __oldb_0d
    xb[n - 1] = xb[n - 1] + __oldb_0
    __oldb_0d = ybd[2]
    __oldb_0 = yb[2]
    ybd[2] = 0.0
    yb[2] = 0.0
    xbd[2] = xbd[2] + __oldb_0d
    xb[2] = xb[2] + __oldb_0
    __oldb_0d = ybd[1]
    __oldb_0 = yb[1]
    ybd[1] = 0.0
    yb[1] = 0.0
    xbd[1] = xbd[1] + __oldb_0d
    xb[1] = xb[1] + __oldb_0
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
