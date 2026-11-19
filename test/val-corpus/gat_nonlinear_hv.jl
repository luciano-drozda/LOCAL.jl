function initstacks_gat_nonlinear_b()
    return nothing
end

function gat_nonlinear_hv(x, xb, y, yb, n, xd, xbd, yd, ybd)
    yd[1] = xd[1]
    y[1] = x[1]
    yd[n] = xd[n]
    y[n] = x[n]
    for i = 2:n - 1
        __hcse_0 = x[i + 1]
        __hcse_1 = x[i - 1]
        __hcse_2 = sin(__hcse_1)
        __hcse_3 = exp(0.1 * x[i])
        yd[i] = (__hcse_0 * (cos(__hcse_1) * xd[i - 1]) + __hcse_2 * xd[i + 1]) + __hcse_3 * (0.1 * xd[i])
        y[i] = __hcse_2 * __hcse_0 + __hcse_3
    end
    for i = n - 1:-1:2
        __oldb_0d = ybd[i]
        __oldb_0 = yb[i]
        ybd[i] = 0.0
        yb[i] = 0.0
        __cse_0d = xd[i - 1]
        __cse_0 = x[i - 1]
        __hcse_4 = x[i + 1]
        __hcse_5 = __hcse_4 * __oldb_0
        __hcse_6 = sin(__cse_0)
        __hcse_7 = cos(__cse_0)
        xbd[i - 1] = xbd[i - 1] + (__hcse_5 * (-__hcse_6 * __cse_0d) + __hcse_7 * (__oldb_0 * xd[i + 1] + __hcse_4 * __oldb_0d))
        xb[i - 1] = xb[i - 1] + __hcse_7 * __hcse_5
        xbd[i + 1] = xbd[i + 1] + (__oldb_0 * (__hcse_7 * __cse_0d) + __hcse_6 * __oldb_0d)
        xb[i + 1] = xb[i + 1] + __hcse_6 * __oldb_0
        __hcse_8 = exp(0.1 * x[i])
        xbd[i] = xbd[i] + 0.1 * (__oldb_0 * (__hcse_8 * (0.1 * xd[i])) + __hcse_8 * __oldb_0d)
        xb[i] = xb[i] + 0.1 * (__hcse_8 * __oldb_0)
    end
    __oldb_0d = ybd[n]
    __oldb_0 = yb[n]
    ybd[n] = 0.0
    yb[n] = 0.0
    xbd[n] = xbd[n] + __oldb_0d
    xb[n] = xb[n] + __oldb_0
    __oldb_0d = ybd[1]
    __oldb_0 = yb[1]
    ybd[1] = 0.0
    yb[1] = 0.0
    xbd[1] = xbd[1] + __oldb_0d
    xb[1] = xb[1] + __oldb_0
    return nothing
end

function gat_nonlinear(x, y, n)
    y[1] = x[1]
    y[n] = x[n]
    for i = 2:n - 1
        y[i] = sin(x[i - 1]) * x[i + 1] + exp(0.1 * x[i])
    end
end
