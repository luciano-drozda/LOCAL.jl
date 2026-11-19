function initstacks_gat_sum_scalar_b(n)
    y_stack = Vector{Float64}(undef, max(0, div((n - 1) - 2, 1) + 1))
    return y_stack
end

function gat_sum_scalar_hv(x, xb, y, yb, s, sb, n, xd, xbd, yd, ybd, sd, sbd, y_stack)
    y_stack_d = Vector{Float64}(undef, length(y_stack))
    t = 0.0
    tb = 0.0
    td = 0.0
    tbd = 0.0
    yd[1] = xd[1]
    y[1] = x[1]
    yd[n] = xd[n]
    y[n] = x[n]
    td = 0.0
    t = 0.0
    for i = 2:n - 1
        __idx_y_stack_0 = (i - 2) + 1
        y_stack_d[__idx_y_stack_0] = yd[i]
        y_stack[__idx_y_stack_0] = y[i]
        yd[i] = xd[i - 1] + 2.0 * xd[i + 1]
        y[i] = x[i - 1] + 2.0 * x[i + 1]
        __cse_0d = yd[i]
        __cse_0 = y[i]
        __hcse_0 = __cse_0 * __cse_0d
        td = td + (__hcse_0 + __hcse_0)
        t = t + __cse_0 * __cse_0
    end
    sd[1] = td
    s[1] = t
    __oldb_0d = sbd[1]
    __oldb_0 = sb[1]
    sbd[1] = 0.0
    sb[1] = 0.0
    tbd = tbd + __oldb_0d
    tb = tb + __oldb_0
    for i = n - 1:-1:2
        __hcse_1 = y[i]
        __cse_1d = tb * yd[i] + __hcse_1 * tbd
        __cse_1 = __hcse_1 * tb
        ybd[i] = ybd[i] + __cse_1d
        yb[i] = yb[i] + __cse_1
        ybd[i] = ybd[i] + __cse_1d
        yb[i] = yb[i] + __cse_1
        __idx_y_stack_0 = (i - 2) + 1
        yd[i] = y_stack_d[__idx_y_stack_0]
        y[i] = y_stack[__idx_y_stack_0]
        __oldb_2d = ybd[i]
        __oldb_2 = yb[i]
        ybd[i] = 0.0
        yb[i] = 0.0
        xbd[i - 1] = xbd[i - 1] + __oldb_2d
        xb[i - 1] = xb[i - 1] + __oldb_2
        xbd[i + 1] = xbd[i + 1] + 2.0__oldb_2d
        xb[i + 1] = xb[i + 1] + 2.0__oldb_2
    end
    tbd = 0.0
    tb = 0.0
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
