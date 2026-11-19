function initstacks_gat_reduce_scalar_b(n)
    y_stack = Vector{Float64}(undef, max(0, div((n - 1) - 2, 1) + 1))
    t_stack = Vector{Float64}(undef, max(0, div((n - 1) - 2, 1) + 1) + 1)
    return (y_stack, t_stack)
end

function gat_reduce_scalar_hv(x, xb, y, yb, s, sb, n, xd, xbd, yd, ybd, sd, sbd, y_stack, t_stack)
    y_stack_d = Vector{Float64}(undef, length(y_stack))
    t_stack_d = Vector{Float64}(undef, length(t_stack))
    t = 0.0
    tb = 0.0
    td = 0.0
    tbd = 0.0
    yd[1] = xd[1]
    y[1] = x[1]
    yd[n] = xd[n]
    y[n] = x[n]
    td = 0.0
    t = 1.0
    for i = 2:n - 1
        __icse_0 = (i - 2) + 1
        __idx_y_stack_0 = __icse_0
        y_stack_d[__idx_y_stack_0] = yd[i]
        y_stack[__idx_y_stack_0] = y[i]
        yd[i] = xd[i - 1] + 2.0 * xd[i + 1]
        y[i] = x[i - 1] + 2.0 * x[i + 1]
        __idx_t_stack_3 = __icse_0
        t_stack_d[__idx_t_stack_3] = td
        t_stack[__idx_t_stack_3] = t
        __hcse_0 = 1.0 + 0.1 * y[i]
        td = __hcse_0 * td + t * (0.1 * yd[i])
        t = t * __hcse_0
    end
    sd[1] = td
    s[1] = t
    __ihcse_1 = max(0, div((n - 1) - 2, 1) + 1) + 1
    __idx_t_stack_5 = __ihcse_1
    t_stack_d[__idx_t_stack_5] = td
    t_stack[__idx_t_stack_5] = t
    __idx_t_stack_0 = __ihcse_1
    td = t_stack_d[__idx_t_stack_0]
    t = t_stack[__idx_t_stack_0]
    __oldb_0d = sbd[1]
    __oldb_0 = sb[1]
    sbd[1] = 0.0
    sb[1] = 0.0
    tbd = tbd + __oldb_0d
    tb = tb + __oldb_0
    for i = n - 1:-1:2
        __icse_1 = (i - 2) + 1
        __idx_t_stack_0 = __icse_1
        td = t_stack_d[__idx_t_stack_0]
        t = t_stack[__idx_t_stack_0]
        __oldb_2d = tbd
        __oldb_2 = tb
        tbd = 0.0
        tb = 0.0
        __hcse_2 = 1.0 + 0.1 * y[i]
        tbd = tbd + (__oldb_2 * (0.1 * yd[i]) + __hcse_2 * __oldb_2d)
        tb = tb + __hcse_2 * __oldb_2
        ybd[i] = ybd[i] + 0.1 * (__oldb_2 * td + t * __oldb_2d)
        yb[i] = yb[i] + 0.1 * (t * __oldb_2)
        __idx_y_stack_0 = __icse_1
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

function gat_reduce_scalar(x, y, s, n)
    y[1] = x[1]
    y[n] = x[n]
    t = 1.0
    for i = 2:n - 1
        y[i] = x[i - 1] + 2.0 * x[i + 1]
        t = t * (1.0 + 0.1 * y[i])
    end
    s[1] = t
end
