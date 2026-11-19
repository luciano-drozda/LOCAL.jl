function initstacks_gat_sum_scalar_b(n)
    y_stack = Vector{Float64}(undef, max(0, div((n - 1) - 2, 1) + 1))
    return y_stack
end

function gat_sum_scalar_b(x, xb, y, yb, s, sb, n, y_stack)
    t = 0.0
    tb = 0.0
    y[1] = x[1]
    y[n] = x[n]
    t = 0.0
    for i = 2:n - 1
        __idx_y_stack_0 = (i - 2) + 1
        y_stack[__idx_y_stack_0] = y[i]
        y[i] = x[i - 1] + 2.0 * x[i + 1]
        __cse_0 = y[i]
        t = t + __cse_0 * __cse_0
    end
    s[1] = t
    __oldb_0 = sb[1]
    sb[1] = 0.0
    tb = tb + __oldb_0
    for i = n - 1:-1:2
        __cse_1 = y[i] * tb
        yb[i] = yb[i] + __cse_1
        yb[i] = yb[i] + __cse_1
        __idx_y_stack_0 = (i - 2) + 1
        y[i] = y_stack[__idx_y_stack_0]
        __oldb_2 = yb[i]
        yb[i] = 0.0
        xb[i - 1] = xb[i - 1] + __oldb_2
        xb[i + 1] = xb[i + 1] + 2.0__oldb_2
    end
    tb = 0.0
    __oldb_0 = yb[n]
    yb[n] = 0.0
    xb[n] = xb[n] + __oldb_0
    __oldb_0 = yb[1]
    yb[1] = 0.0
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
