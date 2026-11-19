function initstacks_lin_carried_scale_b(n)
    branch_stack = Vector{Int64}(undef, max(0, div(n - 1, 1) + 1))
    s_stack = Vector{Float64}(undef, ((max(0, div(n - 1, 1) + 1) + max(0, div(n - 1, 1) + 1)) + max(0, div(n - 1, 1) + 1)) + 1)
    return (branch_stack, s_stack)
end

function lin_carried_scale_hv(x, xb, y, yb, z, zb, n, xd, xbd, yd, ybd, zd, zbd, branch_stack, s_stack)
    s_stack_d = Vector{Float64}(undef, length(s_stack))
    s = 0.0
    sb = 0.0
    sd = 0.0
    sbd = 0.0
    sd = 0.0
    s = 1.0
    for i = 1:n
        yd[i] = 0.5sd
        y[i] = 0.5s
        if x[i] > 0.0
            __icse_0 = (i - 1) + 1
            __idx_branch_stack_0 = __icse_0
            branch_stack[__idx_branch_stack_0] = 1
            __idx_s_stack_0 = __icse_0
            s_stack_d[__idx_s_stack_0] = sd
            s_stack[__idx_s_stack_0] = s
            sd = xd[i]
            s = x[i]
        else
            __icse_1 = (i - 1) + 1
            __idx_branch_stack_0 = __icse_1
            branch_stack[__idx_branch_stack_0] = 0
            __idx_s_stack_0 = max(0, div(n - 1, 1) + 1) + __icse_1
            s_stack_d[__idx_s_stack_0] = sd
            s_stack[__idx_s_stack_0] = s
            sd = 2.0 * xd[i]
            s = 2.0 * x[i]
        end
        __hcse_0 = s * sd
        zd[i] = __hcse_0 + __hcse_0
        z[i] = s * s
        __icse_2 = max(0, div(n - 1, 1) + 1)
        __idx_s_stack_3 = (__icse_2 + __icse_2) + ((i - 1) + 1)
        s_stack_d[__idx_s_stack_3] = sd
        s_stack[__idx_s_stack_3] = s
    end
    __ihcse_1 = max(0, div(n - 1, 1) + 1)
    __icse_3 = __ihcse_1
    __idx_s_stack_2 = ((__icse_3 + __icse_3) + __icse_3) + 1
    s_stack_d[__idx_s_stack_2] = sd
    s_stack[__idx_s_stack_2] = s
    __icse_4 = __ihcse_1
    __idx_s_stack_0 = ((__icse_4 + __icse_4) + __icse_4) + 1
    sd = s_stack_d[__idx_s_stack_0]
    s = s_stack[__idx_s_stack_0]
    for i = n:-1:1
        __icse_5 = max(0, div(n - 1, 1) + 1)
        __icse_6 = (i - 1) + 1
        __idx_s_stack_0 = (__icse_5 + __icse_5) + __icse_6
        sd = s_stack_d[__idx_s_stack_0]
        s = s_stack[__idx_s_stack_0]
        __oldb_0d = zbd[i]
        __oldb_0 = zb[i]
        zbd[i] = 0.0
        zb[i] = 0.0
        __cse_7d = __oldb_0 * sd + s * __oldb_0d
        __cse_7 = s * __oldb_0
        sbd = sbd + __cse_7d
        sb = sb + __cse_7
        sbd = sbd + __cse_7d
        sb = sb + __cse_7
        __idx_branch_stack_6 = __icse_6
        __branch = branch_stack[__idx_branch_stack_6]
        if __branch == 1
            __idx_s_stack_0 = (i - 1) + 1
            sd = s_stack_d[__idx_s_stack_0]
            s = s_stack[__idx_s_stack_0]
            __oldb_2d = sbd
            __oldb_2 = sb
            sbd = 0.0
            sb = 0.0
            xbd[i] = xbd[i] + __oldb_2d
            xb[i] = xb[i] + __oldb_2
        else
            __idx_s_stack_0 = max(0, div(n - 1, 1) + 1) + ((i - 1) + 1)
            sd = s_stack_d[__idx_s_stack_0]
            s = s_stack[__idx_s_stack_0]
            __oldb_2d = sbd
            __oldb_2 = sb
            sbd = 0.0
            sb = 0.0
            xbd[i] = xbd[i] + 2.0__oldb_2d
            xb[i] = xb[i] + 2.0__oldb_2
        end
        __oldb_0d = ybd[i]
        __oldb_0 = yb[i]
        ybd[i] = 0.0
        yb[i] = 0.0
        sbd = sbd + 0.5__oldb_0d
        sb = sb + 0.5__oldb_0
    end
    sbd = 0.0
    sb = 0.0
    return nothing
end

function lin_carried_scale(x, y, z, n)
    s = 1.0
    for i = 1:n
        y[i] = 0.5s
        if x[i] > 0.0
            s = x[i]
        else
            s = 2.0 * x[i]
        end
        z[i] = s * s
    end
end
