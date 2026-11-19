function initstacks_lin_carried_sum_b(n)
    branch_stack = Vector{Int64}(undef, max(0, div(n - 1, 1) + 1))
    s_stack = Vector{Float64}(undef, ((max(0, div(n - 1, 1) + 1) + max(0, div(n - 1, 1) + 1)) + max(0, div(n - 1, 1) + 1)) + 1)
    return (branch_stack, s_stack)
end

function lin_carried_sum_b(x, xb, y, yb, z, zb, n, branch_stack, s_stack)
    s = 0.0
    sb = 0.0
    s = 1.0
    for i = 1:n
        y[i] = s + 1.0
        if x[i] > 0.0
            __icse_0 = (i - 1) + 1
            __idx_branch_stack_0 = __icse_0
            branch_stack[__idx_branch_stack_0] = 1
            __idx_s_stack_0 = __icse_0
            s_stack[__idx_s_stack_0] = s
            s = x[i]
        else
            __icse_1 = (i - 1) + 1
            __idx_branch_stack_0 = __icse_1
            branch_stack[__idx_branch_stack_0] = 0
            __idx_s_stack_0 = max(0, div(n - 1, 1) + 1) + __icse_1
            s_stack[__idx_s_stack_0] = s
            s = 2.0 * x[i]
        end
        z[i] = s * s
        __icse_2 = max(0, div(n - 1, 1) + 1)
        __idx_s_stack_3 = (__icse_2 + __icse_2) + ((i - 1) + 1)
        s_stack[__idx_s_stack_3] = s
    end
    __icse_3 = max(0, div(n - 1, 1) + 1)
    __icse_4 = ((__icse_3 + __icse_3) + __icse_3) + 1
    __idx_s_stack_2 = __icse_4
    s_stack[__idx_s_stack_2] = s
    __idx_s_stack_0 = __icse_4
    s = s_stack[__idx_s_stack_0]
    for i = n:-1:1
        __icse_5 = max(0, div(n - 1, 1) + 1)
        __icse_6 = (i - 1) + 1
        __idx_s_stack_0 = (__icse_5 + __icse_5) + __icse_6
        s = s_stack[__idx_s_stack_0]
        __oldb_0 = zb[i]
        zb[i] = 0.0
        __cse_7 = s * __oldb_0
        sb = sb + __cse_7
        sb = sb + __cse_7
        __idx_branch_stack_6 = __icse_6
        __branch = branch_stack[__idx_branch_stack_6]
        if __branch == 1
            __idx_s_stack_0 = (i - 1) + 1
            s = s_stack[__idx_s_stack_0]
            __oldb_2 = sb
            sb = 0.0
            xb[i] = xb[i] + __oldb_2
        else
            __idx_s_stack_0 = max(0, div(n - 1, 1) + 1) + ((i - 1) + 1)
            s = s_stack[__idx_s_stack_0]
            __oldb_2 = sb
            sb = 0.0
            xb[i] = xb[i] + 2.0__oldb_2
        end
        __oldb_0 = yb[i]
        yb[i] = 0.0
        sb = sb + __oldb_0
    end
    sb = 0.0
    return nothing
end

function lin_carried_sum(x, y, z, n)
    s = 1.0
    for i = 1:n
        y[i] = s + 1.0
        if x[i] > 0.0
            s = x[i]
        else
            s = 2.0 * x[i]
        end
        z[i] = s * s
    end
end
