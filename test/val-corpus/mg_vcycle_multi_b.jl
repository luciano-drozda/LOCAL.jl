function initstacks_mg_vcycle_multi_b(h1, n, nfine, nu1, nu2, num_levels)
    n = n * 2
    nl = nfine
    hl = h1
    prefix_branch_stack_1 = Vector{Int}(undef, max(0, div((num_levels - 1) - 1, 1) + 1))
    __tot_branch_stack_1 = 0
    prefix_f_stack_1 = Vector{Int}(undef, max(0, div((num_levels - 1) - 1, 1) + 1))
    __tot_f_stack_1 = 0
    prefix_hl2_stack_1 = Vector{Int}(undef, max(0, div((num_levels - 1) - 1, 1) + 1))
    __tot_hl2_stack_1 = 0
    prefix_hl_stack_1 = Vector{Int}(undef, max(0, div((num_levels - 1) - 1, 1) + 1))
    __tot_hl_stack_1 = 0
    prefix_left_stack_1 = Vector{Int}(undef, max(0, div((num_levels - 1) - 1, 1) + 1))
    __tot_left_stack_1 = 0
    prefix_right_stack_1 = Vector{Int}(undef, max(0, div((num_levels - 1) - 1, 1) + 1))
    __tot_right_stack_1 = 0
    prefix_tripcount_stack_1 = Vector{Int}(undef, max(0, div((num_levels - 1) - 1, 1) + 1))
    __tot_tripcount_stack_1 = 0
    prefix_u_stack_1 = Vector{Int}(undef, max(0, div((num_levels - 1) - 1, 1) + 1))
    __tot_u_stack_1 = 0
    val_n_1 = Vector{Int64}(undef, max(0, div((num_levels - 1) - 1, 1) + 1))
    val_nc_1 = Vector{Int64}(undef, max(0, div((num_levels - 1) - 1, 1) + 1))
    for i_level = 1:num_levels - 1
        prefix_branch_stack_1[(i_level - 1) + 1] = __tot_branch_stack_1
        prefix_f_stack_1[(i_level - 1) + 1] = __tot_f_stack_1
        prefix_hl2_stack_1[(i_level - 1) + 1] = __tot_hl2_stack_1
        prefix_hl_stack_1[(i_level - 1) + 1] = __tot_hl_stack_1
        prefix_left_stack_1[(i_level - 1) + 1] = __tot_left_stack_1
        prefix_right_stack_1[(i_level - 1) + 1] = __tot_right_stack_1
        prefix_tripcount_stack_1[(i_level - 1) + 1] = __tot_tripcount_stack_1
        prefix_u_stack_1[(i_level - 1) + 1] = __tot_u_stack_1
        n = nl - 1
        hl2 = hl * hl
        ncg = div(nl, 2)
        nc = ncg - 1
        nl = ncg
        hl = hl * 2.0
        val_n_1[(i_level - 1) + 1] = n
        val_nc_1[(i_level - 1) + 1] = nc
        __tot_branch_stack_1 = __tot_branch_stack_1 + (((max(0, div(nu1 - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + max(0, div(nu1 - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + max(0, div(n - 1, 1) + 1)) + max(0, div(n - 1, 1) + 1))
        __tot_f_stack_1 = __tot_f_stack_1 + max(0, div(nc - 1, 1) + 1)
        __tot_hl2_stack_1 = __tot_hl2_stack_1 + 1
        __tot_hl_stack_1 = __tot_hl_stack_1 + 1
        __tot_left_stack_1 = __tot_left_stack_1 + max(0, div(n - 1, 1) + 1)
        __tot_right_stack_1 = __tot_right_stack_1 + max(0, div(n - 1, 1) + 1)
        __tot_tripcount_stack_1 = __tot_tripcount_stack_1 + (((max(0, div(nu1 - 1, 1) + 1) + 1) + 1) + 1)
        __tot_u_stack_1 = __tot_u_stack_1 + (max(0, div(nu1 - 1, 1) + 1) * max(0, div(n - 1, 1) + 1) + max(0, div(nc - 1, 1) + 1))
    end
    hl2 = hl * hl
    prefix_branch_stack_2 = Vector{Int}(undef, max(0, div(1 - (num_levels - 1), -1) + 1))
    __tot_branch_stack_2 = 0
    prefix_hl2_stack_2 = Vector{Int}(undef, max(0, div(1 - (num_levels - 1), -1) + 1))
    __tot_hl2_stack_2 = 0
    prefix_hl_stack_2 = Vector{Int}(undef, max(0, div(1 - (num_levels - 1), -1) + 1))
    __tot_hl_stack_2 = 0
    prefix_tripcount_stack_2 = Vector{Int}(undef, max(0, div(1 - (num_levels - 1), -1) + 1))
    __tot_tripcount_stack_2 = 0
    prefix_u_stack_2 = Vector{Int}(undef, max(0, div(1 - (num_levels - 1), -1) + 1))
    __tot_u_stack_2 = 0
    val_n_2 = Vector{Int64}(undef, max(0, div(1 - (num_levels - 1), -1) + 1))
    val_nc_2 = Vector{Int64}(undef, max(0, div(1 - (num_levels - 1), -1) + 1))
    for i_level = num_levels - 1:-1:1
        prefix_branch_stack_2[div(i_level - (num_levels - 1), -1) + 1] = __tot_branch_stack_2
        prefix_hl2_stack_2[div(i_level - (num_levels - 1), -1) + 1] = __tot_hl2_stack_2
        prefix_hl_stack_2[div(i_level - (num_levels - 1), -1) + 1] = __tot_hl_stack_2
        prefix_tripcount_stack_2[div(i_level - (num_levels - 1), -1) + 1] = __tot_tripcount_stack_2
        prefix_u_stack_2[div(i_level - (num_levels - 1), -1) + 1] = __tot_u_stack_2
        nl = nl * 2
        hl = hl / 2.0
        n = nl - 1
        ncg = div(nl, 2)
        nc = ncg - 1
        hl2 = hl * hl
        val_n_2[div(i_level - (num_levels - 1), -1) + 1] = n
        val_nc_2[div(i_level - (num_levels - 1), -1) + 1] = nc
        __tot_branch_stack_2 = __tot_branch_stack_2 + (((max(0, div((nc + 1) - 1, 1) + 1) + max(0, div((nc + 1) - 1, 1) + 1)) + max(0, div(nu2 - 1, 1) + 1) * max(0, div(n - 1, 1) + 1)) + max(0, div(nu2 - 1, 1) + 1) * max(0, div(n - 1, 1) + 1))
        __tot_hl2_stack_2 = __tot_hl2_stack_2 + 1
        __tot_hl_stack_2 = __tot_hl_stack_2 + 1
        __tot_tripcount_stack_2 = __tot_tripcount_stack_2 + ((1 + 1) + max(0, div(nu2 - 1, 1) + 1))
        __tot_u_stack_2 = __tot_u_stack_2 + ((max(0, div(nc - 1, 1) + 1) + max(0, div((nc + 1) - 1, 1) + 1)) + max(0, div(nu2 - 1, 1) + 1) * max(0, div(n - 1, 1) + 1))
    end
    hl2_stack = Vector{Float64}(undef, ((__tot_hl2_stack_1 + 1) + __tot_hl2_stack_2) + 1)
    tripcount_stack = Vector{Int64}(undef, __tot_tripcount_stack_1 + __tot_tripcount_stack_2)
    branch_stack = Vector{Int64}(undef, __tot_branch_stack_1 + __tot_branch_stack_2)
    u_stack = Vector{Float64}(undef, (__tot_u_stack_1 + 1) + __tot_u_stack_2)
    f_stack = Vector{Float64}(undef, __tot_f_stack_1)
    hl_stack = Vector{Float64}(undef, (__tot_hl_stack_1 + __tot_hl_stack_2) + 1)
    left_stack = Vector{Float64}(undef, __tot_left_stack_1)
    right_stack = Vector{Float64}(undef, __tot_right_stack_1)
    return (hl2_stack, tripcount_stack, branch_stack, u_stack, f_stack, hl_stack, left_stack, right_stack, prefix_branch_stack_1, prefix_f_stack_1, prefix_hl2_stack_1, prefix_hl_stack_1, prefix_left_stack_1, prefix_right_stack_1, prefix_tripcount_stack_1, prefix_u_stack_1, prefix_branch_stack_2, prefix_hl2_stack_2, prefix_hl_stack_2, prefix_tripcount_stack_2, prefix_u_stack_2, __tot_branch_stack_1, __tot_f_stack_1, __tot_hl2_stack_1, __tot_hl_stack_1, __tot_left_stack_1, __tot_right_stack_1, __tot_tripcount_stack_1, __tot_u_stack_1, __tot_branch_stack_2, __tot_hl2_stack_2, __tot_hl_stack_2, __tot_tripcount_stack_2, __tot_u_stack_2, val_n_1, val_nc_1, val_n_2, val_nc_2)
end

function mg_vcycle_multi_b(u, ub, f, fb, r, rb, nfine, num_levels, h1, h1b, nu1, nu2, n, hl2_stack, tripcount_stack, branch_stack, u_stack, f_stack, hl_stack, left_stack, right_stack, prefix_branch_stack_1, prefix_f_stack_1, prefix_hl2_stack_1, prefix_hl_stack_1, prefix_left_stack_1, prefix_right_stack_1, prefix_tripcount_stack_1, prefix_u_stack_1, prefix_branch_stack_2, prefix_hl2_stack_2, prefix_hl_stack_2, prefix_tripcount_stack_2, prefix_u_stack_2, __tot_branch_stack_1, __tot_f_stack_1, __tot_hl2_stack_1, __tot_hl_stack_1, __tot_left_stack_1, __tot_right_stack_1, __tot_tripcount_stack_1, __tot_u_stack_1, __tot_branch_stack_2, __tot_hl2_stack_2, __tot_hl_stack_2, __tot_tripcount_stack_2, __tot_u_stack_2, val_n_1, val_nc_1, val_n_2, val_nc_2)
    cl = 0.0
    cr = 0.0
    hl = 0.0
    hl2 = 0.0
    left = 0.0
    left_mg_relax_c1 = 0.0
    left_mg_relax_c2 = 0.0
    right = 0.0
    right_mg_relax_c1 = 0.0
    right_mg_relax_c2 = 0.0
    clb = 0.0
    crb = 0.0
    hlb = 0.0
    hl2b = 0.0
    leftb = 0.0
    left_mg_relax_c1b = 0.0
    left_mg_relax_c2b = 0.0
    rightb = 0.0
    right_mg_relax_c1b = 0.0
    right_mg_relax_c2b = 0.0
    n = n * 2
    nl = nfine
    hl = h1
    for i_level = 1:num_levels - 1
        n = nl - 1
        __idx_hl2_stack_1_1 = prefix_hl2_stack_1[(i_level - 1) + 1] + 1
        hl2_stack[__idx_hl2_stack_1_1] = hl2
        hl2 = hl * hl
        for i_k_mg_relax_c1 = 1:nu1
            __idx_tripcount_stack_1_0 = prefix_tripcount_stack_1[(i_level - 1) + 1] + ((i_k_mg_relax_c1 - 1) + 1)
            tripcount_stack[__idx_tripcount_stack_1_0] = n
            for i_j_mg_relax_c1 = 1:n
                left_mg_relax_c1 = 0.0
                if i_j_mg_relax_c1 > 1
                    __idx_branch_stack_1_0 = prefix_branch_stack_1[(i_level - 1) + 1] + (((i_k_mg_relax_c1 - 1) * (div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1) + (i_j_mg_relax_c1 - 1)) + 1)
                    branch_stack[__idx_branch_stack_1_0] = 1
                    left_mg_relax_c1 = u[i_j_mg_relax_c1 - 1, i_level]
                else
                    __idx_branch_stack_1_0 = prefix_branch_stack_1[(i_level - 1) + 1] + (((i_k_mg_relax_c1 - 1) * (div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1) + (i_j_mg_relax_c1 - 1)) + 1)
                    branch_stack[__idx_branch_stack_1_0] = 0
                end
                right_mg_relax_c1 = 0.0
                if i_j_mg_relax_c1 < n
                    __icse_0 = div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1
                    __idx_branch_stack_1_0 = (prefix_branch_stack_1[(i_level - 1) + 1] + max(0, div(nu1 - 1, 1) + 1) * max(0, __icse_0)) + (((i_k_mg_relax_c1 - 1) * __icse_0 + (i_j_mg_relax_c1 - 1)) + 1)
                    branch_stack[__idx_branch_stack_1_0] = 1
                    right_mg_relax_c1 = u[i_j_mg_relax_c1 + 1, i_level]
                else
                    __icse_1 = div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1
                    __idx_branch_stack_1_0 = (prefix_branch_stack_1[(i_level - 1) + 1] + max(0, div(nu1 - 1, 1) + 1) * max(0, __icse_1)) + (((i_k_mg_relax_c1 - 1) * __icse_1 + (i_j_mg_relax_c1 - 1)) + 1)
                    branch_stack[__idx_branch_stack_1_0] = 0
                end
                __idx_u_stack_1_4 = prefix_u_stack_1[(i_level - 1) + 1] + (((i_k_mg_relax_c1 - 1) * (div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1) + (i_j_mg_relax_c1 - 1)) + 1)
                u_stack[__idx_u_stack_1_4] = u[i_j_mg_relax_c1, i_level]
                u[i_j_mg_relax_c1, i_level] = 0.5 * (hl2 * f[i_j_mg_relax_c1, i_level] + left_mg_relax_c1 + right_mg_relax_c1)
            end
        end
        __idx_tripcount_stack_1_5 = (prefix_tripcount_stack_1[(i_level - 1) + 1] + max(0, div(nu1 - 1, 1) + 1)) + 1
        tripcount_stack[__idx_tripcount_stack_1_5] = n
        for j = 1:n
            left = 0.0
            if j > 1
                __icse_2 = max(0, div(nu1 - 1, 1) + 1) * max(0, div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1)
                __idx_branch_stack_1_0 = (prefix_branch_stack_1[(i_level - 1) + 1] + (__icse_2 + __icse_2)) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 1
                left = u[j - 1, i_level]
            else
                __icse_3 = max(0, div(nu1 - 1, 1) + 1) * max(0, div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1)
                __idx_branch_stack_1_0 = (prefix_branch_stack_1[(i_level - 1) + 1] + (__icse_3 + __icse_3)) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 0
            end
            right = 0.0
            if j < n
                __icse_4 = max(0, div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1)
                __icse_5 = max(0, div(nu1 - 1, 1) + 1) * __icse_4
                __idx_branch_stack_1_0 = (prefix_branch_stack_1[(i_level - 1) + 1] + ((__icse_5 + __icse_5) + __icse_4)) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 1
                right = u[j + 1, i_level]
            else
                __icse_6 = max(0, div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1)
                __icse_7 = max(0, div(nu1 - 1, 1) + 1) * __icse_6
                __idx_branch_stack_1_0 = (prefix_branch_stack_1[(i_level - 1) + 1] + ((__icse_7 + __icse_7) + __icse_6)) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 0
            end
            r[j, i_level] = f[j, i_level] - ((2.0 * u[j, i_level] - left) - right) / hl2
            __icse_8 = (j - 1) + 1
            __idx_left_stack_1_5 = prefix_left_stack_1[(i_level - 1) + 1] + __icse_8
            left_stack[__idx_left_stack_1_5] = left
            __idx_right_stack_1_7 = prefix_right_stack_1[(i_level - 1) + 1] + __icse_8
            right_stack[__idx_right_stack_1_7] = right
        end
        ncg = div(nl, 2)
        nc = ncg - 1
        __idx_tripcount_stack_1_10 = (prefix_tripcount_stack_1[(i_level - 1) + 1] + (max(0, div(nu1 - 1, 1) + 1) + 1)) + 1
        tripcount_stack[__idx_tripcount_stack_1_10] = nc
        for j = 1:nc
            jf = j * 2
            __idx_f_stack_1_1 = prefix_f_stack_1[(i_level - 1) + 1] + ((j - 1) + 1)
            f_stack[__idx_f_stack_1_1] = f[j, i_level + 1]
            f[j, i_level + 1] = 0.25 * r[jf - 1, i_level] + 0.5 * r[jf, i_level] + 0.25 * r[jf + 1, i_level]
        end
        __idx_tripcount_stack_1_13 = (prefix_tripcount_stack_1[(i_level - 1) + 1] + ((max(0, div(nu1 - 1, 1) + 1) + 1) + 1)) + 1
        tripcount_stack[__idx_tripcount_stack_1_13] = nc
        for j = 1:nc
            __idx_u_stack_1_0 = (prefix_u_stack_1[(i_level - 1) + 1] + max(0, div(nu1 - 1, 1) + 1) * max(0, div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1)) + ((j - 1) + 1)
            u_stack[__idx_u_stack_1_0] = u[j, i_level + 1]
            u[j, i_level + 1] = 0.0
        end
        nl = ncg
        __idx_hl_stack_1_17 = prefix_hl_stack_1[(i_level - 1) + 1] + 1
        hl_stack[__idx_hl_stack_1_17] = hl
        hl = hl * 2.0
    end
    __idx_hl2_stack_4 = __tot_hl2_stack_1 + 1
    hl2_stack[__idx_hl2_stack_4] = hl2
    hl2 = hl * hl
    __idx_u_stack_7 = __tot_u_stack_1 + 1
    u_stack[__idx_u_stack_7] = u[1, num_levels]
    u[1, num_levels] = 0.5 * hl2 * f[1, num_levels]
    for i_level = num_levels - 1:-1:1
        nl = nl * 2
        __idx_hl_stack_2_1 = (__tot_hl_stack_1 + prefix_hl_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + 1
        hl_stack[__idx_hl_stack_2_1] = hl
        hl = hl / 2.0
        n = nl - 1
        ncg = div(nl, 2)
        nc = ncg - 1
        __idx_hl2_stack_2_7 = ((__tot_hl2_stack_1 + 1) + prefix_hl2_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + 1
        hl2_stack[__idx_hl2_stack_2_7] = hl2
        hl2 = hl * hl
        __idx_tripcount_stack_2_10 = (__tot_tripcount_stack_1 + prefix_tripcount_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + 1
        tripcount_stack[__idx_tripcount_stack_2_10] = nc
        for j = 1:nc
            jf = j * 2
            __idx_u_stack_2_1 = ((__tot_u_stack_1 + 1) + prefix_u_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + ((j - 1) + 1)
            __cse_9 = u[jf, i_level]
            u_stack[__idx_u_stack_2_1] = __cse_9
            u[jf, i_level] = __cse_9 + u[j, i_level + 1]
        end
        __idx_tripcount_stack_2_13 = ((__tot_tripcount_stack_1 + prefix_tripcount_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + 1) + 1
        tripcount_stack[__idx_tripcount_stack_2_13] = nc
        for j = 1:nc + 1
            jf = j * 2 - 1
            cl = 0.0
            if j > 1
                __idx_branch_stack_2_0 = (__tot_branch_stack_1 + prefix_branch_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_2_0] = 1
                cl = u[j - 1, i_level + 1]
            else
                __idx_branch_stack_2_0 = (__tot_branch_stack_1 + prefix_branch_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_2_0] = 0
            end
            cr = 0.0
            if j <= nc
                __idx_branch_stack_2_0 = ((__tot_branch_stack_1 + prefix_branch_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + max(0, div((val_nc_2[div(i_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1)) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_2_0] = 1
                cr = u[j, i_level + 1]
            else
                __idx_branch_stack_2_0 = ((__tot_branch_stack_1 + prefix_branch_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + max(0, div((val_nc_2[div(i_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1)) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_2_0] = 0
            end
            __idx_u_stack_2_5 = (((__tot_u_stack_1 + 1) + prefix_u_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + max(0, div(val_nc_2[div(i_level - (num_levels - 1), -1) + 1] - 1, 1) + 1)) + ((j - 1) + 1)
            __cse_10 = u[jf, i_level]
            u_stack[__idx_u_stack_2_5] = __cse_10
            u[jf, i_level] = __cse_10 + 0.5 * (cl + cr)
        end
        for i_k_mg_relax_c2 = 1:nu2
            __idx_tripcount_stack_2_0 = ((__tot_tripcount_stack_1 + prefix_tripcount_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + (1 + 1)) + ((i_k_mg_relax_c2 - 1) + 1)
            tripcount_stack[__idx_tripcount_stack_2_0] = n
            for i_j_mg_relax_c2 = 1:n
                left_mg_relax_c2 = 0.0
                if i_j_mg_relax_c2 > 1
                    __icse_11 = max(0, div((val_nc_2[div(i_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1)
                    __idx_branch_stack_2_0 = ((__tot_branch_stack_1 + prefix_branch_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + (__icse_11 + __icse_11)) + (((i_k_mg_relax_c2 - 1) * (div(val_n_2[div(i_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (i_j_mg_relax_c2 - 1)) + 1)
                    branch_stack[__idx_branch_stack_2_0] = 1
                    left_mg_relax_c2 = u[i_j_mg_relax_c2 - 1, i_level]
                else
                    __icse_12 = max(0, div((val_nc_2[div(i_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1)
                    __idx_branch_stack_2_0 = ((__tot_branch_stack_1 + prefix_branch_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + (__icse_12 + __icse_12)) + (((i_k_mg_relax_c2 - 1) * (div(val_n_2[div(i_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (i_j_mg_relax_c2 - 1)) + 1)
                    branch_stack[__idx_branch_stack_2_0] = 0
                end
                right_mg_relax_c2 = 0.0
                if i_j_mg_relax_c2 < n
                    __icse_13 = max(0, div((val_nc_2[div(i_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1)
                    __icse_14 = div(val_n_2[div(i_level - (num_levels - 1), -1) + 1] - 1, 1) + 1
                    __idx_branch_stack_2_0 = ((__tot_branch_stack_1 + prefix_branch_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + ((__icse_13 + __icse_13) + max(0, div(nu2 - 1, 1) + 1) * max(0, __icse_14))) + (((i_k_mg_relax_c2 - 1) * __icse_14 + (i_j_mg_relax_c2 - 1)) + 1)
                    branch_stack[__idx_branch_stack_2_0] = 1
                    right_mg_relax_c2 = u[i_j_mg_relax_c2 + 1, i_level]
                else
                    __icse_15 = max(0, div((val_nc_2[div(i_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1)
                    __icse_16 = div(val_n_2[div(i_level - (num_levels - 1), -1) + 1] - 1, 1) + 1
                    __idx_branch_stack_2_0 = ((__tot_branch_stack_1 + prefix_branch_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + ((__icse_15 + __icse_15) + max(0, div(nu2 - 1, 1) + 1) * max(0, __icse_16))) + (((i_k_mg_relax_c2 - 1) * __icse_16 + (i_j_mg_relax_c2 - 1)) + 1)
                    branch_stack[__idx_branch_stack_2_0] = 0
                end
                __icse_17 = val_nc_2[div(i_level - (num_levels - 1), -1) + 1]
                __idx_u_stack_2_4 = (((__tot_u_stack_1 + 1) + prefix_u_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + (max(0, div(__icse_17 - 1, 1) + 1) + max(0, div((__icse_17 + 1) - 1, 1) + 1))) + (((i_k_mg_relax_c2 - 1) * (div(val_n_2[div(i_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (i_j_mg_relax_c2 - 1)) + 1)
                u_stack[__idx_u_stack_2_4] = u[i_j_mg_relax_c2, i_level]
                u[i_j_mg_relax_c2, i_level] = 0.5 * (hl2 * f[i_j_mg_relax_c2, i_level] + left_mg_relax_c2 + right_mg_relax_c2)
            end
        end
    end
    __icse_18 = (__tot_hl_stack_1 + __tot_hl_stack_2) + 1
    __idx_hl_stack_11 = __icse_18
    hl_stack[__idx_hl_stack_11] = hl
    __icse_19 = ((__tot_hl2_stack_1 + 1) + __tot_hl2_stack_2) + 1
    __idx_hl2_stack_13 = __icse_19
    hl2_stack[__idx_hl2_stack_13] = hl2
    __idx_hl_stack_0 = __icse_18
    hl = hl_stack[__idx_hl_stack_0]
    __idx_hl2_stack_2 = __icse_19
    hl2 = hl2_stack[__idx_hl2_stack_2]
    for i_level = 1:num_levels - 1
        for i_k_mg_relax_c2 = nu2:-1:1
            __idx_tripcount_stack_2_0 = ((__tot_tripcount_stack_1 + prefix_tripcount_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + (1 + 1)) + ((i_k_mg_relax_c2 - 1) + 1)
            n = tripcount_stack[__idx_tripcount_stack_2_0]
            for i_j_mg_relax_c2 = n:-1:1
                __icse_20 = val_nc_2[div(i_level - (num_levels - 1), -1) + 1]
                __icse_21 = max(0, div((__icse_20 + 1) - 1, 1) + 1)
                __icse_22 = div(val_n_2[div(i_level - (num_levels - 1), -1) + 1] - 1, 1) + 1
                __icse_23 = ((i_k_mg_relax_c2 - 1) * __icse_22 + (i_j_mg_relax_c2 - 1)) + 1
                __idx_u_stack_2_0 = (((__tot_u_stack_1 + 1) + prefix_u_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + (max(0, div(__icse_20 - 1, 1) + 1) + __icse_21)) + __icse_23
                u[i_j_mg_relax_c2, i_level] = u_stack[__idx_u_stack_2_0]
                __oldb_2 = ub[i_j_mg_relax_c2, i_level]
                ub[i_j_mg_relax_c2, i_level] = 0.0
                __cse_24 = 0.5__oldb_2
                hl2b = hl2b + f[i_j_mg_relax_c2, i_level] * __cse_24
                fb[i_j_mg_relax_c2, i_level] = fb[i_j_mg_relax_c2, i_level] + hl2 * __cse_24
                left_mg_relax_c2b = left_mg_relax_c2b + __cse_24
                right_mg_relax_c2b = right_mg_relax_c2b + __cse_24
                __idx_branch_stack_2_8 = ((__tot_branch_stack_1 + prefix_branch_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + ((__icse_21 + __icse_21) + max(0, div(nu2 - 1, 1) + 1) * max(0, __icse_22))) + __icse_23
                __branch = branch_stack[__idx_branch_stack_2_8]
                if __branch == 1
                    __oldb_0 = right_mg_relax_c2b
                    right_mg_relax_c2b = 0.0
                    ub[i_j_mg_relax_c2 + 1, i_level] = ub[i_j_mg_relax_c2 + 1, i_level] + __oldb_0
                end
                right_mg_relax_c2b = 0.0
                __icse_25 = max(0, div((val_nc_2[div(i_level - (num_levels - 1), -1) + 1] + 1) - 1, 1) + 1)
                __idx_branch_stack_2_12 = ((__tot_branch_stack_1 + prefix_branch_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + (__icse_25 + __icse_25)) + (((i_k_mg_relax_c2 - 1) * (div(val_n_2[div(i_level - (num_levels - 1), -1) + 1] - 1, 1) + 1) + (i_j_mg_relax_c2 - 1)) + 1)
                __branch = branch_stack[__idx_branch_stack_2_12]
                if __branch == 1
                    __oldb_0 = left_mg_relax_c2b
                    left_mg_relax_c2b = 0.0
                    ub[i_j_mg_relax_c2 - 1, i_level] = ub[i_j_mg_relax_c2 - 1, i_level] + __oldb_0
                end
                left_mg_relax_c2b = 0.0
            end
        end
        __idx_tripcount_stack_2_1 = ((__tot_tripcount_stack_1 + prefix_tripcount_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + 1) + 1
        nc = tripcount_stack[__idx_tripcount_stack_2_1]
        for j = nc + 1:-1:1
            jf = j * 2 - 1
            __icse_26 = val_nc_2[div(i_level - (num_levels - 1), -1) + 1]
            __icse_27 = (j - 1) + 1
            __idx_u_stack_2_0 = (((__tot_u_stack_1 + 1) + prefix_u_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + max(0, div(__icse_26 - 1, 1) + 1)) + __icse_27
            u[jf, i_level] = u_stack[__idx_u_stack_2_0]
            __cse_28 = 0.5 * ub[jf, i_level]
            clb = clb + __cse_28
            crb = crb + __cse_28
            __idx_branch_stack_2_5 = ((__tot_branch_stack_1 + prefix_branch_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + max(0, div((__icse_26 + 1) - 1, 1) + 1)) + __icse_27
            __branch = branch_stack[__idx_branch_stack_2_5]
            if __branch == 1
                __oldb_0 = crb
                crb = 0.0
                ub[j, i_level + 1] = ub[j, i_level + 1] + __oldb_0
            end
            crb = 0.0
            __idx_branch_stack_2_9 = (__tot_branch_stack_1 + prefix_branch_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + ((j - 1) + 1)
            __branch = branch_stack[__idx_branch_stack_2_9]
            if __branch == 1
                __oldb_0 = clb
                clb = 0.0
                ub[j - 1, i_level + 1] = ub[j - 1, i_level + 1] + __oldb_0
            end
            clb = 0.0
        end
        __idx_tripcount_stack_2_4 = (__tot_tripcount_stack_1 + prefix_tripcount_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + 1
        nc = tripcount_stack[__idx_tripcount_stack_2_4]
        for j = nc:-1:1
            jf = j * 2
            __idx_u_stack_2_0 = ((__tot_u_stack_1 + 1) + prefix_u_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + ((j - 1) + 1)
            u[jf, i_level] = u_stack[__idx_u_stack_2_0]
            ub[j, i_level + 1] = ub[j, i_level + 1] + ub[jf, i_level]
        end
        __idx_hl2_stack_2_0 = ((__tot_hl2_stack_1 + 1) + prefix_hl2_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + 1
        hl2 = hl2_stack[__idx_hl2_stack_2_0]
        __oldb_2 = hl2b
        hl2b = 0.0
        __cse_29 = hl * __oldb_2
        hlb = hlb + __cse_29
        hlb = hlb + __cse_29
        __idx_hl_stack_2_0 = (__tot_hl_stack_1 + prefix_hl_stack_2[div(i_level - (num_levels - 1), -1) + 1]) + 1
        hl = hl_stack[__idx_hl_stack_2_0]
        __oldb_2 = hlb
        hlb = 0.0
        hlb = hlb + 0.5__oldb_2
    end
    __idx_u_stack_0 = __tot_u_stack_1 + 1
    u[1, num_levels] = u_stack[__idx_u_stack_0]
    __oldb_2 = ub[1, num_levels]
    ub[1, num_levels] = 0.0
    hl2b = hl2b + (0.5 * f[1, num_levels]) * __oldb_2
    fb[1, num_levels] = fb[1, num_levels] + (0.5hl2) * __oldb_2
    __idx_hl2_stack_0 = __tot_hl2_stack_1 + 1
    hl2 = hl2_stack[__idx_hl2_stack_0]
    __oldb_2 = hl2b
    hl2b = 0.0
    __cse_30 = hl * __oldb_2
    hlb = hlb + __cse_30
    hlb = hlb + __cse_30
    for i_level = num_levels - 1:-1:1
        __idx_hl_stack_1_0 = prefix_hl_stack_1[(i_level - 1) + 1] + 1
        hl = hl_stack[__idx_hl_stack_1_0]
        __oldb_2 = hlb
        hlb = 0.0
        hlb = hlb + 2.0__oldb_2
        __idx_tripcount_stack_1_5 = (prefix_tripcount_stack_1[(i_level - 1) + 1] + ((max(0, div(nu1 - 1, 1) + 1) + 1) + 1)) + 1
        nc = tripcount_stack[__idx_tripcount_stack_1_5]
        for j = nc:-1:1
            __idx_u_stack_1_0 = (prefix_u_stack_1[(i_level - 1) + 1] + max(0, div(nu1 - 1, 1) + 1) * max(0, div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1)) + ((j - 1) + 1)
            u[j, i_level + 1] = u_stack[__idx_u_stack_1_0]
            ub[j, i_level + 1] = 0.0
        end
        __idx_tripcount_stack_1_8 = (prefix_tripcount_stack_1[(i_level - 1) + 1] + (max(0, div(nu1 - 1, 1) + 1) + 1)) + 1
        nc = tripcount_stack[__idx_tripcount_stack_1_8]
        for j = nc:-1:1
            jf = j * 2
            __idx_f_stack_1_0 = prefix_f_stack_1[(i_level - 1) + 1] + ((j - 1) + 1)
            f[j, i_level + 1] = f_stack[__idx_f_stack_1_0]
            __oldb_2 = fb[j, i_level + 1]
            fb[j, i_level + 1] = 0.0
            __cse_31 = 0.25__oldb_2
            rb[jf - 1, i_level] = rb[jf - 1, i_level] + __cse_31
            rb[jf, i_level] = rb[jf, i_level] + 0.5__oldb_2
            rb[jf + 1, i_level] = rb[jf + 1, i_level] + __cse_31
        end
        __idx_tripcount_stack_1_11 = (prefix_tripcount_stack_1[(i_level - 1) + 1] + max(0, div(nu1 - 1, 1) + 1)) + 1
        n = tripcount_stack[__idx_tripcount_stack_1_11]
        for j = n:-1:1
            __icse_32 = (j - 1) + 1
            __idx_left_stack_1_0 = prefix_left_stack_1[(i_level - 1) + 1] + __icse_32
            left = left_stack[__idx_left_stack_1_0]
            __idx_right_stack_1_2 = prefix_right_stack_1[(i_level - 1) + 1] + __icse_32
            right = right_stack[__idx_right_stack_1_2]
            __oldb_0 = rb[j, i_level]
            rb[j, i_level] = 0.0
            fb[j, i_level] = fb[j, i_level] + __oldb_0
            __cse_33 = -__oldb_0
            __cse_34 = (1.0 / hl2) * __cse_33
            ub[j, i_level] = ub[j, i_level] + 2.0__cse_34
            __cse_35 = -__cse_34
            leftb = leftb + __cse_35
            rightb = rightb + __cse_35
            hl2b = hl2b + -(((2.0 * u[j, i_level] - left) - right) / hl2 ^ 2) * __cse_33
            __icse_36 = max(0, div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1)
            __icse_37 = max(0, div(nu1 - 1, 1) + 1) * __icse_36
            __idx_branch_stack_1_11 = (prefix_branch_stack_1[(i_level - 1) + 1] + ((__icse_37 + __icse_37) + __icse_36)) + __icse_32
            __branch = branch_stack[__idx_branch_stack_1_11]
            if __branch == 1
                __oldb_0 = rightb
                rightb = 0.0
                ub[j + 1, i_level] = ub[j + 1, i_level] + __oldb_0
            end
            rightb = 0.0
            __icse_38 = max(0, div(nu1 - 1, 1) + 1) * max(0, div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1)
            __idx_branch_stack_1_15 = (prefix_branch_stack_1[(i_level - 1) + 1] + (__icse_38 + __icse_38)) + ((j - 1) + 1)
            __branch = branch_stack[__idx_branch_stack_1_15]
            if __branch == 1
                __oldb_0 = leftb
                leftb = 0.0
                ub[j - 1, i_level] = ub[j - 1, i_level] + __oldb_0
            end
            leftb = 0.0
        end
        for i_k_mg_relax_c1 = nu1:-1:1
            __idx_tripcount_stack_1_0 = prefix_tripcount_stack_1[(i_level - 1) + 1] + ((i_k_mg_relax_c1 - 1) + 1)
            n = tripcount_stack[__idx_tripcount_stack_1_0]
            for i_j_mg_relax_c1 = n:-1:1
                __icse_39 = div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1
                __icse_40 = ((i_k_mg_relax_c1 - 1) * __icse_39 + (i_j_mg_relax_c1 - 1)) + 1
                __idx_u_stack_1_0 = prefix_u_stack_1[(i_level - 1) + 1] + __icse_40
                u[i_j_mg_relax_c1, i_level] = u_stack[__idx_u_stack_1_0]
                __oldb_2 = ub[i_j_mg_relax_c1, i_level]
                ub[i_j_mg_relax_c1, i_level] = 0.0
                __cse_41 = 0.5__oldb_2
                hl2b = hl2b + f[i_j_mg_relax_c1, i_level] * __cse_41
                fb[i_j_mg_relax_c1, i_level] = fb[i_j_mg_relax_c1, i_level] + hl2 * __cse_41
                left_mg_relax_c1b = left_mg_relax_c1b + __cse_41
                right_mg_relax_c1b = right_mg_relax_c1b + __cse_41
                __idx_branch_stack_1_8 = (prefix_branch_stack_1[(i_level - 1) + 1] + max(0, div(nu1 - 1, 1) + 1) * max(0, __icse_39)) + __icse_40
                __branch = branch_stack[__idx_branch_stack_1_8]
                if __branch == 1
                    __oldb_0 = right_mg_relax_c1b
                    right_mg_relax_c1b = 0.0
                    ub[i_j_mg_relax_c1 + 1, i_level] = ub[i_j_mg_relax_c1 + 1, i_level] + __oldb_0
                end
                right_mg_relax_c1b = 0.0
                __idx_branch_stack_1_12 = prefix_branch_stack_1[(i_level - 1) + 1] + (((i_k_mg_relax_c1 - 1) * (div(val_n_1[(i_level - 1) + 1] - 1, 1) + 1) + (i_j_mg_relax_c1 - 1)) + 1)
                __branch = branch_stack[__idx_branch_stack_1_12]
                if __branch == 1
                    __oldb_0 = left_mg_relax_c1b
                    left_mg_relax_c1b = 0.0
                    ub[i_j_mg_relax_c1 - 1, i_level] = ub[i_j_mg_relax_c1 - 1, i_level] + __oldb_0
                end
                left_mg_relax_c1b = 0.0
            end
        end
        __idx_hl2_stack_1_0 = prefix_hl2_stack_1[(i_level - 1) + 1] + 1
        hl2 = hl2_stack[__idx_hl2_stack_1_0]
        __oldb_2 = hl2b
        hl2b = 0.0
        __cse_42 = hl * __oldb_2
        hlb = hlb + __cse_42
        hlb = hlb + __cse_42
    end
    __oldb_0 = hlb
    hlb = 0.0
    h1b = h1b + __oldb_0
    return h1b
end

function mg_vcycle_multi(u, f, r, nfine, num_levels, h1, nu1, nu2, n)
    n = n * 2
    nl = nfine
    hl = h1
    for i_level = 1:num_levels - 1
        n = nl - 1
        hl2 = hl * hl
        for i_k_mg_relax_c1 = 1:nu1
            for i_j_mg_relax_c1 = 1:n
                left_mg_relax_c1 = 0.0
                if i_j_mg_relax_c1 > 1
                    left_mg_relax_c1 = u[i_j_mg_relax_c1 - 1, i_level]
                end
                right_mg_relax_c1 = 0.0
                if i_j_mg_relax_c1 < n
                    right_mg_relax_c1 = u[i_j_mg_relax_c1 + 1, i_level]
                end
                u[i_j_mg_relax_c1, i_level] = 0.5 * (hl2 * f[i_j_mg_relax_c1, i_level] + left_mg_relax_c1 + right_mg_relax_c1)
            end
        end
        for j = 1:n
            left = 0.0
            if j > 1
                left = u[j - 1, i_level]
            end
            right = 0.0
            if j < n
                right = u[j + 1, i_level]
            end
            r[j, i_level] = f[j, i_level] - ((2.0 * u[j, i_level] - left) - right) / hl2
        end
        ncg = div(nl, 2)
        nc = ncg - 1
        for j = 1:nc
            jf = j * 2
            f[j, i_level + 1] = 0.25 * r[jf - 1, i_level] + 0.5 * r[jf, i_level] + 0.25 * r[jf + 1, i_level]
        end
        for j = 1:nc
            u[j, i_level + 1] = 0.0
        end
        nl = ncg
        hl = hl * 2.0
    end
    hl2 = hl * hl
    u[1, num_levels] = 0.5 * hl2 * f[1, num_levels]
    for i_level = num_levels - 1:-1:1
        nl = nl * 2
        hl = hl / 2.0
        n = nl - 1
        ncg = div(nl, 2)
        nc = ncg - 1
        hl2 = hl * hl
        for j = 1:nc
            jf = j * 2
            u[jf, i_level] = u[jf, i_level] + u[j, i_level + 1]
        end
        for j = 1:nc + 1
            jf = j * 2 - 1
            cl = 0.0
            if j > 1
                cl = u[j - 1, i_level + 1]
            end
            cr = 0.0
            if j <= nc
                cr = u[j, i_level + 1]
            end
            u[jf, i_level] = u[jf, i_level] + 0.5 * (cl + cr)
        end
        for i_k_mg_relax_c2 = 1:nu2
            for i_j_mg_relax_c2 = 1:n
                left_mg_relax_c2 = 0.0
                if i_j_mg_relax_c2 > 1
                    left_mg_relax_c2 = u[i_j_mg_relax_c2 - 1, i_level]
                end
                right_mg_relax_c2 = 0.0
                if i_j_mg_relax_c2 < n
                    right_mg_relax_c2 = u[i_j_mg_relax_c2 + 1, i_level]
                end
                u[i_j_mg_relax_c2, i_level] = 0.5 * (hl2 * f[i_j_mg_relax_c2, i_level] + left_mg_relax_c2 + right_mg_relax_c2)
            end
        end
    end
end
