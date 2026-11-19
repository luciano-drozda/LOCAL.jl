function initstacks_cascadic_mg_prolong_b(h_coarse, nu, num_levels)
    __mult_2 = 0
    __mult_1 = 0
    nl = 2
    hl = h_coarse
    nc = nl - 1
    hl2 = hl * hl
    __mult_1 = max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1)
    __mult_2 = max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1)
    for i_k = 1:nu
        for i_j = 1:nc
            left = 0.0
            if i_j > 1
            end
            right = 0.0
            if i_j < nc
            end
        end
    end
    prefix_branch_stack_1 = Vector{Int}(undef, max(0, div(num_levels - 2, 1) + 1))
    __tot_branch_stack_1 = 0
    prefix_hl2_stack_1 = Vector{Int}(undef, max(0, div(num_levels - 2, 1) + 1))
    __tot_hl2_stack_1 = 0
    prefix_hl_stack_1 = Vector{Int}(undef, max(0, div(num_levels - 2, 1) + 1))
    __tot_hl_stack_1 = 0
    prefix_tripcount_stack_1 = Vector{Int}(undef, max(0, div(num_levels - 2, 1) + 1))
    __tot_tripcount_stack_1 = 0
    val_nc_1 = Vector{Int64}(undef, max(0, div(num_levels - 2, 1) + 1))
    val_ncoarse_1 = Vector{Int64}(undef, max(0, div(num_levels - 2, 1) + 1))
    for i_level = 2:num_levels
        prefix_branch_stack_1[(i_level - 2) + 1] = __tot_branch_stack_1
        prefix_hl2_stack_1[(i_level - 2) + 1] = __tot_hl2_stack_1
        prefix_hl_stack_1[(i_level - 2) + 1] = __tot_hl_stack_1
        prefix_tripcount_stack_1[(i_level - 2) + 1] = __tot_tripcount_stack_1
        nl = nl * 2
        hl = hl / 2.0
        nc = nl - 1
        hl2 = hl * hl
        ncoarse = div(nl, 2) - 1
        val_nc_1[(i_level - 2) + 1] = nc
        val_ncoarse_1[(i_level - 2) + 1] = ncoarse
        __tot_branch_stack_1 = __tot_branch_stack_1 + (((max(0, div((ncoarse + 1) - 1, 1) + 1) + max(0, div((ncoarse + 1) - 1, 1) + 1)) + max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1)) + max(0, div(nu - 1, 1) + 1) * max(0, div(nc - 1, 1) + 1))
        __tot_hl2_stack_1 = __tot_hl2_stack_1 + 1
        __tot_hl_stack_1 = __tot_hl_stack_1 + 1
        __tot_tripcount_stack_1 = __tot_tripcount_stack_1 + ((1 + 1) + max(0, div(nu - 1, 1) + 1))
    end
    tripcount_stack = Vector{Int64}(undef, max(0, div(nu - 1, 1) + 1) + __tot_tripcount_stack_1)
    branch_stack = Vector{Int64}(undef, (__mult_1 + __mult_2) + __tot_branch_stack_1)
    hl_stack = Vector{Float64}(undef, __tot_hl_stack_1 + 1)
    hl2_stack = Vector{Float64}(undef, __tot_hl2_stack_1 + 1)
    return (tripcount_stack, branch_stack, hl_stack, hl2_stack, prefix_branch_stack_1, prefix_hl2_stack_1, prefix_hl_stack_1, prefix_tripcount_stack_1, __mult_1, __mult_2, __tot_branch_stack_1, __tot_hl2_stack_1, __tot_hl_stack_1, __tot_tripcount_stack_1, val_nc_1, val_ncoarse_1)
end

function cascadic_mg_prolong_b(u, ub, rhs, rhsb, h_coarse, h_coarseb, nu, num_levels, tripcount_stack, branch_stack, hl_stack, hl2_stack, prefix_branch_stack_1, prefix_hl2_stack_1, prefix_hl_stack_1, prefix_tripcount_stack_1, __mult_1, __mult_2, __tot_branch_stack_1, __tot_hl2_stack_1, __tot_hl_stack_1, __tot_tripcount_stack_1, val_nc_1, val_ncoarse_1)
    cl = 0.0
    cr = 0.0
    hl = 0.0
    hl2 = 0.0
    left = 0.0
    right = 0.0
    clb = 0.0
    crb = 0.0
    hlb = 0.0
    hl2b = 0.0
    leftb = 0.0
    rightb = 0.0
    nl = 2
    hl = h_coarse
    nc = nl - 1
    hl2 = hl * hl
    for i_k = 1:nu
        __idx_tripcount_stack_0 = (i_k - 1) + 1
        tripcount_stack[__idx_tripcount_stack_0] = nc
        for i_j = 1:nc
            left = 0.0
            if i_j > 1
                __idx_branch_stack_0 = ((i_k - 1) * (div(nc - 1, 1) + 1) + (i_j - 1)) + 1
                branch_stack[__idx_branch_stack_0] = 1
                left = u[i_j - 1, 1]
            else
                __idx_branch_stack_0 = ((i_k - 1) * (div(nc - 1, 1) + 1) + (i_j - 1)) + 1
                branch_stack[__idx_branch_stack_0] = 0
            end
            right = 0.0
            if i_j < nc
                __idx_branch_stack_0 = __mult_1 + (((i_k - 1) * (div(nc - 1, 1) + 1) + (i_j - 1)) + 1)
                branch_stack[__idx_branch_stack_0] = 1
                right = u[i_j + 1, 1]
            else
                __idx_branch_stack_0 = __mult_1 + (((i_k - 1) * (div(nc - 1, 1) + 1) + (i_j - 1)) + 1)
                branch_stack[__idx_branch_stack_0] = 0
            end
            u[i_j, 1] = 0.5 * (hl2 * rhs[i_j, 1] + left + right)
        end
    end
    for i_level = 2:num_levels
        nl = nl * 2
        __idx_hl_stack_1_1 = prefix_hl_stack_1[(i_level - 2) + 1] + 1
        hl_stack[__idx_hl_stack_1_1] = hl
        hl = hl / 2.0
        nc = nl - 1
        __idx_hl2_stack_1_5 = prefix_hl2_stack_1[(i_level - 2) + 1] + 1
        hl2_stack[__idx_hl2_stack_1_5] = hl2
        hl2 = hl * hl
        ncoarse = div(nl, 2) - 1
        __idx_tripcount_stack_1_9 = (max(0, div(nu - 1, 1) + 1) + prefix_tripcount_stack_1[(i_level - 2) + 1]) + 1
        tripcount_stack[__idx_tripcount_stack_1_9] = ncoarse
        for j = 1:ncoarse
            jf = j * 2
            u[jf, i_level] = u[j, i_level - 1]
        end
        __idx_tripcount_stack_1_12 = ((max(0, div(nu - 1, 1) + 1) + prefix_tripcount_stack_1[(i_level - 2) + 1]) + 1) + 1
        tripcount_stack[__idx_tripcount_stack_1_12] = ncoarse
        for j = 1:ncoarse + 1
            jf = j * 2 - 1
            cl = 0.0
            if j > 1
                __idx_branch_stack_1_0 = ((__mult_1 + __mult_2) + prefix_branch_stack_1[(i_level - 2) + 1]) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 1
                cl = u[j - 1, i_level - 1]
            else
                __idx_branch_stack_1_0 = ((__mult_1 + __mult_2) + prefix_branch_stack_1[(i_level - 2) + 1]) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 0
            end
            cr = 0.0
            if j <= ncoarse
                __idx_branch_stack_1_0 = (((__mult_1 + __mult_2) + prefix_branch_stack_1[(i_level - 2) + 1]) + max(0, div((val_ncoarse_1[(i_level - 2) + 1] + 1) - 1, 1) + 1)) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 1
                cr = u[j, i_level - 1]
            else
                __idx_branch_stack_1_0 = (((__mult_1 + __mult_2) + prefix_branch_stack_1[(i_level - 2) + 1]) + max(0, div((val_ncoarse_1[(i_level - 2) + 1] + 1) - 1, 1) + 1)) + ((j - 1) + 1)
                branch_stack[__idx_branch_stack_1_0] = 0
            end
            u[jf, i_level] = 0.5 * (cl + cr)
        end
        for i_k = 1:nu
            __idx_tripcount_stack_1_0 = ((max(0, div(nu - 1, 1) + 1) + prefix_tripcount_stack_1[(i_level - 2) + 1]) + (1 + 1)) + ((i_k - 1) + 1)
            tripcount_stack[__idx_tripcount_stack_1_0] = nc
            for i_j = 1:nc
                left = 0.0
                if i_j > 1
                    __icse_0 = max(0, div((val_ncoarse_1[(i_level - 2) + 1] + 1) - 1, 1) + 1)
                    __idx_branch_stack_1_0 = (((__mult_1 + __mult_2) + prefix_branch_stack_1[(i_level - 2) + 1]) + (__icse_0 + __icse_0)) + (((i_k - 1) * (div(val_nc_1[(i_level - 2) + 1] - 1, 1) + 1) + (i_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_1_0] = 1
                    left = u[i_j - 1, i_level]
                else
                    __icse_1 = max(0, div((val_ncoarse_1[(i_level - 2) + 1] + 1) - 1, 1) + 1)
                    __idx_branch_stack_1_0 = (((__mult_1 + __mult_2) + prefix_branch_stack_1[(i_level - 2) + 1]) + (__icse_1 + __icse_1)) + (((i_k - 1) * (div(val_nc_1[(i_level - 2) + 1] - 1, 1) + 1) + (i_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_1_0] = 0
                end
                right = 0.0
                if i_j < nc
                    __icse_2 = max(0, div((val_ncoarse_1[(i_level - 2) + 1] + 1) - 1, 1) + 1)
                    __icse_3 = div(val_nc_1[(i_level - 2) + 1] - 1, 1) + 1
                    __idx_branch_stack_1_0 = (((__mult_1 + __mult_2) + prefix_branch_stack_1[(i_level - 2) + 1]) + ((__icse_2 + __icse_2) + max(0, div(nu - 1, 1) + 1) * max(0, __icse_3))) + (((i_k - 1) * __icse_3 + (i_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_1_0] = 1
                    right = u[i_j + 1, i_level]
                else
                    __icse_4 = max(0, div((val_ncoarse_1[(i_level - 2) + 1] + 1) - 1, 1) + 1)
                    __icse_5 = div(val_nc_1[(i_level - 2) + 1] - 1, 1) + 1
                    __idx_branch_stack_1_0 = (((__mult_1 + __mult_2) + prefix_branch_stack_1[(i_level - 2) + 1]) + ((__icse_4 + __icse_4) + max(0, div(nu - 1, 1) + 1) * max(0, __icse_5))) + (((i_k - 1) * __icse_5 + (i_j - 1)) + 1)
                    branch_stack[__idx_branch_stack_1_0] = 0
                end
                u[i_j, i_level] = 0.5 * (hl2 * rhs[i_j, i_level] + left + right)
            end
        end
    end
    __icse_6 = __tot_hl_stack_1 + 1
    __idx_hl_stack_6 = __icse_6
    hl_stack[__idx_hl_stack_6] = hl
    __icse_7 = __tot_hl2_stack_1 + 1
    __idx_hl2_stack_8 = __icse_7
    hl2_stack[__idx_hl2_stack_8] = hl2
    __idx_hl_stack_0 = __icse_6
    hl = hl_stack[__idx_hl_stack_0]
    __idx_hl2_stack_2 = __icse_7
    hl2 = hl2_stack[__idx_hl2_stack_2]
    for i_level = num_levels:-1:2
        for i_k = nu:-1:1
            __idx_tripcount_stack_1_0 = ((max(0, div(nu - 1, 1) + 1) + prefix_tripcount_stack_1[(i_level - 2) + 1]) + (1 + 1)) + ((i_k - 1) + 1)
            nc = tripcount_stack[__idx_tripcount_stack_1_0]
            for i_j = nc:-1:1
                __oldb_0 = ub[i_j, i_level]
                ub[i_j, i_level] = 0.0
                __cse_8 = 0.5__oldb_0
                hl2b = hl2b + rhs[i_j, i_level] * __cse_8
                rhsb[i_j, i_level] = rhsb[i_j, i_level] + hl2 * __cse_8
                leftb = leftb + __cse_8
                rightb = rightb + __cse_8
                __icse_9 = max(0, div((val_ncoarse_1[(i_level - 2) + 1] + 1) - 1, 1) + 1)
                __icse_10 = div(val_nc_1[(i_level - 2) + 1] - 1, 1) + 1
                __idx_branch_stack_1_6 = (((__mult_1 + __mult_2) + prefix_branch_stack_1[(i_level - 2) + 1]) + ((__icse_9 + __icse_9) + max(0, div(nu - 1, 1) + 1) * max(0, __icse_10))) + (((i_k - 1) * __icse_10 + (i_j - 1)) + 1)
                __branch = branch_stack[__idx_branch_stack_1_6]
                if __branch == 1
                    __oldb_0 = rightb
                    rightb = 0.0
                    ub[i_j + 1, i_level] = ub[i_j + 1, i_level] + __oldb_0
                end
                rightb = 0.0
                __icse_11 = max(0, div((val_ncoarse_1[(i_level - 2) + 1] + 1) - 1, 1) + 1)
                __idx_branch_stack_1_10 = (((__mult_1 + __mult_2) + prefix_branch_stack_1[(i_level - 2) + 1]) + (__icse_11 + __icse_11)) + (((i_k - 1) * (div(val_nc_1[(i_level - 2) + 1] - 1, 1) + 1) + (i_j - 1)) + 1)
                __branch = branch_stack[__idx_branch_stack_1_10]
                if __branch == 1
                    __oldb_0 = leftb
                    leftb = 0.0
                    ub[i_j - 1, i_level] = ub[i_j - 1, i_level] + __oldb_0
                end
                leftb = 0.0
            end
        end
        __idx_tripcount_stack_1_1 = ((max(0, div(nu - 1, 1) + 1) + prefix_tripcount_stack_1[(i_level - 2) + 1]) + 1) + 1
        ncoarse = tripcount_stack[__idx_tripcount_stack_1_1]
        for j = ncoarse + 1:-1:1
            jf = j * 2 - 1
            __oldb_0 = ub[jf, i_level]
            ub[jf, i_level] = 0.0
            __cse_12 = 0.5__oldb_0
            clb = clb + __cse_12
            crb = crb + __cse_12
            __idx_branch_stack_1_5 = (((__mult_1 + __mult_2) + prefix_branch_stack_1[(i_level - 2) + 1]) + max(0, div((val_ncoarse_1[(i_level - 2) + 1] + 1) - 1, 1) + 1)) + ((j - 1) + 1)
            __branch = branch_stack[__idx_branch_stack_1_5]
            if __branch == 1
                __oldb_0 = crb
                crb = 0.0
                ub[j, i_level - 1] = ub[j, i_level - 1] + __oldb_0
            end
            crb = 0.0
            __idx_branch_stack_1_9 = ((__mult_1 + __mult_2) + prefix_branch_stack_1[(i_level - 2) + 1]) + ((j - 1) + 1)
            __branch = branch_stack[__idx_branch_stack_1_9]
            if __branch == 1
                __oldb_0 = clb
                clb = 0.0
                ub[j - 1, i_level - 1] = ub[j - 1, i_level - 1] + __oldb_0
            end
            clb = 0.0
        end
        __idx_tripcount_stack_1_4 = (max(0, div(nu - 1, 1) + 1) + prefix_tripcount_stack_1[(i_level - 2) + 1]) + 1
        ncoarse = tripcount_stack[__idx_tripcount_stack_1_4]
        for j = ncoarse:-1:1
            jf = j * 2
            __oldb_0 = ub[jf, i_level]
            ub[jf, i_level] = 0.0
            ub[j, i_level - 1] = ub[j, i_level - 1] + __oldb_0
        end
        __idx_hl2_stack_1_0 = prefix_hl2_stack_1[(i_level - 2) + 1] + 1
        hl2 = hl2_stack[__idx_hl2_stack_1_0]
        __oldb_2 = hl2b
        hl2b = 0.0
        __cse_13 = hl * __oldb_2
        hlb = hlb + __cse_13
        hlb = hlb + __cse_13
        __idx_hl_stack_1_0 = prefix_hl_stack_1[(i_level - 2) + 1] + 1
        hl = hl_stack[__idx_hl_stack_1_0]
        __oldb_2 = hlb
        hlb = 0.0
        hlb = hlb + 0.5__oldb_2
    end
    for i_k = nu:-1:1
        __idx_tripcount_stack_0 = (i_k - 1) + 1
        nc = tripcount_stack[__idx_tripcount_stack_0]
        for i_j = nc:-1:1
            __oldb_0 = ub[i_j, 1]
            ub[i_j, 1] = 0.0
            __cse_14 = 0.5__oldb_0
            hl2b = hl2b + rhs[i_j, 1] * __cse_14
            rhsb[i_j, 1] = rhsb[i_j, 1] + hl2 * __cse_14
            leftb = leftb + __cse_14
            rightb = rightb + __cse_14
            __idx_branch_stack_6 = __mult_1 + (((i_k - 1) * (div(nc - 1, 1) + 1) + (i_j - 1)) + 1)
            __branch = branch_stack[__idx_branch_stack_6]
            if __branch == 1
                __oldb_0 = rightb
                rightb = 0.0
                ub[i_j + 1, 1] = ub[i_j + 1, 1] + __oldb_0
            end
            rightb = 0.0
            __idx_branch_stack_10 = ((i_k - 1) * (div(nc - 1, 1) + 1) + (i_j - 1)) + 1
            __branch = branch_stack[__idx_branch_stack_10]
            if __branch == 1
                __oldb_0 = leftb
                leftb = 0.0
                ub[i_j - 1, 1] = ub[i_j - 1, 1] + __oldb_0
            end
            leftb = 0.0
        end
    end
    __oldb_0 = hl2b
    hl2b = 0.0
    __cse_15 = hl * __oldb_0
    hlb = hlb + __cse_15
    hlb = hlb + __cse_15
    __oldb_0 = hlb
    hlb = 0.0
    h_coarseb = h_coarseb + __oldb_0
    return h_coarseb
end

function cascadic_mg_prolong(u, rhs, h_coarse, nu, num_levels)
    nl = 2
    hl = h_coarse
    nc = nl - 1
    hl2 = hl * hl
    for i_k = 1:nu
        for i_j = 1:nc
            left = 0.0
            if i_j > 1
                left = u[i_j - 1, 1]
            end
            right = 0.0
            if i_j < nc
                right = u[i_j + 1, 1]
            end
            u[i_j, 1] = 0.5 * (hl2 * rhs[i_j, 1] + left + right)
        end
    end
    for i_level = 2:num_levels
        nl = nl * 2
        hl = hl / 2.0
        nc = nl - 1
        hl2 = hl * hl
        ncoarse = div(nl, 2) - 1
        for j = 1:ncoarse
            jf = j * 2
            u[jf, i_level] = u[j, i_level - 1]
        end
        for j = 1:ncoarse + 1
            jf = j * 2 - 1
            cl = 0.0
            if j > 1
                cl = u[j - 1, i_level - 1]
            end
            cr = 0.0
            if j <= ncoarse
                cr = u[j, i_level - 1]
            end
            u[jf, i_level] = 0.5 * (cl + cr)
        end
        for i_k = 1:nu
            for i_j = 1:nc
                left = 0.0
                if i_j > 1
                    left = u[i_j - 1, i_level]
                end
                right = 0.0
                if i_j < nc
                    right = u[i_j + 1, i_level]
                end
                u[i_j, i_level] = 0.5 * (hl2 * rhs[i_j, i_level] + left + right)
            end
        end
    end
end
