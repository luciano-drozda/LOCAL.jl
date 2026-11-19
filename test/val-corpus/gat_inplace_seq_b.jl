function initstacks_gat_inplace_seq_b()
    return nothing
end

function gat_inplace_seq_b(u, ub, n)
    for i = 2:n - 1
        u[i] = 0.5 * u[i] + 0.25 * (u[i - 1] + u[i + 1])
    end
    for i = n - 1:-1:2
        __oldb_0 = ub[i]
        ub[i] = 0.0
        ub[i] = ub[i] + 0.5__oldb_0
        __cse_0 = 0.25__oldb_0
        ub[i - 1] = ub[i - 1] + __cse_0
        ub[i + 1] = ub[i + 1] + __cse_0
    end
    return nothing
end

function gat_inplace_seq(u, n)
    for i = 2:n - 1
        u[i] = 0.5 * u[i] + 0.25 * (u[i - 1] + u[i + 1])
    end
end
