function initstacks_gat_inplace_seq_b()
    return nothing
end

function gat_inplace_seq_hv(u, ub, n, ud, ubd)
    for i = 2:n - 1
        ud[i] = 0.5 * ud[i] + 0.25 * (ud[i - 1] + ud[i + 1])
        u[i] = 0.5 * u[i] + 0.25 * (u[i - 1] + u[i + 1])
    end
    for i = n - 1:-1:2
        __oldb_0d = ubd[i]
        __oldb_0 = ub[i]
        ubd[i] = 0.0
        ub[i] = 0.0
        ubd[i] = ubd[i] + 0.5__oldb_0d
        ub[i] = ub[i] + 0.5__oldb_0
        __cse_0d = 0.25__oldb_0d
        __cse_0 = 0.25__oldb_0
        ubd[i - 1] = ubd[i - 1] + __cse_0d
        ub[i - 1] = ub[i - 1] + __cse_0
        ubd[i + 1] = ubd[i + 1] + __cse_0d
        ub[i + 1] = ub[i + 1] + __cse_0
    end
    return nothing
end

function gat_inplace_seq(u, n)
    for i = 2:n - 1
        u[i] = 0.5 * u[i] + 0.25 * (u[i - 1] + u[i + 1])
    end
end
