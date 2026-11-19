function gat_inplace_seq_d(u, ud, n)
    for i = 2:n - 1
        ud[i] = 0.5 * ud[i] + 0.25 * (ud[i - 1] + ud[i + 1])
        u[i] = 0.5 * u[i] + 0.25 * (u[i - 1] + u[i + 1])
    end
    return nothing
end

function gat_inplace_seq(u, n)
    for i = 2:n - 1
        u[i] = 0.5 * u[i] + 0.25 * (u[i - 1] + u[i + 1])
    end
end
