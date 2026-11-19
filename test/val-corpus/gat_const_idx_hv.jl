function initstacks_gat_const_idx_b()
    return nothing
end

function gat_const_idx_hv(x, xb, y, yb, n, xd, xbd, yd, ybd)
    for i = 2:n - 1
        yd[i] = xd[i - 1] + xd[i + 1]
        y[i] = x[i - 1] + x[i + 1]
        yd[1] = yd[1] + xd[i]
        y[1] = y[1] + x[i]
    end
    for i = n - 1:-1:2
        xbd[i] = xbd[i] + ybd[1]
        xb[i] = xb[i] + yb[1]
        __oldb_0d = ybd[i]
        __oldb_0 = yb[i]
        ybd[i] = 0.0
        yb[i] = 0.0
        xbd[i - 1] = xbd[i - 1] + __oldb_0d
        xb[i - 1] = xb[i - 1] + __oldb_0
        xbd[i + 1] = xbd[i + 1] + __oldb_0d
        xb[i + 1] = xb[i + 1] + __oldb_0
    end
    return nothing
end

function gat_const_idx(x, y, n)
    for i = 2:n - 1
        y[i] = x[i - 1] + x[i + 1]
        y[1] = y[1] + x[i]
    end
end
