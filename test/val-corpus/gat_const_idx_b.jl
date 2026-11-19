function initstacks_gat_const_idx_b()
    return nothing
end

function gat_const_idx_b(x, xb, y, yb, n)
    for i = 2:n - 1
        y[i] = x[i - 1] + x[i + 1]
        y[1] = y[1] + x[i]
    end
    for i = n - 1:-1:2
        xb[i] = xb[i] + yb[1]
        __oldb_0 = yb[i]
        yb[i] = 0.0
        xb[i - 1] = xb[i - 1] + __oldb_0
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
