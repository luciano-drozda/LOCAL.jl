function gat_const_idx_d(x, xd, y, yd, n)
    for i = 2:n - 1
        yd[i] = xd[i - 1] + xd[i + 1]
        y[i] = x[i - 1] + x[i + 1]
        yd[1] = yd[1] + xd[i]
        y[1] = y[1] + x[i]
    end
    return nothing
end

function gat_const_idx(x, y, n)
    for i = 2:n - 1
        y[i] = x[i - 1] + x[i + 1]
        y[1] = y[1] + x[i]
    end
end
