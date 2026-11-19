function gat_diff1_d(x, xd, y, yd, n)
    yd[1] = xd[1]
    y[1] = x[1]
    yd[n] = xd[n]
    y[n] = x[n]
    for i = 2:n - 1
        yd[i] = xd[i + 1] + -(xd[i])
        y[i] = x[i + 1] - x[i]
    end
    return nothing
end

function gat_diff1(x, y, n)
    y[1] = x[1]
    y[n] = x[n]
    for i = 2:n - 1
        y[i] = x[i + 1] - x[i]
    end
end
