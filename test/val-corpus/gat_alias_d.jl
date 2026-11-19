function gat_alias_d(x, xd, y, yd, n)
    yd[1] = xd[1]
    y[1] = x[1]
    yd[n] = xd[n]
    y[n] = x[n]
    for i = 2:n - 1
        ipd = 0.0
        ip = i + 1
        imd = 0.0
        im = i - 1
        yd[i] = (xd[im] + xd[ip]) + -(2.0 * xd[i])
        y[i] = (x[im] + x[ip]) - 2.0 * x[i]
    end
    return nothing
end

function gat_alias(x, y, n)
    y[1] = x[1]
    y[n] = x[n]
    for i = 2:n - 1
        ip = i + 1
        im = i - 1
        y[i] = (x[im] + x[ip]) - 2.0 * x[i]
    end
end
