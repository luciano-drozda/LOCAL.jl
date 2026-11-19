function initstacks_gat_alias_b()
    return nothing
end

function gat_alias_hv(x, xb, y, yb, n, xd, xbd, yd, ybd)
    yd[1] = xd[1]
    y[1] = x[1]
    yd[n] = xd[n]
    y[n] = x[n]
    for i = 2:n - 1
        ip = i + 1
        im = i - 1
        yd[i] = (xd[im] + xd[ip]) + -(2.0 * xd[i])
        y[i] = (x[im] + x[ip]) - 2.0 * x[i]
    end
    for i = n - 1:-1:2
        ip = i + 1
        im = i - 1
        __oldb_0d = ybd[i]
        __oldb_0 = yb[i]
        ybd[i] = 0.0
        yb[i] = 0.0
        xbd[im] = xbd[im] + __oldb_0d
        xb[im] = xb[im] + __oldb_0
        xbd[ip] = xbd[ip] + __oldb_0d
        xb[ip] = xb[ip] + __oldb_0
        xbd[i] = xbd[i] + 2.0 * -__oldb_0d
        xb[i] = xb[i] + 2.0 * -__oldb_0
    end
    __oldb_0d = ybd[n]
    __oldb_0 = yb[n]
    ybd[n] = 0.0
    yb[n] = 0.0
    xbd[n] = xbd[n] + __oldb_0d
    xb[n] = xb[n] + __oldb_0
    __oldb_0d = ybd[1]
    __oldb_0 = yb[1]
    ybd[1] = 0.0
    yb[1] = 0.0
    xbd[1] = xbd[1] + __oldb_0d
    xb[1] = xb[1] + __oldb_0
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
