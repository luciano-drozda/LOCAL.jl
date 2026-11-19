function initstacks_gat_alias_b()
    return nothing
end

function gat_alias_b(x, xb, y, yb, n)
    y[1] = x[1]
    y[n] = x[n]
    for i = 2:n - 1
        ip = i + 1
        im = i - 1
        y[i] = (x[im] + x[ip]) - 2.0 * x[i]
    end
    for i = n - 1:-1:2
        ip = i + 1
        im = i - 1
        __oldb_0 = yb[i]
        yb[i] = 0.0
        xb[im] = xb[im] + __oldb_0
        xb[ip] = xb[ip] + __oldb_0
        xb[i] = xb[i] + 2.0 * -__oldb_0
    end
    __oldb_0 = yb[n]
    yb[n] = 0.0
    xb[n] = xb[n] + __oldb_0
    __oldb_0 = yb[1]
    yb[1] = 0.0
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
