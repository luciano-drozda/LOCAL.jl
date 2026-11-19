function initstacks_gat_two_targets_b()
    return nothing
end

function gat_two_targets_b(a, ab, x, xb, y, yb, z, zb, n)
    y[1] = x[1]
    y[n] = x[n]
    z[1] = a[1]
    z[n] = a[n]
    for i = 2:n - 1
        y[i] = a[i - 1] * x[i]
        z[i] = x[i + 1] * a[i] + y[i]
    end
    for i = n - 1:-1:2
        __oldb_0 = zb[i]
        zb[i] = 0.0
        xb[i + 1] = xb[i + 1] + a[i] * __oldb_0
        ab[i] = ab[i] + x[i + 1] * __oldb_0
        yb[i] = yb[i] + __oldb_0
        __oldb_0 = yb[i]
        yb[i] = 0.0
        ab[i - 1] = ab[i - 1] + x[i] * __oldb_0
        xb[i] = xb[i] + a[i - 1] * __oldb_0
    end
    __oldb_0 = zb[n]
    zb[n] = 0.0
    ab[n] = ab[n] + __oldb_0
    __oldb_0 = zb[1]
    zb[1] = 0.0
    ab[1] = ab[1] + __oldb_0
    __oldb_0 = yb[n]
    yb[n] = 0.0
    xb[n] = xb[n] + __oldb_0
    __oldb_0 = yb[1]
    yb[1] = 0.0
    xb[1] = xb[1] + __oldb_0
    return nothing
end

function gat_two_targets(a, x, y, z, n)
    y[1] = x[1]
    y[n] = x[n]
    z[1] = a[1]
    z[n] = a[n]
    for i = 2:n - 1
        y[i] = a[i - 1] * x[i]
        z[i] = x[i + 1] * a[i] + y[i]
    end
end
