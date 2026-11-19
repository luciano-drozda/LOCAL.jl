function initstacks_gat_two_targets_b()
    return nothing
end

function gat_two_targets_hv(a, ab, x, xb, y, yb, z, zb, n, ad, abd, xd, xbd, yd, ybd, zd, zbd)
    yd[1] = xd[1]
    y[1] = x[1]
    yd[n] = xd[n]
    y[n] = x[n]
    zd[1] = ad[1]
    z[1] = a[1]
    zd[n] = ad[n]
    z[n] = a[n]
    for i = 2:n - 1
        __hcse_0 = x[i]
        __hcse_1 = a[i - 1]
        yd[i] = __hcse_0 * ad[i - 1] + __hcse_1 * xd[i]
        y[i] = __hcse_1 * __hcse_0
        __hcse_2 = a[i]
        __hcse_3 = x[i + 1]
        zd[i] = (__hcse_2 * xd[i + 1] + __hcse_3 * ad[i]) + yd[i]
        z[i] = __hcse_3 * __hcse_2 + y[i]
    end
    for i = n - 1:-1:2
        __oldb_0d = zbd[i]
        __oldb_0 = zb[i]
        zbd[i] = 0.0
        zb[i] = 0.0
        __hcse_4 = a[i]
        xbd[i + 1] = xbd[i + 1] + (__oldb_0 * ad[i] + __hcse_4 * __oldb_0d)
        xb[i + 1] = xb[i + 1] + __hcse_4 * __oldb_0
        __hcse_5 = x[i + 1]
        abd[i] = abd[i] + (__oldb_0 * xd[i + 1] + __hcse_5 * __oldb_0d)
        ab[i] = ab[i] + __hcse_5 * __oldb_0
        ybd[i] = ybd[i] + __oldb_0d
        yb[i] = yb[i] + __oldb_0
        __oldb_0d = ybd[i]
        __oldb_0 = yb[i]
        ybd[i] = 0.0
        yb[i] = 0.0
        __hcse_6 = x[i]
        abd[i - 1] = abd[i - 1] + (__oldb_0 * xd[i] + __hcse_6 * __oldb_0d)
        ab[i - 1] = ab[i - 1] + __hcse_6 * __oldb_0
        __hcse_7 = a[i - 1]
        xbd[i] = xbd[i] + (__oldb_0 * ad[i - 1] + __hcse_7 * __oldb_0d)
        xb[i] = xb[i] + __hcse_7 * __oldb_0
    end
    __oldb_0d = zbd[n]
    __oldb_0 = zb[n]
    zbd[n] = 0.0
    zb[n] = 0.0
    abd[n] = abd[n] + __oldb_0d
    ab[n] = ab[n] + __oldb_0
    __oldb_0d = zbd[1]
    __oldb_0 = zb[1]
    zbd[1] = 0.0
    zb[1] = 0.0
    abd[1] = abd[1] + __oldb_0d
    ab[1] = ab[1] + __oldb_0
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
