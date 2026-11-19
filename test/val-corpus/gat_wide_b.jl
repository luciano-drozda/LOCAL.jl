function initstacks_gat_wide_b()
    return nothing
end

function gat_wide_b(x, xb, w, wb, y, yb, n)
    y[1] = x[1]
    y[2] = x[2]
    y[n - 1] = x[n - 1]
    y[n] = x[n]
    for i = 3:n - 2
        y[i] = (w[i] * x[i - 2] + x[i - 1] * x[i + 1] + w[i + 1] * x[i + 2]) - 2.0 * x[i]
    end
    for i = n - 2:-1:3
        __oldb_0 = yb[i]
        yb[i] = 0.0
        wb[i] = wb[i] + x[i - 2] * __oldb_0
        xb[i - 2] = xb[i - 2] + w[i] * __oldb_0
        xb[i - 1] = xb[i - 1] + x[i + 1] * __oldb_0
        xb[i + 1] = xb[i + 1] + x[i - 1] * __oldb_0
        wb[i + 1] = wb[i + 1] + x[i + 2] * __oldb_0
        xb[i + 2] = xb[i + 2] + w[i + 1] * __oldb_0
        xb[i] = xb[i] + 2.0 * -__oldb_0
    end
    __oldb_0 = yb[n]
    yb[n] = 0.0
    xb[n] = xb[n] + __oldb_0
    __oldb_0 = yb[n - 1]
    yb[n - 1] = 0.0
    xb[n - 1] = xb[n - 1] + __oldb_0
    __oldb_0 = yb[2]
    yb[2] = 0.0
    xb[2] = xb[2] + __oldb_0
    __oldb_0 = yb[1]
    yb[1] = 0.0
    xb[1] = xb[1] + __oldb_0
    return nothing
end

function gat_wide(x, w, y, n)
    y[1] = x[1]
    y[2] = x[2]
    y[n - 1] = x[n - 1]
    y[n] = x[n]
    for i = 3:n - 2
        y[i] = (w[i] * x[i - 2] + x[i - 1] * x[i + 1] + w[i + 1] * x[i + 2]) - 2.0 * x[i]
    end
end
