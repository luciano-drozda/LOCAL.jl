function initstacks_gat_nonlinear_b()
    return nothing
end

function gat_nonlinear_b(x, xb, y, yb, n)
    y[1] = x[1]
    y[n] = x[n]
    for i = 2:n - 1
        y[i] = sin(x[i - 1]) * x[i + 1] + exp(0.1 * x[i])
    end
    for i = n - 1:-1:2
        __oldb_0 = yb[i]
        yb[i] = 0.0
        __cse_0 = x[i - 1]
        xb[i - 1] = xb[i - 1] + cos(__cse_0) * (x[i + 1] * __oldb_0)
        xb[i + 1] = xb[i + 1] + sin(__cse_0) * __oldb_0
        xb[i] = xb[i] + 0.1 * (exp(0.1 * x[i]) * __oldb_0)
    end
    __oldb_0 = yb[n]
    yb[n] = 0.0
    xb[n] = xb[n] + __oldb_0
    __oldb_0 = yb[1]
    yb[1] = 0.0
    xb[1] = xb[1] + __oldb_0
    return nothing
end

function gat_nonlinear(x, y, n)
    y[1] = x[1]
    y[n] = x[n]
    for i = 2:n - 1
        y[i] = sin(x[i - 1]) * x[i + 1] + exp(0.1 * x[i])
    end
end
