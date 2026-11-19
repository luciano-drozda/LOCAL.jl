function initstacks_gat_accum_out_b()
    return nothing
end

function gat_accum_out_hv(x, xb, y, yb, n, xd, xbd, yd, ybd)
    for i = 2:n - 1
        yd[i] = yd[i] + 0.5 * (xd[i - 1] + xd[i + 1])
        y[i] = y[i] + 0.5 * (x[i - 1] + x[i + 1])
    end
    for i = n - 1:-1:2
        __cse_0d = 0.5 * ybd[i]
        __cse_0 = 0.5 * yb[i]
        xbd[i - 1] = xbd[i - 1] + __cse_0d
        xb[i - 1] = xb[i - 1] + __cse_0
        xbd[i + 1] = xbd[i + 1] + __cse_0d
        xb[i + 1] = xb[i + 1] + __cse_0
    end
    return nothing
end

function gat_accum_out(x, y, n)
    for i = 2:n - 1
        y[i] = y[i] + 0.5 * (x[i - 1] + x[i + 1])
    end
end
