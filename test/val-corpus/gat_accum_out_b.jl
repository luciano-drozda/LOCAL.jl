function initstacks_gat_accum_out_b()
    return nothing
end

function gat_accum_out_b(x, xb, y, yb, n)
    for i = 2:n - 1
        y[i] = y[i] + 0.5 * (x[i - 1] + x[i + 1])
    end
    for i = n - 1:-1:2
        __cse_0 = 0.5 * yb[i]
        xb[i - 1] = xb[i - 1] + __cse_0
        xb[i + 1] = xb[i + 1] + __cse_0
    end
    return nothing
end

function gat_accum_out(x, y, n)
    for i = 2:n - 1
        y[i] = y[i] + 0.5 * (x[i - 1] + x[i + 1])
    end
end
