function initstacks_gat_stride_b()
    return nothing
end

function gat_stride_hv(x, xb, y, yb, m, xd, xbd, yd, ybd)
    for i = 1:m
        yd[i] = 0.5 * xd[2i - 1] + 0.5 * xd[2i]
        y[i] = 0.5 * x[2i - 1] + 0.5 * x[2i]
    end
    for i = m:-1:1
        __oldb_0d = ybd[i]
        __oldb_0 = yb[i]
        ybd[i] = 0.0
        yb[i] = 0.0
        __cse_0d = 0.5__oldb_0d
        __cse_0 = 0.5__oldb_0
        xbd[2i - 1] = xbd[2i - 1] + __cse_0d
        xb[2i - 1] = xb[2i - 1] + __cse_0
        xbd[2i] = xbd[2i] + __cse_0d
        xb[2i] = xb[2i] + __cse_0
    end
    return nothing
end

function gat_stride(x, y, m)
    for i = 1:m
        y[i] = 0.5 * x[2i - 1] + 0.5 * x[2i]
    end
end
