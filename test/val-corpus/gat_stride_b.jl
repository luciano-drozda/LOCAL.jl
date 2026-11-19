function initstacks_gat_stride_b()
    return nothing
end

function gat_stride_b(x, xb, y, yb, m)
    for i = 1:m
        y[i] = 0.5 * x[2i - 1] + 0.5 * x[2i]
    end
    for i = m:-1:1
        __oldb_0 = yb[i]
        yb[i] = 0.0
        __cse_0 = 0.5__oldb_0
        xb[2i - 1] = xb[2i - 1] + __cse_0
        xb[2i] = xb[2i] + __cse_0
    end
    return nothing
end

function gat_stride(x, y, m)
    for i = 1:m
        y[i] = 0.5 * x[2i - 1] + 0.5 * x[2i]
    end
end
