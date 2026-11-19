function gat_stride_d(x, xd, y, yd, m)
    for i = 1:m
        yd[i] = 0.5 * xd[2i - 1] + 0.5 * xd[2i]
        y[i] = 0.5 * x[2i - 1] + 0.5 * x[2i]
    end
    return nothing
end

function gat_stride(x, y, m)
    for i = 1:m
        y[i] = 0.5 * x[2i - 1] + 0.5 * x[2i]
    end
end
