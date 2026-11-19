function gat_accum_out_d(x, xd, y, yd, n)
    for i = 2:n - 1
        yd[i] = yd[i] + 0.5 * (xd[i - 1] + xd[i + 1])
        y[i] = y[i] + 0.5 * (x[i - 1] + x[i + 1])
    end
    return nothing
end

function gat_accum_out(x, y, n)
    for i = 2:n - 1
        y[i] = y[i] + 0.5 * (x[i - 1] + x[i + 1])
    end
end
