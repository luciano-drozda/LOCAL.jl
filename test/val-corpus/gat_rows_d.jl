function gat_rows_d(x, xd, y, yd, n, m)
    for j = 1:m
        yd[j] = xd[j]
        y[j] = x[j]
        yd[(n - 1) * m + j] = xd[(n - 1) * m + j]
        y[(n - 1) * m + j] = x[(n - 1) * m + j]
    end
    for i = 2:n - 1
        for j = 1:m
            yd[(i - 1) * m + j] = xd[(i - 2) * m + j] + xd[i * m + j]
            y[(i - 1) * m + j] = x[(i - 2) * m + j] + x[i * m + j]
        end
    end
    return nothing
end

function gat_rows(x, y, n, m)
    for j = 1:m
        y[j] = x[j]
        y[(n - 1) * m + j] = x[(n - 1) * m + j]
    end
    for i = 2:n - 1
        for j = 1:m
            y[(i - 1) * m + j] = x[(i - 2) * m + j] + x[i * m + j]
        end
    end
end
