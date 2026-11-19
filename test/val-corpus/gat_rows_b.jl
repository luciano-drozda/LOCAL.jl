function initstacks_gat_rows_b()
    return nothing
end

function gat_rows_b(x, xb, y, yb, n, m)
    for j = 1:m
        y[j] = x[j]
        y[(n - 1) * m + j] = x[(n - 1) * m + j]
    end
    for i = 2:n - 1
        for j = 1:m
            y[(i - 1) * m + j] = x[(i - 2) * m + j] + x[i * m + j]
        end
    end
    for i = n - 1:-1:2
        for j = m:-1:1
            __oldb_0 = yb[(i - 1) * m + j]
            yb[(i - 1) * m + j] = 0.0
            xb[(i - 2) * m + j] = xb[(i - 2) * m + j] + __oldb_0
            xb[i * m + j] = xb[i * m + j] + __oldb_0
        end
    end
    for j = m:-1:1
        __oldb_0 = yb[(n - 1) * m + j]
        yb[(n - 1) * m + j] = 0.0
        xb[(n - 1) * m + j] = xb[(n - 1) * m + j] + __oldb_0
        __oldb_0 = yb[j]
        yb[j] = 0.0
        xb[j] = xb[j] + __oldb_0
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
