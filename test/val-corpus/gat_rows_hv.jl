function initstacks_gat_rows_b()
    return nothing
end

function gat_rows_hv(x, xb, y, yb, n, m, xd, xbd, yd, ybd)
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
    for i = n - 1:-1:2
        for j = m:-1:1
            __oldb_0d = ybd[(i - 1) * m + j]
            __oldb_0 = yb[(i - 1) * m + j]
            ybd[(i - 1) * m + j] = 0.0
            yb[(i - 1) * m + j] = 0.0
            xbd[(i - 2) * m + j] = xbd[(i - 2) * m + j] + __oldb_0d
            xb[(i - 2) * m + j] = xb[(i - 2) * m + j] + __oldb_0
            xbd[i * m + j] = xbd[i * m + j] + __oldb_0d
            xb[i * m + j] = xb[i * m + j] + __oldb_0
        end
    end
    for j = m:-1:1
        __oldb_0d = ybd[(n - 1) * m + j]
        __oldb_0 = yb[(n - 1) * m + j]
        ybd[(n - 1) * m + j] = 0.0
        yb[(n - 1) * m + j] = 0.0
        xbd[(n - 1) * m + j] = xbd[(n - 1) * m + j] + __oldb_0d
        xb[(n - 1) * m + j] = xb[(n - 1) * m + j] + __oldb_0
        __oldb_0d = ybd[j]
        __oldb_0 = yb[j]
        ybd[j] = 0.0
        yb[j] = 0.0
        xbd[j] = xbd[j] + __oldb_0d
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
