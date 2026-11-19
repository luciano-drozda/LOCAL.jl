function initstacks_fixed_sweeps_b()
    return nothing
end

function fixed_sweeps_hv(y, yb, x, xb, i_n, yd, ybd, xd, xbd)
    for i_k = 1:3
        for i_i = 1:i_n
            yd[i_i] = 0.5 * yd[i_i] + xd[i_i]
            y[i_i] = 0.5 * y[i_i] + x[i_i]
        end
    end
    for i_k = 3:-1:1
        for i_i = i_n:-1:1
            __oldb_0d = ybd[i_i]
            __oldb_0 = yb[i_i]
            ybd[i_i] = 0.0
            yb[i_i] = 0.0
            ybd[i_i] = ybd[i_i] + 0.5__oldb_0d
            yb[i_i] = yb[i_i] + 0.5__oldb_0
            xbd[i_i] = xbd[i_i] + __oldb_0d
            xb[i_i] = xb[i_i] + __oldb_0
        end
    end
    return nothing
end

function fixed_sweeps(y, x, i_n)
    for i_k = 1:3
        for i_i = 1:i_n
            y[i_i] = 0.5 * y[i_i] + x[i_i]
        end
    end
end
