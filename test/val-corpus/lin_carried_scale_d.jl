function lin_carried_scale_d(x, xd, y, yd, z, zd, n)
    sd = 0.0
    s = 1.0
    for i = 1:n
        yd[i] = 0.5sd
        y[i] = 0.5s
        if x[i] > 0.0
            sd = xd[i]
            s = x[i]
        else
            sd = 2.0 * xd[i]
            s = 2.0 * x[i]
        end
        __cse_0 = s * sd
        zd[i] = __cse_0 + __cse_0
        z[i] = s * s
    end
    return nothing
end

function lin_carried_scale(x, y, z, n)
    s = 1.0
    for i = 1:n
        y[i] = 0.5s
        if x[i] > 0.0
            s = x[i]
        else
            s = 2.0 * x[i]
        end
        z[i] = s * s
    end
end
