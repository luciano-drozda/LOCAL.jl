# sp_carry(x, y, n)
#
# The norm of x scales x into y. The norm is computed once and is read again by the backward sweep, so a split must carry it from the forward half.
#
# x: input array of length n
# y: input and output array of length n, updated in place
# n: number of elements in x and y
function sp_carry(x, y, n)
    s = 0.0
    # sum of squares, accumulated over the whole array
    for i = 1:n
        s += x[i] * x[i]
    end
    c = sqrt(s)
    # scale: the partial of y[i] with respect to x[i] and y[i] holds c
    for i = 1:n
        y[i] = y[i] * c + x[i]
    end
    return nothing
end
