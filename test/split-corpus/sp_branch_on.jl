# sp_branch_on(x, y, n)
#
# The same shape as sp_branch_off, with a branch that is always taken for the usual draws: the carried scalar is assigned, and the backward sweep reads it.
#
# x: input array of length n
# y: input and output array of length n, updated in place
# n: number of elements in x and y
function sp_branch_on(x, y, n)
    if x[1] > -50.0
        w = 0.5 * n
        for i = 1:n
            y[i] = y[i] * w + x[i]
        end
    end
    return nothing
end
