# sp_branch_off(x, y, n)
#
# A scalar is assigned in the forward sweep only inside a branch that is never taken for the usual draws, and the backward sweep reads it under the same
# branch. A split must carry it, and its forward half must not fail on the run that skips the assignment.
#
# x: input array of length n
# y: input and output array of length n, updated in place
# n: number of elements in x and y
function sp_branch_off(x, y, n)
    if x[1] > 50.0
        w = 0.5 * n
        for i = 1:n
            y[i] = y[i] * w + x[i]
        end
    end
    return nothing
end
