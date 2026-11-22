# sp_restore(x, y, n)
#
# x is squared in place, then y reads the squared x. The backward sweep restores x, and it reads the squared x before it does, so a second
# backward sweep needs the squared x back.
#
# x: input and output array of length n, squared in place
# y: input and output array of length n, updated in place
# n: number of elements in x and y
function sp_restore(x, y, n)
    # x is overwritten here: the backward sweep puts the old value back
    for i = 1:n
        x[i] = x[i] * x[i]
    end
    # y reads the new x: the backward sweep needs the new x for this partial
    for i = 1:n
        y[i] = y[i] + x[i] * x[i]
    end
    return nothing
end
