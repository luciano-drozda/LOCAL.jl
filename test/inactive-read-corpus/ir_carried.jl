# ir_carried(x, y, c0, n)
#
# An inactive scalar carried between the iterations of one loop and read next to an active array in the same iteration.
#
# x: active input of length n
# y: result of length n
# c0: start value of the carried scalar; inactive when only x is independent
# n: length
function ir_carried(x, y, c0, n)
    c = c0
    # the scalar is carried from one iteration to the next: a sequential loop
    for i = 1:n
        y[i] = c * x[i]
        c = 0.9 * c + 0.05
    end
    return nothing
end
