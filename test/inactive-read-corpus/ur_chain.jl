# ur_chain(x, y, n)
#
# Two scalars defined before their use, so a single top-to-bottom scan does not see that the first one reaches the output.
#
# x: input of length n
# y: output of length n
# n: length
function ur_chain(x, y, n)
    # q reaches y only through p, and p is first used further down
    q = x[1] * x[1]
    p = q + 1.0
    for i = 1:n
        y[i] = p * x[i]
    end
    return nothing
end
