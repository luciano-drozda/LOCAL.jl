# remat_unused_stack(x0, z0, pa, out, i_n)
#
# A scalar multiplied by a top-level statement (`z = z * x`) while it still holds an unchanged input: the pre-write value is rematerialized from z0, so the
# push is gone and its stack is never created. The adjoint's signature must list only the stacks initstacks creates (found by the fixed-point fuzz).
#
# x0: initial state
# z0: scale input, read once
# pa: forcing, length >= i_n
# out: single-element output
# i_n: number of steps
function remat_unused_stack(x0, z0, pa, out, i_n)
    x = x0
    z = z0
    # the multiplication that overwrites z
    z = z * x
    # a state that changes between steps
    for t = 1:i_n
        x = 0.5 * cos(x) + 0.1 * pa[t]
    end
    out[1] = x * z + x
    return nothing
end
