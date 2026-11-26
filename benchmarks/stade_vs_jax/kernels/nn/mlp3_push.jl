# mlp3_push(x, w1, b1, w2, b2, w3, b3, z1, a1, z2, a2, y, n0, n1, n2, n3)
#
# The same network as mlp3 in push form: one thread per weight, each adds its product into the
# pre-activation of its output unit. Weights are stored input-major, so the weight index is the
# thread index and every weight read is coalesced. The primal pays n_in-way atomic contention on
# every output unit. The adjoint pays none for the input gradient (it becomes a gather) and
# writes every weight gradient once. It exists to measure which form is better for the adjoint.
#
# x: input vector, length n0
# w1: first weight matrix, length n0 * n1, input-major
# b1: first bias, length n1
# w2: second weight matrix, length n1 * n2, input-major
# b2: second bias, length n2
# w3: third weight matrix, length n2 * n3, input-major
# b3: third bias, length n3
# z1: pre-activation of layer 1, length n1, filled in place
# a1: activation of layer 1, length n1, filled in place
# z2: pre-activation of layer 2, length n2, filled in place
# a2: activation of layer 2, length n2, filled in place
# y: output vector, length n3, filled in place
# n0: input size
# n1: size of layer 1
# n2: size of layer 2
# n3: output size
function mlp3_push(x, w1, b1, w2, b2, w3, b3, z1, a1, z2, a2, y, n0, n1, n2, n3)
    # layer: z1 = w1^T x + b1
    for j = 1:n1
        z1[j] = b1[j]
    end
    # outer-product form: one thread per weight, the products are accumulated into z1[j]
    for idx = 1:n0 * n1
        i = div(idx - 1, n1) + 1
        j = mod(idx - 1, n1) + 1
        z1[j] = z1[j] + w1[idx] * x[i]
    end
    for j = 1:n1
        a1[j] = tanh(z1[j])
    end
    # layer: z2 = w2^T a1 + b2
    for j = 1:n2
        z2[j] = b2[j]
    end
    # outer-product form: one thread per weight, the products are accumulated into z2[j]
    for idx = 1:n1 * n2
        i = div(idx - 1, n2) + 1
        j = mod(idx - 1, n2) + 1
        z2[j] = z2[j] + w2[idx] * a1[i]
    end
    for j = 1:n2
        a2[j] = tanh(z2[j])
    end
    # layer: y = w3^T a2 + b3
    for j = 1:n3
        y[j] = b3[j]
    end
    # outer-product form: one thread per weight, the products are accumulated into y[j]
    for idx = 1:n2 * n3
        i = div(idx - 1, n3) + 1
        j = mod(idx - 1, n3) + 1
        y[j] = y[j] + w3[idx] * a2[i]
    end
    return nothing
end
