# mlp3(x, w1, b1, w2, b2, w3, b3, h1, h2, y, n0, n1, n2, n3)
#
# Forward pass of a 3-layer perceptron for one sample (batch size 1): tanh, tanh, linear.
# Pull form: one thread per output unit of a layer, the reduction over the inputs runs inside the
# thread. Weights are stored input-major, w[(i - 1) * nout + j], so the threads of a warp (which
# differ in j) read adjacent addresses. No copy, no concatenation, bias and activation are
# fused into the layer loop, and no loop writes an element it reads later.
#
# x: input vector, length n0
# w1: first weight matrix, length n0 * n1, input-major
# b1: first bias, length n1
# w2: second weight matrix, length n1 * n2, input-major
# b2: second bias, length n2
# w3: third weight matrix, length n2 * n3, input-major
# b3: third bias, length n3
# h1: activations of layer 1, length n1, filled in place
# h2: activations of layer 2, length n2, filled in place
# y: output vector, length n3, filled in place
# n0: input size
# n1: size of layer 1
# n2: size of layer 2
# n3: output size
function mlp3(x, w1, b1, w2, b2, w3, b3, h1, h2, y, n0, n1, n2, n3)
    # layer: h1 = tanh(w1^T x + b1); one thread per output unit
    for j = 1:n1
        s = b1[j]
        # running sum over the inputs; the weight index makes neighbor threads read neighbor addresses
        for i = 1:n0
            s += w1[(i - 1) * n1 + j] * x[i]
        end
        h1[j] = tanh(s)
    end
    # layer: h2 = tanh(w2^T h1 + b2); one thread per output unit
    for j = 1:n2
        s = b2[j]
        # running sum over the inputs; the weight index makes neighbor threads read neighbor addresses
        for i = 1:n1
            s += w2[(i - 1) * n2 + j] * h1[i]
        end
        h2[j] = tanh(s)
    end
    # layer: y = w3^T h2 + b3; one thread per output unit
    for j = 1:n3
        s = b3[j]
        # running sum over the inputs; the weight index makes neighbor threads read neighbor addresses
        for i = 1:n2
            s += w3[(i - 1) * n3 + j] * h2[i]
        end
        y[j] = s
    end
    return nothing
end
