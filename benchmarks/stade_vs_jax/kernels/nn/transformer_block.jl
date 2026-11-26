# transformer_block(x, gam1, bet1, wqkv, bqkv, wo, bo, gam2, bet2, w1, bf1, w2, bf2,
#                   mu1, rs1, xn1, qkv, sc, den, att, x2, mu2, rs2, xn2, hmid, y, tt, d, nh, dff)
#
# One pre-norm transformer encoder block for one sequence (batch size 1), no mask:
#   y = x2 + MLP(LN2(x2)),  x2 = x + Attention(LN1(x)) Wo
# Written for GPU use and for its adjoint:
#  1. Every matrix product is one flattened loop with one thread per output element. The weight
#     index is input-major, w[(c - 1) * nout + o], so a warp (adjacent o) reads adjacent addresses.
#  2. Bias, activation and residual are fused into the loop of the product. No copy loops.
#  3. Query, key and value share one weight matrix and one output array (qkv), so the three
#     products are one loop and the attention reads head slices from it.
#  4. Attention keeps the scores in an array (sc) and the softmax is split into three loops: exp,
#     row sums, weighted sum. Nothing is overwritten, so the adjoint needs no stack. The row-max
#     shift is left out: it does not change the result, and it would need a running maximum in a
#     scalar that the adjoint must store. The inputs must keep the scaled scores below about 30
#     in magnitude (layer-normalized input and weights of scale 1/sqrt(d) do).
#  5. Layer-norm statistics are computed once per token (mu, rs) and applied in a second loop.
# Layout: sequences are token-major, x[(t - 1) * d + c]; heads are contiguous slices of d.
#
# x: block input, length tt * d
# gam1: scale of the first layer norm, length d
# bet1: shift of the first layer norm, length d
# wqkv: query, key, value weights, length d * 3 * d, input-major, columns q then k then v
# bqkv: query, key, value bias, length 3 * d
# wo: output projection weights, length d * d, input-major
# bo: output projection bias, length d
# gam2: scale of the second layer norm, length d
# bet2: shift of the second layer norm, length d
# w1: first MLP weights, length d * dff, input-major
# bf1: first MLP bias, length dff
# w2: second MLP weights, length dff * d, input-major
# bf2: second MLP bias, length d
# mu1: scratch, mean of x per token, length tt, filled in place
# rs1: scratch, inverse standard deviation of x per token, length tt, filled in place
# xn1: scratch, first layer-norm output, length tt * d, filled in place
# qkv: scratch, queries, keys and values, length tt * 3 * d, filled in place
# sc: scratch, exp of the scaled scores, length nh * tt * tt, filled in place
# den: scratch, softmax denominators, length nh * tt, filled in place
# att: scratch, attention output before the projection, length tt * d, filled in place
# x2: scratch, residual stream after attention, length tt * d, filled in place
# mu2: scratch, mean of x2 per token, length tt, filled in place
# rs2: scratch, inverse standard deviation of x2 per token, length tt, filled in place
# xn2: scratch, second layer-norm output, length tt * d, filled in place
# hmid: scratch, MLP hidden activations, length tt * dff, filled in place
# y: block output, length tt * d, filled in place
# tt: number of tokens
# d: model width
# nh: number of heads, d must be divisible by nh
# dff: MLP hidden width
function transformer_block(x, gam1, bet1, wqkv, bqkv, wo, bo, gam2, bet2, w1, bf1, w2, bf2, mu1, rs1, xn1, qkv, sc, den, att, x2, mu2, rs2, xn2, hmid, y, tt, d, nh, dff)
    d3 = 3 * d
    dh = div(d, nh)
    scale = 1.0 / sqrt(dh * 1.0)

    # layer norm 1: statistics, one thread per token
    for t = 1:tt
        sm = 0.0
        # running sum over the features
        for c = 1:d
            sm += x[(t - 1) * d + c]
        end
        mean1 = sm / d
        sv = 0.0
        for c = 1:d
            sv += (x[(t - 1) * d + c] - mean1) * (x[(t - 1) * d + c] - mean1)
        end
        mu1[t] = mean1
        rs1[t] = 1.0 / sqrt(sv / d + 0.00001)
    end
    # layer norm 1: apply, one thread per element
    for idx = 1:tt * d
        c = mod(idx - 1, d) + 1
        t = div(idx - 1, d) + 1
        xn1[idx] = (x[idx] - mu1[t]) * rs1[t] * gam1[c] + bet1[c]
    end

    # query, key and value in one product, one thread per output element
    for idx = 1:tt * d3
        o = mod(idx - 1, d3) + 1
        t = div(idx - 1, d3) + 1
        sq = bqkv[o]
        for c = 1:d
            sq += xn1[(t - 1) * d + c] * wqkv[(c - 1) * d3 + o]
        end
        qkv[idx] = sq
    end

    # scores: one thread per (head, query token, key token), key token fastest
    for idx = 1:nh * tt * tt
        u = mod(idx - 1, tt) + 1
        t = mod(div(idx - 1, tt), tt) + 1
        h = div(idx - 1, tt * tt) + 1
        ss = 0.0
        # running sum over the head width
        for c = 1:dh
            ss += qkv[(t - 1) * d3 + (h - 1) * dh + c] * qkv[(u - 1) * d3 + d + (h - 1) * dh + c]
        end
        sc[idx] = exp(ss * scale)
    end
    # softmax denominators, one thread per (head, query token)
    for idx = 1:nh * tt
        sd = 0.0
        for u = 1:tt
            sd += sc[(idx - 1) * tt + u]
        end
        den[idx] = sd
    end
    # weighted sum of the values, one thread per (query token, feature)
    for idx = 1:tt * d
        c = mod(idx - 1, d) + 1
        t = div(idx - 1, d) + 1
        h = div(c - 1, dh) + 1
        sa = 0.0
        for u = 1:tt
            sa += sc[((h - 1) * tt + t - 1) * tt + u] * qkv[(u - 1) * d3 + 2 * d + c]
        end
        att[idx] = sa / den[(h - 1) * tt + t]
    end

    # output projection with bias and residual
    for idx = 1:tt * d
        o = mod(idx - 1, d) + 1
        t = div(idx - 1, d) + 1
        sp = bo[o] + x[idx]
        for c = 1:d
            sp += att[(t - 1) * d + c] * wo[(c - 1) * d + o]
        end
        x2[idx] = sp
    end

    # layer norm 2
    for t = 1:tt
        sm2 = 0.0
        for c = 1:d
            sm2 += x2[(t - 1) * d + c]
        end
        mean2 = sm2 / d
        sv2 = 0.0
        for c = 1:d
            sv2 += (x2[(t - 1) * d + c] - mean2) * (x2[(t - 1) * d + c] - mean2)
        end
        mu2[t] = mean2
        rs2[t] = 1.0 / sqrt(sv2 / d + 0.00001)
    end
    for idx = 1:tt * d
        c = mod(idx - 1, d) + 1
        t = div(idx - 1, d) + 1
        xn2[idx] = (x2[idx] - mu2[t]) * rs2[t] * gam2[c] + bet2[c]
    end

    # MLP: first product with gelu (tanh form), one thread per hidden element
    for idx = 1:tt * dff
        f = mod(idx - 1, dff) + 1
        t = div(idx - 1, dff) + 1
        sh = bf1[f]
        for c = 1:d
            sh += xn2[(t - 1) * d + c] * w1[(c - 1) * dff + f]
        end
        hmid[idx] = 0.5 * sh * (1.0 + tanh(0.7978845608028654 * (sh + 0.044715 * sh * sh * sh)))
    end
    # MLP: second product with bias and residual
    for idx = 1:tt * d
        o = mod(idx - 1, d) + 1
        t = div(idx - 1, d) + 1
        sy = bf2[o] + x2[idx]
        for f = 1:dff
            sy += hmid[(t - 1) * dff + f] * w2[(f - 1) * d + o]
        end
        y[idx] = sy
    end
    return nothing
end
