# mpnn_gather(h, ef, src, rowptr, wsrc, wdst, wedge, bmsg, wuh, wua, bupd, asrc, adst, agg, hout, nn, nf, nm)
#
# One message-passing layer for one graph (batch size 1), restructured for GPU use and for its adjoint.
# Message on edge (j -> v):  m = relu(Ws h_j + Wd h_v + We e + bm)   (4 edge features, a literal)
# Aggregation:               agg_v = sum of m over the edges that enter v
# Update:                    h'_v = relu(Wuh h_v + Wua agg_v + bu)
# Differences from the corpus mpnn, each one a primal-level choice:
#  1. Linearity hoisted out of the edge loop. Ws h_j and Wd h_v are computed once per node (asrc,
#     adst) and the edge loop only adds them. The edge loop does 4 multiply-adds per message
#     component, not 2*nf + 4.
#  2. Gather by receiver. Edges are stored sorted by receiver (CSR: rowptr), so one thread owns one
#     (node, message component) pair and sums its incoming messages in a register. No atomic write
#     in the primal, and the adjoint of the sender gather is the only scatter.
#  3. No concatenated input, no per-edge scratch, no copy loops. Bias and relu are fused.
#  4. The 4 edge features are a literal-length expression, not a loop, so the message needs no
#     reassigned scalar and the adjoint needs no stack.
# Weights are stored input-major, w[(i - 1) * nout + o], so neighbor threads read neighbor addresses.
#
# h: node features, length nn * nf, node-major
# ef: edge features, length 4 * (number of edges), edge-major, edges sorted by receiver
# src: sender node of each edge (1-based), same order as ef
# rowptr: first edge of each receiver, length nn + 1, rowptr[1] = 1, edges of node v are rowptr[v] .. rowptr[v+1] - 1
# wsrc: sender weights, length nf * nm
# wdst: receiver weights, length nf * nm
# wedge: edge weights, length 4 * nm
# bmsg: message bias, length nm
# wuh: update weights on the node feature, length nf * nf
# wua: update weights on the aggregate, length nm * nf
# bupd: update bias, length nf
# asrc: scratch, length nn * nm, filled in place
# adst: scratch, length nn * nm, filled in place
# agg: scratch, length nn * nm, filled in place
# hout: updated node features, length nn * nf, filled in place
# nn: number of nodes
# nf: node feature size
# nm: message size
function mpnn_gather(h, ef, src, rowptr, wsrc, wdst, wedge, bmsg, wuh, wua, bupd, asrc, adst, agg, hout, nn, nf, nm)
    # per-node projections of the node feature, one thread per (node, message component)
    for idx = 1:nn * nm
        k = mod(idx - 1, nm) + 1
        v = div(idx - 1, nm) + 1
        sa = 0.0
        sdst = 0.0
        # running sums over the node features
        for f = 1:nf
            hv = h[(v - 1) * nf + f]
            sa += hv * wsrc[(f - 1) * nm + k]
            sdst += hv * wdst[(f - 1) * nm + k]
        end
        asrc[idx] = sa
        adst[idx] = sdst
    end
    # aggregation: each thread sums the messages that enter its node, for one message component
    for idx = 1:nn * nm
        k = mod(idx - 1, nm) + 1
        v = div(idx - 1, nm) + 1
        lo = rowptr[v]
        hi = rowptr[v + 1] - 1
        s = 0.0
        # running sum over the incoming edges
        for e = lo:hi
            j = src[e]
            s += max(asrc[(j - 1) * nm + k] + adst[idx] + bmsg[k] + ef[(e - 1) * 4 + 1] * wedge[k] + ef[(e - 1) * 4 + 2] * wedge[nm + k] + ef[(e - 1) * 4 + 3] * wedge[2 * nm + k] + ef[(e - 1) * 4 + 4] * wedge[3 * nm + k], 0.0)
        end
        agg[idx] = s
    end
    # node update, one thread per (node, output feature)
    for idx = 1:nn * nf
        o = mod(idx - 1, nf) + 1
        v = div(idx - 1, nf) + 1
        su = bupd[o]
        for f = 1:nf
            su += h[(v - 1) * nf + f] * wuh[(f - 1) * nf + o]
        end
        for k = 1:nm
            su += agg[(v - 1) * nm + k] * wua[(k - 1) * nf + o]
        end
        hout[idx] = max(su, 0.0)
    end
    return nothing
end
