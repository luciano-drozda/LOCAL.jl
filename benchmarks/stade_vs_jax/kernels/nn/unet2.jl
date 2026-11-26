# unet2(img, wts, bs, ea1, ea2, pl1, eb1, eb2, pl2, bn1, bn2, da1, da2, fa1, fa2, outp, q, c0, c, co)
#
# Forward pass of a small U-Net for one image (batch size 1), image side n = 4 * q.
# Encoder: two 3x3 convolutions at each of three resolutions (n, n/2, n/4) with channel counts
# c, 2c, 4c, average pooling between them. Decoder: nearest-neighbor upsampling, concatenation
# with the encoder feature of the same resolution, two 3x3 convolutions at n/2 and at n. A final
# 1x1 convolution gives co output channels. Relu after every 3x3 convolution, replicate padding.
# Written for GPU use and for its adjoint:
#  1. One flattened loop per convolution, one thread per output element, x fastest, so a warp
#     reads adjacent pixels of one channel (coalesced) and all its threads use the same weight.
#  2. Padding by clamping the index with min/max: no boundary branch, no padded copy.
#  3. Upsampling and concatenation are folded into the convolution that consumes them: the
#     convolution reads the coarse tensor at index div(y - 1, 2) + 1 and the skip tensor in
#     two input-channel ranges. No upsampled or concatenated array exists.
#  4. Bias and relu are fused into the convolution loop. Every layer writes its own array,
#     and no array is overwritten, so the adjoint needs no array snapshot.
#  5. All weights live in one array (wts) and all biases in one array (bs), at offsets computed
#     in the kernel from the channel counts. Weight layout: [out channel][in channel][ky][kx].
# Tensors are channel-major: a[((ch - 1) * side + y - 1) * side + x].
#
# img: input image, length c0 * n * n
# wts: all convolution weights, length 9 * (c0*c + c*c + c*2c + 2c*2c + 2c*4c + 4c*4c + 6c*2c + 2c*2c + 3c*c + c*c) + c*co
# bs: all biases, length 10 layers (c, c, 2c, 2c, 4c, 4c, 2c, 2c, c, c) plus co
# ea1: encoder 1 first convolution output, length c * n * n, filled in place
# ea2: encoder 1 second convolution output, length c * n * n, filled in place
# pl1: pooled encoder 1, length c * (n/2)^2, filled in place
# eb1: encoder 2 first convolution output, length 2c * (n/2)^2, filled in place
# eb2: encoder 2 second convolution output, length 2c * (n/2)^2, filled in place
# pl2: pooled encoder 2, length 2c * (n/4)^2, filled in place
# bn1: bottleneck first convolution output, length 4c * (n/4)^2, filled in place
# bn2: bottleneck second convolution output, length 4c * (n/4)^2, filled in place
# da1: decoder 2 first convolution output, length 2c * (n/2)^2, filled in place
# da2: decoder 2 second convolution output, length 2c * (n/2)^2, filled in place
# fa1: decoder 1 first convolution output, length c * n * n, filled in place
# fa2: decoder 1 second convolution output, length c * n * n, filled in place
# outp: network output, length co * n * n, filled in place
# q: quarter of the image side
# c0: input channels
# c: base channel count
# co: output channels
function unet2(img, wts, bs, ea1, ea2, pl1, eb1, eb2, pl2, bn1, bn2, da1, da2, fa1, fa2, outp, q, c0, c, co)
    n = 4 * q
    n2 = 2 * q
    n4 = q
    c2 = 2 * c
    c4 = 4 * c

    # offsets of each layer in wts and bs
    wo1 = 0
    bo1 = 0
    wo2 = wo1 + 9 * (c0) * c
    bo2 = bo1 + c
    wo3 = wo2 + 9 * (c) * c
    bo3 = bo2 + c
    wo4 = wo3 + 9 * (c) * c2
    bo4 = bo3 + c2
    wo5 = wo4 + 9 * (c2) * c2
    bo5 = bo4 + c2
    wo6 = wo5 + 9 * (c2) * c4
    bo6 = bo5 + c4
    wo7 = wo6 + 9 * (c4) * c4
    bo7 = bo6 + c4
    wo8 = wo7 + 9 * (c4 + c2) * c2
    bo8 = bo7 + c2
    wo9 = wo8 + 9 * (c2) * c2
    bo9 = bo8 + c2
    wo10 = wo9 + 9 * (c2 + c) * c
    bo10 = bo9 + c
    wo11 = wo10 + 9 * (c) * c
    bo11 = bo10 + c

    # L1: 3x3 convolution + bias + relu into ea1
    for idx1 = 1:c * n * n
        px1 = mod(idx1 - 1, n) + 1
        py1 = mod(div(idx1 - 1, n), n) + 1
        oc1 = div(idx1 - 1, n * n) + 1
        s1 = bs[bo1 + oc1]
        # input channels from img
        for ci10 = 1:c0
            for ky10 = 1:3
                for kx10 = 1:3
                    yy10 = min(max(py1 + ky10 - 2, 1), n)
                    xx10 = min(max(px1 + kx10 - 2, 1), n)
                    s1 += wts[wo1 + ((oc1 - 1) * (c0) + 0 + ci10 - 1) * 9 + (ky10 - 1) * 3 + kx10] * img[((ci10 - 1) * n + yy10 - 1) * n + xx10]
                end
            end
        end
        ea1[idx1] = max(s1, 0.0)
    end
    # L2: 3x3 convolution + bias + relu into ea2
    for idx2 = 1:c * n * n
        px2 = mod(idx2 - 1, n) + 1
        py2 = mod(div(idx2 - 1, n), n) + 1
        oc2 = div(idx2 - 1, n * n) + 1
        s2 = bs[bo2 + oc2]
        # input channels from ea1
        for ci20 = 1:c
            for ky20 = 1:3
                for kx20 = 1:3
                    yy20 = min(max(py2 + ky20 - 2, 1), n)
                    xx20 = min(max(px2 + kx20 - 2, 1), n)
                    s2 += wts[wo2 + ((oc2 - 1) * (c) + 0 + ci20 - 1) * 9 + (ky20 - 1) * 3 + kx20] * ea1[((ci20 - 1) * n + yy20 - 1) * n + xx20]
                end
            end
        end
        ea2[idx2] = max(s2, 0.0)
    end
    # pool 1: 2x2 average pooling ea2 -> pl1
    for idxp1 = 1:c * n2 * n2
        pxp1 = mod(idxp1 - 1, n2) + 1
        pyp1 = mod(div(idxp1 - 1, n2), n2) + 1
        ocp1 = div(idxp1 - 1, n2 * n2) + 1
        pl1[idxp1] = 0.25 * (ea2[((ocp1 - 1) * n + 2 * pyp1 - 2) * n + 2 * pxp1 - 1] + ea2[((ocp1 - 1) * n + 2 * pyp1 - 2) * n + 2 * pxp1] + ea2[((ocp1 - 1) * n + 2 * pyp1 - 1) * n + 2 * pxp1 - 1] + ea2[((ocp1 - 1) * n + 2 * pyp1 - 1) * n + 2 * pxp1])
    end
    # L3: 3x3 convolution + bias + relu into eb1
    for idx3 = 1:c2 * n2 * n2
        px3 = mod(idx3 - 1, n2) + 1
        py3 = mod(div(idx3 - 1, n2), n2) + 1
        oc3 = div(idx3 - 1, n2 * n2) + 1
        s3 = bs[bo3 + oc3]
        # input channels from pl1
        for ci30 = 1:c
            for ky30 = 1:3
                for kx30 = 1:3
                    yy30 = min(max(py3 + ky30 - 2, 1), n2)
                    xx30 = min(max(px3 + kx30 - 2, 1), n2)
                    s3 += wts[wo3 + ((oc3 - 1) * (c) + 0 + ci30 - 1) * 9 + (ky30 - 1) * 3 + kx30] * pl1[((ci30 - 1) * n2 + yy30 - 1) * n2 + xx30]
                end
            end
        end
        eb1[idx3] = max(s3, 0.0)
    end
    # L4: 3x3 convolution + bias + relu into eb2
    for idx4 = 1:c2 * n2 * n2
        px4 = mod(idx4 - 1, n2) + 1
        py4 = mod(div(idx4 - 1, n2), n2) + 1
        oc4 = div(idx4 - 1, n2 * n2) + 1
        s4 = bs[bo4 + oc4]
        # input channels from eb1
        for ci40 = 1:c2
            for ky40 = 1:3
                for kx40 = 1:3
                    yy40 = min(max(py4 + ky40 - 2, 1), n2)
                    xx40 = min(max(px4 + kx40 - 2, 1), n2)
                    s4 += wts[wo4 + ((oc4 - 1) * (c2) + 0 + ci40 - 1) * 9 + (ky40 - 1) * 3 + kx40] * eb1[((ci40 - 1) * n2 + yy40 - 1) * n2 + xx40]
                end
            end
        end
        eb2[idx4] = max(s4, 0.0)
    end
    # pool 2: 2x2 average pooling eb2 -> pl2
    for idxp2 = 1:c2 * n4 * n4
        pxp2 = mod(idxp2 - 1, n4) + 1
        pyp2 = mod(div(idxp2 - 1, n4), n4) + 1
        ocp2 = div(idxp2 - 1, n4 * n4) + 1
        pl2[idxp2] = 0.25 * (eb2[((ocp2 - 1) * n2 + 2 * pyp2 - 2) * n2 + 2 * pxp2 - 1] + eb2[((ocp2 - 1) * n2 + 2 * pyp2 - 2) * n2 + 2 * pxp2] + eb2[((ocp2 - 1) * n2 + 2 * pyp2 - 1) * n2 + 2 * pxp2 - 1] + eb2[((ocp2 - 1) * n2 + 2 * pyp2 - 1) * n2 + 2 * pxp2])
    end
    # L5: 3x3 convolution + bias + relu into bn1
    for idx5 = 1:c4 * n4 * n4
        px5 = mod(idx5 - 1, n4) + 1
        py5 = mod(div(idx5 - 1, n4), n4) + 1
        oc5 = div(idx5 - 1, n4 * n4) + 1
        s5 = bs[bo5 + oc5]
        # input channels from pl2
        for ci50 = 1:c2
            for ky50 = 1:3
                for kx50 = 1:3
                    yy50 = min(max(py5 + ky50 - 2, 1), n4)
                    xx50 = min(max(px5 + kx50 - 2, 1), n4)
                    s5 += wts[wo5 + ((oc5 - 1) * (c2) + 0 + ci50 - 1) * 9 + (ky50 - 1) * 3 + kx50] * pl2[((ci50 - 1) * n4 + yy50 - 1) * n4 + xx50]
                end
            end
        end
        bn1[idx5] = max(s5, 0.0)
    end
    # L6: 3x3 convolution + bias + relu into bn2
    for idx6 = 1:c4 * n4 * n4
        px6 = mod(idx6 - 1, n4) + 1
        py6 = mod(div(idx6 - 1, n4), n4) + 1
        oc6 = div(idx6 - 1, n4 * n4) + 1
        s6 = bs[bo6 + oc6]
        # input channels from bn1
        for ci60 = 1:c4
            for ky60 = 1:3
                for kx60 = 1:3
                    yy60 = min(max(py6 + ky60 - 2, 1), n4)
                    xx60 = min(max(px6 + kx60 - 2, 1), n4)
                    s6 += wts[wo6 + ((oc6 - 1) * (c4) + 0 + ci60 - 1) * 9 + (ky60 - 1) * 3 + kx60] * bn1[((ci60 - 1) * n4 + yy60 - 1) * n4 + xx60]
                end
            end
        end
        bn2[idx6] = max(s6, 0.0)
    end
    # L7: 3x3 convolution + bias + relu into da1
    for idx7 = 1:c2 * n2 * n2
        px7 = mod(idx7 - 1, n2) + 1
        py7 = mod(div(idx7 - 1, n2), n2) + 1
        oc7 = div(idx7 - 1, n2 * n2) + 1
        s7 = bs[bo7 + oc7]
        # input channels from bn2 (nearest-neighbor upsampled)
        for ci70 = 1:c4
            for ky70 = 1:3
                for kx70 = 1:3
                    yy70 = min(max(py7 + ky70 - 2, 1), n2)
                    xx70 = min(max(px7 + kx70 - 2, 1), n2)
                    ys70 = div(yy70 - 1, 2) + 1
                    xs70 = div(xx70 - 1, 2) + 1
                    s7 += wts[wo7 + ((oc7 - 1) * (c4 + c2) + 0 + ci70 - 1) * 9 + (ky70 - 1) * 3 + kx70] * bn2[((ci70 - 1) * n4 + ys70 - 1) * n4 + xs70]
                end
            end
        end
        # input channels from eb2
        for ci71 = 1:c2
            for ky71 = 1:3
                for kx71 = 1:3
                    yy71 = min(max(py7 + ky71 - 2, 1), n2)
                    xx71 = min(max(px7 + kx71 - 2, 1), n2)
                    s7 += wts[wo7 + ((oc7 - 1) * (c4 + c2) + c4 + ci71 - 1) * 9 + (ky71 - 1) * 3 + kx71] * eb2[((ci71 - 1) * n2 + yy71 - 1) * n2 + xx71]
                end
            end
        end
        da1[idx7] = max(s7, 0.0)
    end
    # L8: 3x3 convolution + bias + relu into da2
    for idx8 = 1:c2 * n2 * n2
        px8 = mod(idx8 - 1, n2) + 1
        py8 = mod(div(idx8 - 1, n2), n2) + 1
        oc8 = div(idx8 - 1, n2 * n2) + 1
        s8 = bs[bo8 + oc8]
        # input channels from da1
        for ci80 = 1:c2
            for ky80 = 1:3
                for kx80 = 1:3
                    yy80 = min(max(py8 + ky80 - 2, 1), n2)
                    xx80 = min(max(px8 + kx80 - 2, 1), n2)
                    s8 += wts[wo8 + ((oc8 - 1) * (c2) + 0 + ci80 - 1) * 9 + (ky80 - 1) * 3 + kx80] * da1[((ci80 - 1) * n2 + yy80 - 1) * n2 + xx80]
                end
            end
        end
        da2[idx8] = max(s8, 0.0)
    end
    # L9: 3x3 convolution + bias + relu into fa1
    for idx9 = 1:c * n * n
        px9 = mod(idx9 - 1, n) + 1
        py9 = mod(div(idx9 - 1, n), n) + 1
        oc9 = div(idx9 - 1, n * n) + 1
        s9 = bs[bo9 + oc9]
        # input channels from da2 (nearest-neighbor upsampled)
        for ci90 = 1:c2
            for ky90 = 1:3
                for kx90 = 1:3
                    yy90 = min(max(py9 + ky90 - 2, 1), n)
                    xx90 = min(max(px9 + kx90 - 2, 1), n)
                    ys90 = div(yy90 - 1, 2) + 1
                    xs90 = div(xx90 - 1, 2) + 1
                    s9 += wts[wo9 + ((oc9 - 1) * (c2 + c) + 0 + ci90 - 1) * 9 + (ky90 - 1) * 3 + kx90] * da2[((ci90 - 1) * n2 + ys90 - 1) * n2 + xs90]
                end
            end
        end
        # input channels from ea2
        for ci91 = 1:c
            for ky91 = 1:3
                for kx91 = 1:3
                    yy91 = min(max(py9 + ky91 - 2, 1), n)
                    xx91 = min(max(px9 + kx91 - 2, 1), n)
                    s9 += wts[wo9 + ((oc9 - 1) * (c2 + c) + c2 + ci91 - 1) * 9 + (ky91 - 1) * 3 + kx91] * ea2[((ci91 - 1) * n + yy91 - 1) * n + xx91]
                end
            end
        end
        fa1[idx9] = max(s9, 0.0)
    end
    # L10: 3x3 convolution + bias + relu into fa2
    for idx10 = 1:c * n * n
        px10 = mod(idx10 - 1, n) + 1
        py10 = mod(div(idx10 - 1, n), n) + 1
        oc10 = div(idx10 - 1, n * n) + 1
        s10 = bs[bo10 + oc10]
        # input channels from fa1
        for ci100 = 1:c
            for ky100 = 1:3
                for kx100 = 1:3
                    yy100 = min(max(py10 + ky100 - 2, 1), n)
                    xx100 = min(max(px10 + kx100 - 2, 1), n)
                    s10 += wts[wo10 + ((oc10 - 1) * (c) + 0 + ci100 - 1) * 9 + (ky100 - 1) * 3 + kx100] * fa1[((ci100 - 1) * n + yy100 - 1) * n + xx100]
                end
            end
        end
        fa2[idx10] = max(s10, 0.0)
    end
    # L11: 1x1 convolution into outp, no activation
    for idx11 = 1:co * n * n
        oc11 = div(idx11 - 1, n * n) + 1
        pix11 = mod(idx11 - 1, n * n) + 1
        s11 = bs[bo11 + oc11]
        for ci11 = 1:c
            s11 += wts[wo11 + (oc11 - 1) * c + ci11] * fa2[(ci11 - 1) * n * n + pix11]
        end
        outp[idx11] = s11
    end
    return nothing
end
