# swe2d_step(h, hu, hv, hn, hun, hvn, n)
#
# One finite-volume step of the 2D shallow-water equations (flat bed, g = 9.81) on an n x n
# periodic grid. HLL flux with Davis wave speeds, first order. dt/dx = 0.1 is a literal.
# Written for GPU use, in gather form: each cell recomputes the flux at its four faces from the
# states on both sides, so each cell writes only itself. A face flux is therefore computed twice
# (once per neighbor), which costs arithmetic but avoids scatter-accumulate writes in the adjoint.
# The HLL regions are selected by clamping the wave speeds, not by branches.
#
# h: water depth, length n^2, x index fastest, values above 0.5
# hu: x momentum, length n^2
# hv: y momentum, length n^2
# hn: depth at the new time level, length n^2, filled in place
# hun: x momentum at the new time level, length n^2, filled in place
# hvn: y momentum at the new time level, length n^2, filled in place
# n: number of cells per direction
function swe2d_step(h, hu, hv, hn, hun, hvn, n)
    for idx = 1:n * n
        i = mod(idx - 1, n) + 1
        j = div(idx - 1, n) + 1
        im = mod(i - 2, n) + 1
        ip = mod(i, n) + 1
        jm = mod(j - 2, n) + 1
        jp = mod(j, n) + 1
        kw = (j - 1) * n + im
        ke = (j - 1) * n + ip
        ks = (jm - 1) * n + i
        kn = (jp - 1) * n + i
        h_c = h[idx]
        hu_c = hu[idx]
        hv_c = hv[idx]
        h_w = h[kw]
        hu_w = hu[kw]
        hv_w = hv[kw]
        h_e = h[ke]
        hu_e = hu[ke]
        hv_e = hv[ke]
        h_s = h[ks]
        hu_s = hu[ks]
        hv_s = hv[ks]
        h_n = h[kn]
        hu_n = hu[kn]
        hv_n = hv[kn]
        # face fw: x direction, left state w, right state c
        fw_unL = hu_w / h_w
        fw_unR = hu_c / h_c
        fw_cL = sqrt(9.81 * h_w)
        fw_cR = sqrt(9.81 * h_c)
        fw_sL = min(min(fw_unL - fw_cL, fw_unR - fw_cR), 0.0)
        fw_sR = max(max(fw_unL + fw_cL, fw_unR + fw_cR), 0.0)
        fw_iw = 1.0 / (fw_sR - fw_sL)
        fw_f1L = hu_w * fw_unL + 4.905 * h_w * h_w
        fw_f1R = hu_c * fw_unR + 4.905 * h_c * h_c
        fw_f2L = hv_w * fw_unL
        fw_f2R = hv_c * fw_unR
        fw_sLR = fw_sL * fw_sR
        fw_F0 = (fw_sR * hu_w - fw_sL * hu_c + fw_sLR * (h_c - h_w)) * fw_iw
        fw_F1 = (fw_sR * fw_f1L - fw_sL * fw_f1R + fw_sLR * (hu_c - hu_w)) * fw_iw
        fw_F2 = (fw_sR * fw_f2L - fw_sL * fw_f2R + fw_sLR * (hv_c - hv_w)) * fw_iw
        # face fe: x direction, left state c, right state e
        fe_unL = hu_c / h_c
        fe_unR = hu_e / h_e
        fe_cL = sqrt(9.81 * h_c)
        fe_cR = sqrt(9.81 * h_e)
        fe_sL = min(min(fe_unL - fe_cL, fe_unR - fe_cR), 0.0)
        fe_sR = max(max(fe_unL + fe_cL, fe_unR + fe_cR), 0.0)
        fe_iw = 1.0 / (fe_sR - fe_sL)
        fe_f1L = hu_c * fe_unL + 4.905 * h_c * h_c
        fe_f1R = hu_e * fe_unR + 4.905 * h_e * h_e
        fe_f2L = hv_c * fe_unL
        fe_f2R = hv_e * fe_unR
        fe_sLR = fe_sL * fe_sR
        fe_F0 = (fe_sR * hu_c - fe_sL * hu_e + fe_sLR * (h_e - h_c)) * fe_iw
        fe_F1 = (fe_sR * fe_f1L - fe_sL * fe_f1R + fe_sLR * (hu_e - hu_c)) * fe_iw
        fe_F2 = (fe_sR * fe_f2L - fe_sL * fe_f2R + fe_sLR * (hv_e - hv_c)) * fe_iw
        # face fs: y direction, left state s, right state c
        fs_unL = hv_s / h_s
        fs_unR = hv_c / h_c
        fs_cL = sqrt(9.81 * h_s)
        fs_cR = sqrt(9.81 * h_c)
        fs_sL = min(min(fs_unL - fs_cL, fs_unR - fs_cR), 0.0)
        fs_sR = max(max(fs_unL + fs_cL, fs_unR + fs_cR), 0.0)
        fs_iw = 1.0 / (fs_sR - fs_sL)
        fs_f1L = hv_s * fs_unL + 4.905 * h_s * h_s
        fs_f1R = hv_c * fs_unR + 4.905 * h_c * h_c
        fs_f2L = hu_s * fs_unL
        fs_f2R = hu_c * fs_unR
        fs_sLR = fs_sL * fs_sR
        fs_F0 = (fs_sR * hv_s - fs_sL * hv_c + fs_sLR * (h_c - h_s)) * fs_iw
        fs_F1 = (fs_sR * fs_f1L - fs_sL * fs_f1R + fs_sLR * (hv_c - hv_s)) * fs_iw
        fs_F2 = (fs_sR * fs_f2L - fs_sL * fs_f2R + fs_sLR * (hu_c - hu_s)) * fs_iw
        # face fn: y direction, left state c, right state n
        fn_unL = hv_c / h_c
        fn_unR = hv_n / h_n
        fn_cL = sqrt(9.81 * h_c)
        fn_cR = sqrt(9.81 * h_n)
        fn_sL = min(min(fn_unL - fn_cL, fn_unR - fn_cR), 0.0)
        fn_sR = max(max(fn_unL + fn_cL, fn_unR + fn_cR), 0.0)
        fn_iw = 1.0 / (fn_sR - fn_sL)
        fn_f1L = hv_c * fn_unL + 4.905 * h_c * h_c
        fn_f1R = hv_n * fn_unR + 4.905 * h_n * h_n
        fn_f2L = hu_c * fn_unL
        fn_f2R = hu_n * fn_unR
        fn_sLR = fn_sL * fn_sR
        fn_F0 = (fn_sR * hv_c - fn_sL * hv_n + fn_sLR * (h_n - h_c)) * fn_iw
        fn_F1 = (fn_sR * fn_f1L - fn_sL * fn_f1R + fn_sLR * (hv_n - hv_c)) * fn_iw
        fn_F2 = (fn_sR * fn_f2L - fn_sL * fn_f2R + fn_sLR * (hu_n - hu_c)) * fn_iw
        hn[idx] = h_c - 0.1 * (fe_F0 - fw_F0 + fn_F0 - fs_F0)
        hun[idx] = hu_c - 0.1 * (fe_F1 - fw_F1 + fn_F2 - fs_F2)
        hvn[idx] = hv_c - 0.1 * (fe_F2 - fw_F2 + fn_F1 - fs_F1)
    end
    return nothing
end
