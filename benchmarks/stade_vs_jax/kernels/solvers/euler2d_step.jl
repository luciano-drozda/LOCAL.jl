# euler2d_step(rho, mx, my, ener, rhon, mxn, myn, enern, n)
#
# One finite-volume step of the 2D compressible Euler equations (ideal gas, gamma = 1.4) on an
# n x n periodic grid: MUSCL reconstruction of the conserved variables with the minmod limiter,
# Rusanov flux, forward Euler in time. dt/dx = 0.1 is a literal.
# Written for GPU use, in gather form: each cell reads a 9-cell cross (two cells each way),
# rebuilds the limited slopes of its own and its four neighbors' cells, computes the four face
# fluxes and writes only itself. Slopes and face fluxes are recomputed per cell, which trades
# arithmetic for the absence of scatter-accumulate writes in the adjoint.
# The limiter is written with sign/min/abs, so the loop has no branch.
#
# rho: density, length n^2, x index fastest, values above 0.5
# mx: x momentum, length n^2
# my: y momentum, length n^2
# ener: total energy, length n^2, with pressure above 0.3
# rhon: density at the new time level, length n^2, filled in place
# mxn: x momentum at the new time level, length n^2, filled in place
# myn: y momentum at the new time level, length n^2, filled in place
# enern: total energy at the new time level, length n^2, filled in place
# n: number of cells per direction
function euler2d_step(rho, mx, my, ener, rhon, mxn, myn, enern, n)
    for idx = 1:n * n
        i = mod(idx - 1, n) + 1
        j = div(idx - 1, n) + 1
        # two cells each way, periodic
        iw = mod(i - 2, n) + 1
        iww = mod(i - 3, n) + 1
        ie = mod(i, n) + 1
        iee = mod(i + 1, n) + 1
        js = mod(j - 2, n) + 1
        jss = mod(j - 3, n) + 1
        jn = mod(j, n) + 1
        jnn = mod(j + 1, n) + 1
        kw = (j - 1) * n + iw
        kww = (j - 1) * n + iww
        ke = (j - 1) * n + ie
        kee = (j - 1) * n + iee
        ks = (js - 1) * n + i
        kss = (jss - 1) * n + i
        kn = (jn - 1) * n + i
        knn = (jnn - 1) * n + i
        r_c = rho[idx]
        mx_c = mx[idx]
        my_c = my[idx]
        E_c = ener[idx]
        r_w = rho[kw]
        mx_w = mx[kw]
        my_w = my[kw]
        E_w = ener[kw]
        r_ww = rho[kww]
        mx_ww = mx[kww]
        my_ww = my[kww]
        E_ww = ener[kww]
        r_e = rho[ke]
        mx_e = mx[ke]
        my_e = my[ke]
        E_e = ener[ke]
        r_ee = rho[kee]
        mx_ee = mx[kee]
        my_ee = my[kee]
        E_ee = ener[kee]
        r_s = rho[ks]
        mx_s = mx[ks]
        my_s = my[ks]
        E_s = ener[ks]
        r_ss = rho[kss]
        mx_ss = mx[kss]
        my_ss = my[kss]
        E_ss = ener[kss]
        r_n = rho[kn]
        mx_n = mx[kn]
        my_n = my[kn]
        E_n = ener[kn]
        r_nn = rho[knn]
        mx_nn = mx[knn]
        my_nn = my[knn]
        E_nn = ener[knn]
        # limited slopes along x (cells w, c, e) and along y (cells s, c, n)
        dx1_r = r_w - r_ww
        dx2_r = r_c - r_w
        dx3_r = r_e - r_c
        dx4_r = r_ee - r_e
        sxm_r = 0.5 * (sign(dx1_r) + sign(dx2_r)) * min(abs(dx1_r), abs(dx2_r))
        sxc_r = 0.5 * (sign(dx2_r) + sign(dx3_r)) * min(abs(dx2_r), abs(dx3_r))
        sxp_r = 0.5 * (sign(dx3_r) + sign(dx4_r)) * min(abs(dx3_r), abs(dx4_r))
        dx1_mx = mx_w - mx_ww
        dx2_mx = mx_c - mx_w
        dx3_mx = mx_e - mx_c
        dx4_mx = mx_ee - mx_e
        sxm_mx = 0.5 * (sign(dx1_mx) + sign(dx2_mx)) * min(abs(dx1_mx), abs(dx2_mx))
        sxc_mx = 0.5 * (sign(dx2_mx) + sign(dx3_mx)) * min(abs(dx2_mx), abs(dx3_mx))
        sxp_mx = 0.5 * (sign(dx3_mx) + sign(dx4_mx)) * min(abs(dx3_mx), abs(dx4_mx))
        dx1_my = my_w - my_ww
        dx2_my = my_c - my_w
        dx3_my = my_e - my_c
        dx4_my = my_ee - my_e
        sxm_my = 0.5 * (sign(dx1_my) + sign(dx2_my)) * min(abs(dx1_my), abs(dx2_my))
        sxc_my = 0.5 * (sign(dx2_my) + sign(dx3_my)) * min(abs(dx2_my), abs(dx3_my))
        sxp_my = 0.5 * (sign(dx3_my) + sign(dx4_my)) * min(abs(dx3_my), abs(dx4_my))
        dx1_E = E_w - E_ww
        dx2_E = E_c - E_w
        dx3_E = E_e - E_c
        dx4_E = E_ee - E_e
        sxm_E = 0.5 * (sign(dx1_E) + sign(dx2_E)) * min(abs(dx1_E), abs(dx2_E))
        sxc_E = 0.5 * (sign(dx2_E) + sign(dx3_E)) * min(abs(dx2_E), abs(dx3_E))
        sxp_E = 0.5 * (sign(dx3_E) + sign(dx4_E)) * min(abs(dx3_E), abs(dx4_E))
        dy1_r = r_s - r_ss
        dy2_r = r_c - r_s
        dy3_r = r_n - r_c
        dy4_r = r_nn - r_n
        sym_r = 0.5 * (sign(dy1_r) + sign(dy2_r)) * min(abs(dy1_r), abs(dy2_r))
        syc_r = 0.5 * (sign(dy2_r) + sign(dy3_r)) * min(abs(dy2_r), abs(dy3_r))
        syp_r = 0.5 * (sign(dy3_r) + sign(dy4_r)) * min(abs(dy3_r), abs(dy4_r))
        dy1_mx = mx_s - mx_ss
        dy2_mx = mx_c - mx_s
        dy3_mx = mx_n - mx_c
        dy4_mx = mx_nn - mx_n
        sym_mx = 0.5 * (sign(dy1_mx) + sign(dy2_mx)) * min(abs(dy1_mx), abs(dy2_mx))
        syc_mx = 0.5 * (sign(dy2_mx) + sign(dy3_mx)) * min(abs(dy2_mx), abs(dy3_mx))
        syp_mx = 0.5 * (sign(dy3_mx) + sign(dy4_mx)) * min(abs(dy3_mx), abs(dy4_mx))
        dy1_my = my_s - my_ss
        dy2_my = my_c - my_s
        dy3_my = my_n - my_c
        dy4_my = my_nn - my_n
        sym_my = 0.5 * (sign(dy1_my) + sign(dy2_my)) * min(abs(dy1_my), abs(dy2_my))
        syc_my = 0.5 * (sign(dy2_my) + sign(dy3_my)) * min(abs(dy2_my), abs(dy3_my))
        syp_my = 0.5 * (sign(dy3_my) + sign(dy4_my)) * min(abs(dy3_my), abs(dy4_my))
        dy1_E = E_s - E_ss
        dy2_E = E_c - E_s
        dy3_E = E_n - E_c
        dy4_E = E_nn - E_n
        sym_E = 0.5 * (sign(dy1_E) + sign(dy2_E)) * min(abs(dy1_E), abs(dy2_E))
        syc_E = 0.5 * (sign(dy2_E) + sign(dy3_E)) * min(abs(dy2_E), abs(dy3_E))
        syp_E = 0.5 * (sign(dy3_E) + sign(dy4_E)) * min(abs(dy3_E), abs(dy4_E))
        # face fw: reconstructed left and right states
        fw_rL = r_w + 0.5 * sxm_r
        fw_mxL = mx_w + 0.5 * sxm_mx
        fw_myL = my_w + 0.5 * sxm_my
        fw_EL = E_w + 0.5 * sxm_E
        fw_rR = r_c - 0.5 * sxc_r
        fw_mxR = mx_c - 0.5 * sxc_mx
        fw_myR = my_c - 0.5 * sxc_my
        fw_ER = E_c - 0.5 * sxc_E
        fw_uL = fw_mxL / fw_rL
        fw_pL = 0.4 * (fw_EL - 0.5 * (fw_mxL * fw_mxL + fw_myL * fw_myL) / fw_rL)
        fw_cL = sqrt(1.4 * fw_pL / fw_rL)
        fw_F1L = fw_mxL * fw_uL + fw_pL
        fw_F2L = fw_myL * fw_uL
        fw_F3L = (fw_EL + fw_pL) * fw_uL
        fw_uR = fw_mxR / fw_rR
        fw_pR = 0.4 * (fw_ER - 0.5 * (fw_mxR * fw_mxR + fw_myR * fw_myR) / fw_rR)
        fw_cR = sqrt(1.4 * fw_pR / fw_rR)
        fw_F1R = fw_mxR * fw_uR + fw_pR
        fw_F2R = fw_myR * fw_uR
        fw_F3R = (fw_ER + fw_pR) * fw_uR
        fw_a = max(abs(fw_uL) + fw_cL, abs(fw_uR) + fw_cR)
        fw_G0 = 0.5 * (fw_mxL + fw_mxR) - 0.5 * fw_a * (fw_rR - fw_rL)
        fw_G1 = 0.5 * (fw_F1L + fw_F1R) - 0.5 * fw_a * (fw_mxR - fw_mxL)
        fw_G2 = 0.5 * (fw_F2L + fw_F2R) - 0.5 * fw_a * (fw_myR - fw_myL)
        fw_G3 = 0.5 * (fw_F3L + fw_F3R) - 0.5 * fw_a * (fw_ER - fw_EL)
        # face fe: reconstructed left and right states
        fe_rL = r_c + 0.5 * sxc_r
        fe_mxL = mx_c + 0.5 * sxc_mx
        fe_myL = my_c + 0.5 * sxc_my
        fe_EL = E_c + 0.5 * sxc_E
        fe_rR = r_e - 0.5 * sxp_r
        fe_mxR = mx_e - 0.5 * sxp_mx
        fe_myR = my_e - 0.5 * sxp_my
        fe_ER = E_e - 0.5 * sxp_E
        fe_uL = fe_mxL / fe_rL
        fe_pL = 0.4 * (fe_EL - 0.5 * (fe_mxL * fe_mxL + fe_myL * fe_myL) / fe_rL)
        fe_cL = sqrt(1.4 * fe_pL / fe_rL)
        fe_F1L = fe_mxL * fe_uL + fe_pL
        fe_F2L = fe_myL * fe_uL
        fe_F3L = (fe_EL + fe_pL) * fe_uL
        fe_uR = fe_mxR / fe_rR
        fe_pR = 0.4 * (fe_ER - 0.5 * (fe_mxR * fe_mxR + fe_myR * fe_myR) / fe_rR)
        fe_cR = sqrt(1.4 * fe_pR / fe_rR)
        fe_F1R = fe_mxR * fe_uR + fe_pR
        fe_F2R = fe_myR * fe_uR
        fe_F3R = (fe_ER + fe_pR) * fe_uR
        fe_a = max(abs(fe_uL) + fe_cL, abs(fe_uR) + fe_cR)
        fe_G0 = 0.5 * (fe_mxL + fe_mxR) - 0.5 * fe_a * (fe_rR - fe_rL)
        fe_G1 = 0.5 * (fe_F1L + fe_F1R) - 0.5 * fe_a * (fe_mxR - fe_mxL)
        fe_G2 = 0.5 * (fe_F2L + fe_F2R) - 0.5 * fe_a * (fe_myR - fe_myL)
        fe_G3 = 0.5 * (fe_F3L + fe_F3R) - 0.5 * fe_a * (fe_ER - fe_EL)
        # face fs: reconstructed left and right states
        fs_rL = r_s + 0.5 * sym_r
        fs_mxL = mx_s + 0.5 * sym_mx
        fs_myL = my_s + 0.5 * sym_my
        fs_EL = E_s + 0.5 * sym_E
        fs_rR = r_c - 0.5 * syc_r
        fs_mxR = mx_c - 0.5 * syc_mx
        fs_myR = my_c - 0.5 * syc_my
        fs_ER = E_c - 0.5 * syc_E
        fs_uL = fs_myL / fs_rL
        fs_pL = 0.4 * (fs_EL - 0.5 * (fs_mxL * fs_mxL + fs_myL * fs_myL) / fs_rL)
        fs_cL = sqrt(1.4 * fs_pL / fs_rL)
        fs_F1L = fs_myL * fs_uL + fs_pL
        fs_F2L = fs_mxL * fs_uL
        fs_F3L = (fs_EL + fs_pL) * fs_uL
        fs_uR = fs_myR / fs_rR
        fs_pR = 0.4 * (fs_ER - 0.5 * (fs_mxR * fs_mxR + fs_myR * fs_myR) / fs_rR)
        fs_cR = sqrt(1.4 * fs_pR / fs_rR)
        fs_F1R = fs_myR * fs_uR + fs_pR
        fs_F2R = fs_mxR * fs_uR
        fs_F3R = (fs_ER + fs_pR) * fs_uR
        fs_a = max(abs(fs_uL) + fs_cL, abs(fs_uR) + fs_cR)
        fs_G0 = 0.5 * (fs_myL + fs_myR) - 0.5 * fs_a * (fs_rR - fs_rL)
        fs_G1 = 0.5 * (fs_F2L + fs_F2R) - 0.5 * fs_a * (fs_mxR - fs_mxL)
        fs_G2 = 0.5 * (fs_F1L + fs_F1R) - 0.5 * fs_a * (fs_myR - fs_myL)
        fs_G3 = 0.5 * (fs_F3L + fs_F3R) - 0.5 * fs_a * (fs_ER - fs_EL)
        # face fn: reconstructed left and right states
        fn_rL = r_c + 0.5 * syc_r
        fn_mxL = mx_c + 0.5 * syc_mx
        fn_myL = my_c + 0.5 * syc_my
        fn_EL = E_c + 0.5 * syc_E
        fn_rR = r_n - 0.5 * syp_r
        fn_mxR = mx_n - 0.5 * syp_mx
        fn_myR = my_n - 0.5 * syp_my
        fn_ER = E_n - 0.5 * syp_E
        fn_uL = fn_myL / fn_rL
        fn_pL = 0.4 * (fn_EL - 0.5 * (fn_mxL * fn_mxL + fn_myL * fn_myL) / fn_rL)
        fn_cL = sqrt(1.4 * fn_pL / fn_rL)
        fn_F1L = fn_myL * fn_uL + fn_pL
        fn_F2L = fn_mxL * fn_uL
        fn_F3L = (fn_EL + fn_pL) * fn_uL
        fn_uR = fn_myR / fn_rR
        fn_pR = 0.4 * (fn_ER - 0.5 * (fn_mxR * fn_mxR + fn_myR * fn_myR) / fn_rR)
        fn_cR = sqrt(1.4 * fn_pR / fn_rR)
        fn_F1R = fn_myR * fn_uR + fn_pR
        fn_F2R = fn_mxR * fn_uR
        fn_F3R = (fn_ER + fn_pR) * fn_uR
        fn_a = max(abs(fn_uL) + fn_cL, abs(fn_uR) + fn_cR)
        fn_G0 = 0.5 * (fn_myL + fn_myR) - 0.5 * fn_a * (fn_rR - fn_rL)
        fn_G1 = 0.5 * (fn_F2L + fn_F2R) - 0.5 * fn_a * (fn_mxR - fn_mxL)
        fn_G2 = 0.5 * (fn_F1L + fn_F1R) - 0.5 * fn_a * (fn_myR - fn_myL)
        fn_G3 = 0.5 * (fn_F3L + fn_F3R) - 0.5 * fn_a * (fn_ER - fn_EL)
        rhon[idx] = r_c - 0.1 * (fe_G0 - fw_G0 + fn_G0 - fs_G0)
        mxn[idx] = mx_c - 0.1 * (fe_G1 - fw_G1 + fn_G1 - fs_G1)
        myn[idx] = my_c - 0.1 * (fe_G2 - fw_G2 + fn_G2 - fs_G2)
        enern[idx] = E_c - 0.1 * (fe_G3 - fw_G3 + fn_G3 - fs_G3)
    end
    return nothing
end
