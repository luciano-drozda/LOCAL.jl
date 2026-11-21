# ir_mg2d_solve_3
# Multigrid V-cycles (3 levels) for -Laplace(u) = f, at most maxit cycles, each run only while the squared fine-grid residual is above tol (a counted loop with a guard). Same cycle as ir_mg2d_cycle_3.
# Array of level l has extent n_l + 2 in both indices: interior 2..n_l+1, the ghost layer stays zero (zero boundary value).
# u1: fine-level solution, updated in place; f1: fine-level right-hand side; r1: fine-level residual (scratch)
# u2..u3, f2..f3, r2..r2: coarse-level solutions, right-hand sides and residuals (scratch, overwritten every cycle)
# n1: fine interior points per side, must be 2^3 - 1 so the coarsest level has one point
# h1: fine mesh width
# w: damping of the Jacobi smoother (0.8 is the usual value in 2D)
# nu1: pre-smoothing sweeps; nu2: post-smoothing sweeps
# tol: stop when the squared fine residual is at or below tol; maxit: cap on the number of cycles
function ir_mg2d_solve_3(u1, f1, r1, u2, f2, r2, u3, f3, n1, h1, w, nu1, nu2, tol, maxit)
    # sizes and scalars of every level, each assigned once before all loops
    n2 = div(n1 - 1, 2)
    n3 = div(n2 - 1, 2)
    hh1 = h1 * h1
    hh2 = 4.0 * hh1
    hh3 = 4.0 * hh2
    wq1 = 0.25 * w * hh1
    wq2 = 0.25 * w * hh2
    wq3 = 0.25 * w * hh3
    rn = 1.0
    for kk = 1:maxit
        if rn > tol
            # level 1: pre-smooth, residual, restrict
            # damped Jacobi on level 1: residual into r1, then one independent update of every point
            for sk1 = 1:nu1
                for k2 = 1:n1 * n1
                    i2 = div(k2 - 1, n1) + 2
                    j2 = mod(k2 - 1, n1) + 2
                    r1[i2, j2] = f1[i2, j2] - (4.0 * u1[i2, j2] - u1[i2 - 1, j2] - u1[i2 + 1, j2] - u1[i2, j2 - 1] - u1[i2, j2 + 1]) / hh1
                end
                for k3 = 1:n1 * n1
                    i3 = div(k3 - 1, n1) + 2
                    j3 = mod(k3 - 1, n1) + 2
                    u1[i3, j3] = u1[i3, j3] + wq1 * r1[i3, j3]
                end
            end
            for k4 = 1:n1 * n1
                i4 = div(k4 - 1, n1) + 2
                j4 = mod(k4 - 1, n1) + 2
                r1[i4, j4] = f1[i4, j4] - (4.0 * u1[i4, j4] - u1[i4 - 1, j4] - u1[i4 + 1, j4] - u1[i4, j4 - 1] - u1[i4, j4 + 1]) / hh1
            end
            # full-weighting restriction of r1 into f2; coarse point (q, s) sits on fine point (2q - 1, 2s - 1)
            for k5 = 1:n2 * n2
                q5 = div(k5 - 1, n2) + 2
                s5 = mod(k5 - 1, n2) + 2
                a5 = 2 * q5 - 1
                b5 = 2 * s5 - 1
                f2[q5, s5] = 0.0625 * (r1[a5 - 1, b5 - 1] + r1[a5 - 1, b5 + 1] + r1[a5 + 1, b5 - 1] + r1[a5 + 1, b5 + 1]) + 0.125 * (r1[a5 - 1, b5] + r1[a5 + 1, b5] + r1[a5, b5 - 1] + r1[a5, b5 + 1]) + 0.25 * r1[a5, b5]
            end
            # coarse correction starts from zero
            for k6 = 1:n2 * n2
                i6 = div(k6 - 1, n2) + 2
                j6 = mod(k6 - 1, n2) + 2
                u2[i6, j6] = 0.0
            end
            # level 2: pre-smooth, residual, restrict
            # damped Jacobi on level 2: residual into r2, then one independent update of every point
            for sk7 = 1:nu1
                for k8 = 1:n2 * n2
                    i8 = div(k8 - 1, n2) + 2
                    j8 = mod(k8 - 1, n2) + 2
                    r2[i8, j8] = f2[i8, j8] - (4.0 * u2[i8, j8] - u2[i8 - 1, j8] - u2[i8 + 1, j8] - u2[i8, j8 - 1] - u2[i8, j8 + 1]) / hh2
                end
                for k9 = 1:n2 * n2
                    i9 = div(k9 - 1, n2) + 2
                    j9 = mod(k9 - 1, n2) + 2
                    u2[i9, j9] = u2[i9, j9] + wq2 * r2[i9, j9]
                end
            end
            for k10 = 1:n2 * n2
                i10 = div(k10 - 1, n2) + 2
                j10 = mod(k10 - 1, n2) + 2
                r2[i10, j10] = f2[i10, j10] - (4.0 * u2[i10, j10] - u2[i10 - 1, j10] - u2[i10 + 1, j10] - u2[i10, j10 - 1] - u2[i10, j10 + 1]) / hh2
            end
            # full-weighting restriction of r2 into f3; coarse point (q, s) sits on fine point (2q - 1, 2s - 1)
            for k11 = 1:n3 * n3
                q11 = div(k11 - 1, n3) + 2
                s11 = mod(k11 - 1, n3) + 2
                a11 = 2 * q11 - 1
                b11 = 2 * s11 - 1
                f3[q11, s11] = 0.0625 * (r2[a11 - 1, b11 - 1] + r2[a11 - 1, b11 + 1] + r2[a11 + 1, b11 - 1] + r2[a11 + 1, b11 + 1]) + 0.125 * (r2[a11 - 1, b11] + r2[a11 + 1, b11] + r2[a11, b11 - 1] + r2[a11, b11 + 1]) + 0.25 * r2[a11, b11]
            end
            # coarse correction starts from zero
            for k12 = 1:n3 * n3
                i12 = div(k12 - 1, n3) + 2
                j12 = mod(k12 - 1, n3) + 2
                u3[i12, j12] = 0.0
            end
            # coarsest level (one point when n1 = 2^3 - 1): exact solve of 4 u / hh = f
            for k13 = 1:n3 * n3
                i13 = div(k13 - 1, n3) + 2
                j13 = mod(k13 - 1, n3) + 2
                u3[i13, j13] = 0.25 * hh3 * f3[i13, j13]
            end
            # level 2: prolong the correction, post-smooth
            # bilinear prolongation of u3 added into u2: four parity classes of fine points
            # fine points that coincide with a coarse point
            for k14 = 1:n3 * n3
                q14 = div(k14 - 1, n3) + 1
                s14 = mod(k14 - 1, n3) + 1
                u2[2 * q14 + 1, 2 * s14 + 1] = u2[2 * q14 + 1, 2 * s14 + 1] + u3[q14 + 1, s14 + 1]
            end
            # fine points between two coarse points along the first index
            for k15 = 1:(n3 + 1) * n3
                q15 = div(k15 - 1, n3)
                s15 = mod(k15 - 1, n3) + 1
                u2[2 * q15 + 2, 2 * s15 + 1] = u2[2 * q15 + 2, 2 * s15 + 1] + 0.5 * (u3[q15 + 1, s15 + 1] + u3[q15 + 2, s15 + 1])
            end
            # fine points between two coarse points along the second index
            for k16 = 1:n3 * (n3 + 1)
                q16 = div(k16 - 1, n3 + 1) + 1
                s16 = mod(k16 - 1, n3 + 1)
                u2[2 * q16 + 1, 2 * s16 + 2] = u2[2 * q16 + 1, 2 * s16 + 2] + 0.5 * (u3[q16 + 1, s16 + 1] + u3[q16 + 1, s16 + 2])
            end
            # fine points at the centre of four coarse points
            for k17 = 1:(n3 + 1) * (n3 + 1)
                q17 = div(k17 - 1, n3 + 1)
                s17 = mod(k17 - 1, n3 + 1)
                u2[2 * q17 + 2, 2 * s17 + 2] = u2[2 * q17 + 2, 2 * s17 + 2] + 0.25 * (u3[q17 + 1, s17 + 1] + u3[q17 + 2, s17 + 1] + u3[q17 + 1, s17 + 2] + u3[q17 + 2, s17 + 2])
            end
            # damped Jacobi on level 2: residual into r2, then one independent update of every point
            for sk18 = 1:nu2
                for k19 = 1:n2 * n2
                    i19 = div(k19 - 1, n2) + 2
                    j19 = mod(k19 - 1, n2) + 2
                    r2[i19, j19] = f2[i19, j19] - (4.0 * u2[i19, j19] - u2[i19 - 1, j19] - u2[i19 + 1, j19] - u2[i19, j19 - 1] - u2[i19, j19 + 1]) / hh2
                end
                for k20 = 1:n2 * n2
                    i20 = div(k20 - 1, n2) + 2
                    j20 = mod(k20 - 1, n2) + 2
                    u2[i20, j20] = u2[i20, j20] + wq2 * r2[i20, j20]
                end
            end
            # level 1: prolong the correction, post-smooth
            # bilinear prolongation of u2 added into u1: four parity classes of fine points
            # fine points that coincide with a coarse point
            for k21 = 1:n2 * n2
                q21 = div(k21 - 1, n2) + 1
                s21 = mod(k21 - 1, n2) + 1
                u1[2 * q21 + 1, 2 * s21 + 1] = u1[2 * q21 + 1, 2 * s21 + 1] + u2[q21 + 1, s21 + 1]
            end
            # fine points between two coarse points along the first index
            for k22 = 1:(n2 + 1) * n2
                q22 = div(k22 - 1, n2)
                s22 = mod(k22 - 1, n2) + 1
                u1[2 * q22 + 2, 2 * s22 + 1] = u1[2 * q22 + 2, 2 * s22 + 1] + 0.5 * (u2[q22 + 1, s22 + 1] + u2[q22 + 2, s22 + 1])
            end
            # fine points between two coarse points along the second index
            for k23 = 1:n2 * (n2 + 1)
                q23 = div(k23 - 1, n2 + 1) + 1
                s23 = mod(k23 - 1, n2 + 1)
                u1[2 * q23 + 1, 2 * s23 + 2] = u1[2 * q23 + 1, 2 * s23 + 2] + 0.5 * (u2[q23 + 1, s23 + 1] + u2[q23 + 1, s23 + 2])
            end
            # fine points at the centre of four coarse points
            for k24 = 1:(n2 + 1) * (n2 + 1)
                q24 = div(k24 - 1, n2 + 1)
                s24 = mod(k24 - 1, n2 + 1)
                u1[2 * q24 + 2, 2 * s24 + 2] = u1[2 * q24 + 2, 2 * s24 + 2] + 0.25 * (u2[q24 + 1, s24 + 1] + u2[q24 + 2, s24 + 1] + u2[q24 + 1, s24 + 2] + u2[q24 + 2, s24 + 2])
            end
            # damped Jacobi on level 1: residual into r1, then one independent update of every point
            for sk25 = 1:nu2
                for k26 = 1:n1 * n1
                    i26 = div(k26 - 1, n1) + 2
                    j26 = mod(k26 - 1, n1) + 2
                    r1[i26, j26] = f1[i26, j26] - (4.0 * u1[i26, j26] - u1[i26 - 1, j26] - u1[i26 + 1, j26] - u1[i26, j26 - 1] - u1[i26, j26 + 1]) / hh1
                end
                for k27 = 1:n1 * n1
                    i27 = div(k27 - 1, n1) + 2
                    j27 = mod(k27 - 1, n1) + 2
                    u1[i27, j27] = u1[i27, j27] + wq1 * r1[i27, j27]
                end
            end
            # fine-grid residual, then its squared norm in a loop of its own
            for k28 = 1:n1 * n1
                i28 = div(k28 - 1, n1) + 2
                j28 = mod(k28 - 1, n1) + 2
                r1[i28, j28] = f1[i28, j28] - (4.0 * u1[i28, j28] - u1[i28 - 1, j28] - u1[i28 + 1, j28] - u1[i28, j28 - 1] - u1[i28, j28 + 1]) / hh1
            end
            rn = 0.0
            for k29 = 1:n1 * n1
                i29 = div(k29 - 1, n1) + 2
                j29 = mod(k29 - 1, n1) + 2
                rn = rn + r1[i29, j29] * r1[i29, j29]
            end
        end
    end
    return nothing
end
