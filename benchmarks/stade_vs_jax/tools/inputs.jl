# Input builders for the benchmark kernels. One function per kernel returns the argument list,
# in kernel order, as Pair{Symbol,Any}. SIZES maps a size label to the integer arguments.
# Float arrays are Float64, integer arrays are Int64 (Julia indices, 1-based).
using Random

bench_kernel_path(name) = begin
    root = joinpath(@__DIR__, "..", "kernels")
    for sub in ("solvers", "nn")
        p = joinpath(root, sub, name * ".jl")
        isfile(p) && return p
    end
    error("no kernel file for $name")
end

bi_rng(name, size) = MersenneTwister(hash((name, size)) % typemax(Int32))
bi_u(rng, n, lo, hi) = lo .+ (hi - lo) .* rand(rng, n)
bi_w(rng, nin, nout) = randn(rng, nin * nout) ./ sqrt(nin)   # input-major weight matrix
bi_z(n) = zeros(n)

# size label -> integer parameters; "tiny" is for the checks, "bench" are the benchmark sizes
function bench_sizes(name)
    t = Dict{String,Any}()
    if name in ("heat3d_step", "heat3d_step_s")
        t["small"] = (n = 7,)
        t["tiny"] = (n = 5,); t["bench_1e5"] = (n = 46,); t["bench_1e6"] = (n = 100,); t["bench_8e6"] = (n = 200,)
    elseif name in ("heat1d_multi", "heat1d_multi_t2")
        t["tiny"] = (n = 12, nsteps = 3); t["small"] = (n = 64, nsteps = 4)
        for (lab, nn) in (("1e5", 100_000), ("1e6", 1_000_000)), p in (2, 8, 32)
            t["bench_$(lab)_t$(p)"] = (n = nn, nsteps = p)
        end
    elseif name in ("heat3d_multi", "heat3d_multi_s")
        t["small"] = (n = 5, npairs = 3)
        t["tiny"] = (n = 4, npairs = 2); t["bench_1e5"] = (n = 46, npairs = 2); t["bench_1e6"] = (n = 100, npairs = 2)
    elseif name in ("conv_b1", "conv_b4", "conv_b8", "conv_b4i", "conv_b8i")
        t["tiny"] = (n = 6, cin = 3, cout = 8); t["small"] = (n = 9, cin = 4, cout = 8)
        t["bench_n256_c16"] = (n = 256, cin = 16, cout = 16); t["bench_n128_c32"] = (n = 128, cin = 32, cout = 32)
        t["bench_n64_c64"] = (n = 64, cin = 64, cout = 64); t["bench_n32_c128"] = (n = 32, cin = 128, cout = 128)
    elseif name == "unet2"
        t["small"] = (q = 3, c0 = 3, c = 3, co = 1)
        t["tiny"] = (q = 2, c0 = 2, c = 2, co = 2)
        t["bench_c32_n256"] = (q = 64, c0 = 3, c = 32, co = 3)
        t["bench_c64_n256"] = (q = 64, c0 = 3, c = 64, co = 3)
        t["bench_c128_n256"] = (q = 64, c0 = 3, c = 128, co = 3)
        # sizes whose keep_push_pop = false stacks fit in 16 GB (about 6 GiB); the three above need 24, 96 and 385 GiB
        t["bench_c16_n256"] = (q = 64, c0 = 3, c = 16, co = 3)
        t["bench_c32_n128"] = (q = 32, c0 = 3, c = 32, co = 3)
        t["bench_c64_n64"] = (q = 16, c0 = 3, c = 64, co = 3)
        t["bench_c128_n32"] = (q = 8, c0 = 3, c = 128, co = 3)
        # sizes of the narrow and wide U-Net plot (v0.4.44): 32 and 128 base channels at 64, 128 and 256 pixels
        t["bench_c32_n64"] = (q = 16, c0 = 3, c = 32, co = 3)
        t["bench_c128_n64"] = (q = 16, c0 = 3, c = 128, co = 3)
        t["bench_c128_n128"] = (q = 32, c0 = 3, c = 128, co = 3)
    elseif name == "transformer_block"
        t["small"] = (tt = 7, d = 12, nh = 3, dff = 20)
        t["tiny"] = (tt = 5, d = 8, nh = 2, dff = 12)
        t["bench_d32_T32"] = (tt = 32, d = 32, nh = 1, dff = 128); t["bench_d64_T64"] = (tt = 64, d = 64, nh = 2, dff = 256)
        t["bench_d128_T128"] = (tt = 128, d = 128, nh = 4, dff = 512); t["bench_d256_T256"] = (tt = 256, d = 256, nh = 8, dff = 1024)
        t["bench_d512_T512"] = (tt = 512, d = 512, nh = 8, dff = 2048)
        t["bench_d1024_T1024"] = (tt = 1024, d = 1024, nh = 16, dff = 4096)
        t["bench_d2048_T1024"] = (tt = 1024, d = 2048, nh = 32, dff = 8192)
        # shapes for the batched-product sweep: the score matrix is tt x tt per head, the inner length is d / nh
        t["bench_d128_T512_h16"] = (tt = 512, d = 128, nh = 16, dff = 512); t["bench_d256_T512_h16"] = (tt = 512, d = 256, nh = 16, dff = 1024)
        t["bench_d256_T512_h8"] = (tt = 512, d = 256, nh = 8, dff = 1024); t["bench_d512_T128_h8"] = (tt = 128, d = 512, nh = 8, dff = 2048)
        t["bench_d512_T64_h8"] = (tt = 64, d = 512, nh = 8, dff = 2048); t["bench_d1024_T256_h8"] = (tt = 256, d = 1024, nh = 8, dff = 4096)
        t["bench_d1024_T64_h4"] = (tt = 64, d = 1024, nh = 4, dff = 4096); t["bench_d1024_T128_h4"] = (tt = 128, d = 1024, nh = 4, dff = 4096)
    elseif name == "mpnn_gather"
        t["small"] = (nn = 40, nf = 6, nm = 5)
        t["tiny"] = (nn = 9, nf = 4, nm = 3)
        t["bench_1e5"] = (nn = 100_000, nf = 16, nm = 16); t["bench_1e6"] = (nn = 1_000_000, nf = 16, nm = 16)
    elseif name in ("mlp3", "mlp3_push")
        t["small"] = (n0 = 17, n1 = 9, n2 = 11, n3 = 7)
        t["tiny"] = (n0 = 5, n1 = 4, n2 = 6, n3 = 3)
        t["bench_3M"] = (n0 = 1024, n1 = 1024, n2 = 1024, n3 = 1024)
        t["bench_13M"] = (n0 = 2048, n1 = 2048, n2 = 2048, n3 = 2048)
        t["bench_79M"] = (n0 = 5120, n1 = 5120, n2 = 5120, n3 = 5120)
    elseif name in ("ufv_scatter_step", "ufv_gather_step")
        t["small"] = (nx = 9,)
        t["tiny"] = (nx = 6,); t["bench_1e5"] = (nx = 316,); t["bench_1e6"] = (nx = 1000,); t["bench_4e6"] = (nx = 2000,)
    elseif name in ("wave2d_step", "burgers2d_step", "swe2d_step", "euler2d_step") || startswith(name, "capprobe_")
        t["small"] = (n = 12,)
        t["tiny"] = (n = 7,); t["bench_1e5"] = (n = 316,); t["bench_1e6"] = (n = 1000,); t["bench_16e6"] = (n = 4000,)
    end
    return t
end

# Periodic nx x nx torus with 4 faces per cell, cells numbered in random order so that memory
# access has the locality of an unstructured mesh. Returns the face list and the cell-major table.
function bi_mesh(rng, nx)
    nc = nx * nx
    perm = randperm(rng, nc)                     # grid position -> cell number
    pos(x, y) = perm[mod(y - 1, nx) * nx + mod(x - 1, nx) + 1]
    ne = 2 * nc
    e1 = zeros(Int, ne); e2 = zeros(Int, ne); vel = bi_u(rng, ne, -1, 1)
    for y in 1:nx, x in 1:nx
        g = (y - 1) * nx + x
        e1[2g - 1] = pos(x, y); e2[2g - 1] = pos(x + 1, y)    # east face
        e1[2g] = pos(x, y);     e2[2g] = pos(x, y + 1)        # north face
    end
    deg = 4
    nbr = zeros(Int, nc * deg); velc = zeros(nc * deg)
    for y in 1:nx, x in 1:nx
        g = (y - 1) * nx + x; c = pos(x, y); o = (c - 1) * deg
        nbr[o + 1] = pos(x + 1, y); velc[o + 1] = vel[2g - 1]                        # east
        nbr[o + 2] = pos(x - 1, y); velc[o + 2] = -vel[2 * ((y - 1) * nx + mod(x - 2, nx) + 1) - 1]   # west
        nbr[o + 3] = pos(x, y + 1); velc[o + 3] = vel[2g]                            # north
        nbr[o + 4] = pos(x, y - 1); velc[o + 4] = -vel[2 * (mod(y - 2, nx) * nx + x)]  # south
    end
    return (nc = nc, ne = ne, deg = deg, e1 = e1, e2 = e2, vel = vel, nbr = nbr, velc = velc)
end

function bench_inputs(name, size)
    sz = bench_sizes(name)[size]
    rng = bi_rng(name, size)
    if name in ("heat3d_step", "heat3d_step_s")
        N = sz.n^3
        return Pair{Symbol,Any}[:u => randn(rng, N), :un => zeros(N), :n => sz.n]
    elseif name in ("heat1d_multi", "heat1d_multi_t2")
        return Pair{Symbol,Any}[:u => randn(rng, sz.n), :v => zeros(sz.n), :n => sz.n, :nsteps => sz.nsteps]
    elseif name in ("heat3d_multi", "heat3d_multi_s")
        N = sz.n^3
        return Pair{Symbol,Any}[:u => randn(rng, N), :v => zeros(N), :n => sz.n, :npairs => sz.npairs]
    elseif name == "wave2d_step"
        N = sz.n^2
        return Pair{Symbol,Any}[:um => randn(rng, N), :u => randn(rng, N), :c2 => bi_u(rng, N, 0.1, 0.4), :up => zeros(N), :n => sz.n]
    elseif startswith(name, "capprobe_")
        N = sz.n^2
        return Pair{Symbol,Any}[:u => bi_u(rng, N, -1, 1), :un => zeros(N), :n => sz.n]
    elseif name == "burgers2d_step"
        N = sz.n^2
        return Pair{Symbol,Any}[:u => bi_u(rng, N, -1, 1), :v => bi_u(rng, N, -1, 1), :un => zeros(N), :vn => zeros(N), :n => sz.n]
    elseif name == "swe2d_step"
        N = sz.n^2
        return Pair{Symbol,Any}[:h => bi_u(rng, N, 0.8, 1.2), :hu => bi_u(rng, N, -0.3, 0.3), :hv => bi_u(rng, N, -0.3, 0.3),
                                :hn => zeros(N), :hun => zeros(N), :hvn => zeros(N), :n => sz.n]
    elseif name == "euler2d_step"
        N = sz.n^2
        rho = bi_u(rng, N, 0.8, 1.2); u = bi_u(rng, N, -0.3, 0.3); v = bi_u(rng, N, -0.3, 0.3); p = bi_u(rng, N, 0.8, 1.2)
        return Pair{Symbol,Any}[:rho => rho, :mx => rho .* u, :my => rho .* v, :ener => p ./ 0.4 .+ 0.5 .* rho .* (u .^ 2 .+ v .^ 2),
                                :rhon => zeros(N), :mxn => zeros(N), :myn => zeros(N), :enern => zeros(N), :n => sz.n]
    end
    if name in ("conv_b1", "conv_b4", "conv_b8", "conv_b4i", "conv_b8i")
        return Pair{Symbol,Any}[:img => randn(rng, sz.cin * sz.n^2), :w => randn(rng, sz.cout * sz.cin * 9) ./ sqrt(9 * sz.cin), :bs => randn(rng, sz.cout), :out => zeros(sz.cout * sz.n^2), :n => sz.n, :cin => sz.cin, :cout => sz.cout]
    end
    if name == "unet2"
        q, c0, c, co = sz.q, sz.c0, sz.c, sz.co
        n = 4q; c2 = 2c; c4 = 4c
        nw = 9 * (c0 * c + c * c + c * c2 + c2 * c2 + c2 * c4 + c4 * c4 + (c4 + c2) * c2 + c2 * c2 + (c2 + c) * c + c * c) + c * co
        nb = 20c + co
        return Pair{Symbol,Any}[:img => randn(rng, c0 * n * n), :wts => 0.3 .* randn(rng, nw) ./ sqrt(9c), :bs => 0.05 .* randn(rng, nb),
            :ea1 => bi_z(c * n * n), :ea2 => bi_z(c * n * n), :pl1 => bi_z(c * (n ÷ 2)^2), :eb1 => bi_z(c2 * (n ÷ 2)^2), :eb2 => bi_z(c2 * (n ÷ 2)^2),
            :pl2 => bi_z(c2 * (n ÷ 4)^2), :bn1 => bi_z(c4 * (n ÷ 4)^2), :bn2 => bi_z(c4 * (n ÷ 4)^2), :da1 => bi_z(c2 * (n ÷ 2)^2), :da2 => bi_z(c2 * (n ÷ 2)^2),
            :fa1 => bi_z(c * n * n), :fa2 => bi_z(c * n * n), :outp => bi_z(co * n * n), :q => q, :c0 => c0, :c => c, :co => co]
    end
    if name == "transformer_block"
        T, d, nh, dff = sz.tt, sz.d, sz.nh, sz.dff
        return Pair{Symbol,Any}[:x => randn(rng, T * d), :gam1 => 1 .+ 0.1 .* randn(rng, d), :bet1 => 0.1 .* randn(rng, d),
            :wqkv => bi_w(rng, d, 3d), :bqkv => 0.1 .* randn(rng, 3d), :wo => bi_w(rng, d, d), :bo => 0.1 .* randn(rng, d),
            :gam2 => 1 .+ 0.1 .* randn(rng, d), :bet2 => 0.1 .* randn(rng, d), :w1 => bi_w(rng, d, dff), :bf1 => 0.1 .* randn(rng, dff),
            :w2 => bi_w(rng, dff, d), :bf2 => 0.1 .* randn(rng, d),
            :mu1 => bi_z(T), :rs1 => bi_z(T), :xn1 => bi_z(T * d), :qkv => bi_z(T * 3d), :sc => bi_z(nh * T * T), :den => bi_z(nh * T), :att => bi_z(T * d),
            :x2 => bi_z(T * d), :mu2 => bi_z(T), :rs2 => bi_z(T), :xn2 => bi_z(T * d), :hmid => bi_z(T * dff), :y => bi_z(T * d),
            :tt => T, :d => d, :nh => nh, :dff => dff]
    end
    if name == "mpnn_gather"
        # random graph, 3 to 9 incoming edges per node, edges sorted by receiver
        indeg = rand(rng, 3:9, sz.nn); rowptr = vcat(1, 1 .+ cumsum(indeg)); E = rowptr[end] - 1
        src = rand(rng, 1:sz.nn, E)
        return Pair{Symbol,Any}[:h => randn(rng, sz.nn * sz.nf), :ef => randn(rng, 4E), :src => src, :rowptr => rowptr,
            :wsrc => bi_w(rng, sz.nf, sz.nm), :wdst => bi_w(rng, sz.nf, sz.nm), :wedge => bi_w(rng, 4, sz.nm), :bmsg => 0.1 .* randn(rng, sz.nm),
            :wuh => bi_w(rng, sz.nf, sz.nf), :wua => bi_w(rng, sz.nm, sz.nf), :bupd => 0.1 .* randn(rng, sz.nf),
            :asrc => zeros(sz.nn * sz.nm), :adst => zeros(sz.nn * sz.nm), :agg => zeros(sz.nn * sz.nm), :hout => zeros(sz.nn * sz.nf),
            :nn => sz.nn, :nf => sz.nf, :nm => sz.nm]
    end
    if name in ("mlp3", "mlp3_push")
        base = Pair{Symbol,Any}[:x => randn(rng, sz.n0), :w1 => bi_w(rng, sz.n0, sz.n1), :b1 => 0.1 .* randn(rng, sz.n1),
                                :w2 => bi_w(rng, sz.n1, sz.n2), :b2 => 0.1 .* randn(rng, sz.n2), :w3 => bi_w(rng, sz.n2, sz.n3), :b3 => 0.1 .* randn(rng, sz.n3)]
        scratch = name == "mlp3" ? Pair{Symbol,Any}[:h1 => zeros(sz.n1), :h2 => zeros(sz.n2)] :
                                   Pair{Symbol,Any}[:z1 => zeros(sz.n1), :a1 => zeros(sz.n1), :z2 => zeros(sz.n2), :a2 => zeros(sz.n2)]
        return vcat(base, scratch, Pair{Symbol,Any}[:y => zeros(sz.n3), :n0 => sz.n0, :n1 => sz.n1, :n2 => sz.n2, :n3 => sz.n3])
    end
    if name in ("ufv_scatter_step", "ufv_gather_step")
        m = bi_mesh(rng, sz.nx)
        phi = randn(rng, m.nc)
        name == "ufv_scatter_step" && return Pair{Symbol,Any}[:phi => phi, :vel => m.vel, :e1 => m.e1, :e2 => m.e2, :rhs => zeros(m.nc), :phin => zeros(m.nc), :nc => m.nc, :ne => m.ne]
        return Pair{Symbol,Any}[:phi => phi, :velc => m.velc, :nbr => m.nbr, :phin => zeros(m.nc), :nc => m.nc, :deg => m.deg]
    end
    error("no inputs for $name")
end
