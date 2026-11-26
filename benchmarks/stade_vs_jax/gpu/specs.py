SZ = {
 "heat3d_step": ["bench_1e5","bench_1e6","bench_8e6"],
 "heat3d_multi": ["bench_1e5","bench_1e6"],
 "heat1d_multi": ["bench_1e5_t8","bench_1e6_t2","bench_1e6_t8","bench_1e6_t32"],
 "wave2d_step": ["bench_1e5","bench_1e6","bench_16e6"], "burgers2d_step": ["bench_1e5","bench_1e6","bench_16e6"],
 "swe2d_step": ["bench_1e5","bench_1e6","bench_16e6"], "euler2d_step": ["bench_1e5","bench_1e6","bench_16e6"],
 "ufv_scatter_step": ["bench_1e5","bench_1e6","bench_4e6"], "ufv_gather_step": ["bench_1e5","bench_1e6","bench_4e6"],
 "mlp3": ["bench_3M","bench_13M","bench_79M"], "mlp3_push": ["bench_3M","bench_13M","bench_79M"],
 "mpnn_gather": ["bench_1e5","bench_1e6"],
 "transformer_block": ["bench_d512_T512","bench_d1024_T1024","bench_d2048_T1024"],
 "unet2": ["bench_c16_n256","bench_c32_n128","bench_c64_n64","bench_c128_n32"],
}
def spec(k, be, fuse, full=True):
    sizes = ["tiny!", "small!"] + (SZ[k] if full else [])
    return f"{k}:{be}:{fuse}:" + "+".join(sizes)
