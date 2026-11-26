"""JAX versions of the benchmark kernels, same math and same array layouts as the STADE kernels.

Every function takes a dict `a` (the kernel's argument names, flat numpy/JAX arrays and Python ints)
and returns a dict of every array the kernel writes (intermediates included, flat). That is what the
parity check compares. The benchmark variants (jit, scan, checkpoint, fused attention) are built on top
of these functions in the driver; they are written here in the form a JAX user would write them:
whole-array operations, roll for periodic neighbors, each face flux computed once and shifted.
"""
import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)


def _nb(u, axis):  # (west/south neighbor, east/north neighbor) of every cell, periodic
    return jnp.roll(u, 1, axis), jnp.roll(u, -1, axis)


# ---------------------------------------------------------------- finite differences
def heat3d_step_core(u):
    s = 0.0
    for ax in range(3):
        lo, hi = _nb(u, ax)
        s = s + lo + hi
    return u + 0.1 * (s - 6.0 * u)


def heat3d_step(a):
    n = int(a["n"])
    u = jnp.asarray(a["u"]).reshape(n, n, n)
    return {"un": heat3d_step_core(u).reshape(-1)}


def heat3d_multi(a):
    n, npairs = int(a["n"]), int(a["npairs"])
    u = jnp.asarray(a["u"]).reshape(n, n, n)
    v = u
    for _ in range(npairs):
        v = heat3d_step_core(u)
        u = heat3d_step_core(v)
    return {"u": u.reshape(-1), "v": v.reshape(-1)}


def heat1d_multi(a):
    # time loop inside the kernel: 2 * nsteps three-point steps, ping-pong between u and v; boundaries fixed
    n, nsteps = int(a["n"]), int(a["nsteps"])
    u = jnp.asarray(a["u"])
    v = jnp.asarray(a["v"]).at[0].set(u[0]).at[n - 1].set(u[n - 1])
    for _ in range(nsteps):
        v = v.at[1:-1].set(u[1:-1] + 0.25 * (u[:-2] - 2.0 * u[1:-1] + u[2:]))
        u = u.at[1:-1].set(v[1:-1] + 0.25 * (v[:-2] - 2.0 * v[1:-1] + v[2:]))
    return {"u": u, "v": v}


def wave2d_step(a):
    n = int(a["n"])
    um, u, c2 = (jnp.asarray(a[k]).reshape(n, n) for k in ("um", "u", "c2"))
    lap = sum(sum(_nb(u, ax)) for ax in range(2)) - 4.0 * u
    return {"up": (2.0 * u - um + c2 * lap).reshape(-1)}


def burgers2d_step(a):
    n = int(a["n"])
    u, v = (jnp.asarray(a[k]).reshape(n, n) for k in ("u", "v"))
    upos, uneg, vpos, vneg = jnp.maximum(u, 0.0), jnp.minimum(u, 0.0), jnp.maximum(v, 0.0), jnp.minimum(v, 0.0)

    def upd(f):
        fw, fe = _nb(f, 1)
        fs, fn = _nb(f, 0)
        adv = upos * (f - fw) + uneg * (fe - f) + vpos * (f - fs) + vneg * (fn - f)
        return f - 0.1 * adv + 0.05 * (fw + fe + fs + fn - 4.0 * f)

    return {"un": upd(u).reshape(-1), "vn": upd(v).reshape(-1)}


# ---------------------------------------------------------------- finite volumes
def _hll(hL, nL, tL, hR, nR, tR):
    unL, unR = nL / hL, nR / hR
    cL, cR = jnp.sqrt(9.81 * hL), jnp.sqrt(9.81 * hR)
    sL = jnp.minimum(jnp.minimum(unL - cL, unR - cR), 0.0)
    sR = jnp.maximum(jnp.maximum(unL + cL, unR + cR), 0.0)
    iw = 1.0 / (sR - sL)
    f1L, f1R = nL * unL + 4.905 * hL * hL, nR * unR + 4.905 * hR * hR
    f2L, f2R = tL * unL, tR * unR
    sLR = sL * sR
    return ((sR * nL - sL * nR + sLR * (hR - hL)) * iw,
            (sR * f1L - sL * f1R + sLR * (nR - nL)) * iw,
            (sR * f2L - sL * f2R + sLR * (tR - tL)) * iw)


def swe2d_step(a):
    n = int(a["n"])
    h, hu, hv = (jnp.asarray(a[k]).reshape(n, n) for k in ("h", "hu", "hv"))
    # x faces (east face of each cell), y faces (north face)
    Fe = _hll(h, hu, hv, jnp.roll(h, -1, 1), jnp.roll(hu, -1, 1), jnp.roll(hv, -1, 1))
    Fh_x, Fhu_x, Fhv_x = Fe
    Gn = _hll(h, hv, hu, jnp.roll(h, -1, 0), jnp.roll(hv, -1, 0), jnp.roll(hu, -1, 0))
    Gh_y, Ghv_y, Ghu_y = Gn
    def net(Fx, Gy):
        return jnp.roll(Fx, 1, 1), jnp.roll(Gy, 1, 0)
    out = {}
    for name, q, Fx, Gy in (("hn", h, Fh_x, Gh_y), ("hun", hu, Fhu_x, Ghu_y), ("hvn", hv, Fhv_x, Ghv_y)):
        Fw, Gs = net(Fx, Gy)
        out[name] = (q - 0.1 * (Fx - Fw + Gy - Gs)).reshape(-1)
    return out


def _minmod(x, y):
    return 0.5 * (jnp.sign(x) + jnp.sign(y)) * jnp.minimum(jnp.abs(x), jnp.abs(y))


def _euler_flux(UL, UR, axis):
    def side(U):
        r, mx, my, E = U
        un, ut = (mx, my) if axis == 1 else (my, mx)
        u = un / r
        p = 0.4 * (E - 0.5 * (mx * mx + my * my) / r)
        c = jnp.sqrt(1.4 * p / r)
        F = {"mass": un, "norm": un * u + p, "tan": ut * u, "en": (E + p) * u}
        return u, c, F
    uL, cL, FL = side(UL)
    uR, cR, FR = side(UR)
    sp = jnp.maximum(jnp.abs(uL) + cL, jnp.abs(uR) + cR)
    names = ("mass", "norm", "tan", "en") if axis == 1 else ("mass", "tan", "norm", "en")
    return [0.5 * (FL[nm] + FR[nm]) - 0.5 * sp * (UR[q] - UL[q]) for q, nm in enumerate(names)]


def euler2d_step(a):
    n = int(a["n"])
    U = [jnp.asarray(a[k]).reshape(n, n) for k in ("rho", "mx", "my", "ener")]
    res = [0.0] * 4
    for axis in (1, 0):
        L, R = [], []
        for q in U:
            lo, hi = _nb(q, axis)
            s = _minmod(q - lo, hi - q)              # limited slope of every cell
            L.append(q + 0.5 * s)                      # state at the high face of the cell
            R.append(jnp.roll(q - 0.5 * s, -1, axis))  # state at the low face of the next cell
        G = _euler_flux(L, R, axis)                    # flux through the high face of every cell
        for i in range(4):
            res[i] = res[i] + G[i] - jnp.roll(G[i], 1, axis)
    return {nm: (U[i] - 0.1 * res[i]).reshape(-1) for i, nm in enumerate(("rhon", "mxn", "myn", "enern"))}


def ufv_scatter_step(a):
    phi, vel = jnp.asarray(a["phi"]), jnp.asarray(a["vel"])
    e1, e2 = jnp.asarray(a["e1"]) - 1, jnp.asarray(a["e2"]) - 1
    nc = int(a["nc"])
    f = vel * 0.5 * (phi[e1] + phi[e2]) + 0.5 * (phi[e1] - phi[e2])
    rhs = jnp.zeros(nc).at[e1].add(f).at[e2].add(-f)
    return {"rhs": rhs, "phin": phi - 0.1 * rhs}


def ufv_gather_step(a):
    nc, deg = int(a["nc"]), int(a["deg"])
    phi = jnp.asarray(a["phi"])
    velc = jnp.asarray(a["velc"]).reshape(nc, deg)
    nbr = jnp.asarray(a["nbr"]).reshape(nc, deg) - 1
    pj = phi[nbr]
    acc = jnp.sum(velc * 0.5 * (phi[:, None] + pj) + 0.5 * (phi[:, None] - pj), axis=1)
    return {"phin": phi - 0.1 * acc}


# ---------------------------------------------------------------- networks
def mlp3(a):
    n0, n1, n2, n3 = (int(a[k]) for k in ("n0", "n1", "n2", "n3"))
    x = jnp.asarray(a["x"])
    h1 = jnp.tanh(x @ jnp.asarray(a["w1"]).reshape(n0, n1) + a["b1"])
    h2 = jnp.tanh(h1 @ jnp.asarray(a["w2"]).reshape(n1, n2) + a["b2"])
    y = h2 @ jnp.asarray(a["w3"]).reshape(n2, n3) + a["b3"]
    return {"h1": h1, "h2": h2, "y": y}


def mlp3_push(a):
    n0, n1, n2, n3 = (int(a[k]) for k in ("n0", "n1", "n2", "n3"))
    x = jnp.asarray(a["x"])
    z1 = x @ jnp.asarray(a["w1"]).reshape(n0, n1) + a["b1"]
    a1 = jnp.tanh(z1)
    z2 = a1 @ jnp.asarray(a["w2"]).reshape(n1, n2) + a["b2"]
    a2 = jnp.tanh(z2)
    y = a2 @ jnp.asarray(a["w3"]).reshape(n2, n3) + a["b3"]
    return {"z1": z1, "a1": a1, "z2": z2, "a2": a2, "y": y}


def mpnn_gather(a):
    nn, nf, nm = (int(a[k]) for k in ("nn", "nf", "nm"))
    rowptr = jnp.asarray(a["rowptr"])
    src = jnp.asarray(a["src"]) - 1
    E = src.shape[0]
    indeg = rowptr[1:] - rowptr[:-1]
    dst = jnp.repeat(jnp.arange(nn), indeg, total_repeat_length=E)   # receiver of each edge
    h = jnp.asarray(a["h"]).reshape(nn, nf)
    asrc = h @ jnp.asarray(a["wsrc"]).reshape(nf, nm)
    adst = h @ jnp.asarray(a["wdst"]).reshape(nf, nm)
    msg = jax.nn.relu(asrc[src] + adst[dst] + a["bmsg"] + jnp.asarray(a["ef"]).reshape(E, 4) @ jnp.asarray(a["wedge"]).reshape(4, nm))
    agg = jax.ops.segment_sum(msg, dst, num_segments=nn)
    hout = jax.nn.relu(h @ jnp.asarray(a["wuh"]).reshape(nf, nf) + agg @ jnp.asarray(a["wua"]).reshape(nm, nf) + a["bupd"])
    return {"asrc": asrc.reshape(-1), "adst": adst.reshape(-1), "agg": agg.reshape(-1), "hout": hout.reshape(-1)}


def _ln(x, gam, bet):
    mu = x.mean(-1)
    var = ((x - mu[:, None]) ** 2).mean(-1)
    rs = 1.0 / jnp.sqrt(var + 1e-5)
    return mu, rs, (x - mu[:, None]) * rs[:, None] * gam + bet


def transformer_block(a):
    T, d, nh, dff = (int(a[k]) for k in ("tt", "d", "nh", "dff"))
    dh = d // nh
    x = jnp.asarray(a["x"]).reshape(T, d)
    mu1, rs1, xn1 = _ln(x, a["gam1"], a["bet1"])
    qkv = xn1 @ jnp.asarray(a["wqkv"]).reshape(d, 3 * d) + a["bqkv"]
    q = qkv[:, :d].reshape(T, nh, dh)
    k = qkv[:, d:2 * d].reshape(T, nh, dh)
    v = qkv[:, 2 * d:].reshape(T, nh, dh)
    sc = jnp.exp(jnp.einsum("thc,uhc->htu", q, k) / jnp.sqrt(dh * 1.0))
    den = sc.sum(-1)
    att = (jnp.einsum("htu,uhc->thc", sc, v) / den.T[:, :, None]).reshape(T, d)
    x2 = x + att @ jnp.asarray(a["wo"]).reshape(d, d) + a["bo"]
    mu2, rs2, xn2 = _ln(x2, a["gam2"], a["bet2"])
    z = xn2 @ jnp.asarray(a["w1"]).reshape(d, dff) + a["bf1"]
    hmid = 0.5 * z * (1.0 + jnp.tanh(0.7978845608028654 * (z + 0.044715 * z ** 3)))
    y = x2 + hmid @ jnp.asarray(a["w2"]).reshape(dff, d) + a["bf2"]
    f = lambda t: t.reshape(-1)
    return {"mu1": mu1, "rs1": rs1, "xn1": f(xn1), "qkv": f(qkv), "sc": f(sc), "den": f(den), "att": f(att), "x2": f(x2),
            "mu2": mu2, "rs2": rs2, "xn2": f(xn2), "hmid": f(hmid), "y": f(y)}


def unet2(a):
    q, c0, c, co = (int(a[k]) for k in ("q", "c0", "c", "co"))
    n = 4 * q
    c2, c4 = 2 * c, 4 * c
    wts, bs = jnp.asarray(a["wts"]), jnp.asarray(a["bs"])
    # same offsets as the kernel
    cins = [c0, c, c, c2, c2, c4, c4 + c2, c2, c2 + c, c]
    couts = [c, c, c2, c2, c4, c4, c2, c2, c, c]
    wo, bo = [], []
    w_at = b_at = 0
    for ci, cout in zip(cins, couts):
        wo.append(w_at); bo.append(b_at)
        w_at += 9 * ci * cout; b_at += cout
    wo.append(w_at); bo.append(b_at)

    def conv(x, i, relu=True):                       # x: (cin, H, H)
        cin, cout = cins[i], couts[i]
        w = wts[wo[i]:wo[i] + 9 * cin * cout].reshape(cout, cin, 3, 3)
        xp = jnp.pad(x, ((0, 0), (1, 1), (1, 1)), mode="edge")
        y = lax.conv_general_dilated(xp[None], w, (1, 1), "VALID", dimension_numbers=("NCHW", "OIHW", "NCHW"))[0]
        y = y + bs[bo[i]:bo[i] + cout][:, None, None]
        return jax.nn.relu(y) if relu else y

    pool = lambda t: t.reshape(t.shape[0], t.shape[1] // 2, 2, t.shape[2] // 2, 2).mean((2, 4))
    up = lambda t: jnp.repeat(jnp.repeat(t, 2, axis=1), 2, axis=2)
    img = jnp.asarray(a["img"]).reshape(c0, n, n)
    ea1 = conv(img, 0); ea2 = conv(ea1, 1); pl1 = pool(ea2)
    eb1 = conv(pl1, 2); eb2 = conv(eb1, 3); pl2 = pool(eb2)
    bn1 = conv(pl2, 4); bn2 = conv(bn1, 5)
    da1 = conv(jnp.concatenate([up(bn2), eb2], 0), 6); da2 = conv(da1, 7)
    fa1 = conv(jnp.concatenate([up(da2), ea2], 0), 8); fa2 = conv(fa1, 9)
    wf = wts[wo[10]:wo[10] + c * co].reshape(co, c)
    outp = jnp.einsum("oc,cyx->oyx", wf, fa2) + bs[bo[10]:bo[10] + co][:, None, None]
    f = lambda t: t.reshape(-1)
    return {k: f(v) for k, v in dict(ea1=ea1, ea2=ea2, pl1=pl1, eb1=eb1, eb2=eb2, pl2=pl2, bn1=bn1, bn2=bn2,
                                     da1=da1, da2=da2, fa1=fa1, fa2=fa2, outp=outp).items()}


FUNCS = {f.__name__: f for f in (heat3d_step, heat3d_multi, heat1d_multi, wave2d_step, burgers2d_step, swe2d_step, euler2d_step,
                                 ufv_scatter_step, ufv_gather_step, mlp3, mlp3_push, mpnn_gather, transformer_block, unet2)}
