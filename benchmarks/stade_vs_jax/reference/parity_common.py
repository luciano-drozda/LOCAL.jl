"""Case reader shared by parity_jax.py and parity_torch.py (numpy only)."""
import json, os
import numpy as np


def load(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    args = {}
    for e in meta["args"]:
        n = e["name"]
        if e["kind"] == "scalar":
            args[n] = e["value"]
        else:
            dt = np.float64 if e["kind"] == "float" else np.int64
            args[n] = np.fromfile(os.path.join(d, n + ".in.bin"), dtype=dt)
    rd = lambda n, suf: np.fromfile(os.path.join(d, f"{n}.{suf}.bin"), dtype=np.float64)
    out = {n: rd(n, "out") for n in meta["deps"] if os.path.exists(os.path.join(d, n + ".out.bin"))}
    seed = {n: rd(n, "seed") for n in out}
    grad = {n: rd(n, "grad") for n in meta["indeps"] if os.path.exists(os.path.join(d, n + ".grad.bin"))}
    return meta, args, out, seed, grad

def rel(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))
