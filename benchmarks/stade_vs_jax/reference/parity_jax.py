"""Compares JAX against the STADE kernels on the cases written by tools/dump_case.jl.
   python parity_jax.py CASEDIR [CASEDIR ...]
Checks the primal outputs and the gradients (JAX vjp with the same cotangent against the STADE adjoint)."""
import json, sys, os
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(__file__))
from jax_ref import FUNCS

from parity_common import load, rel


def main(dirs):
    bad = 0
    for d in dirs:
        meta, args, out, seed, grad = load(d)
        f = FUNCS[meta["kernel"]]
        ref = f({k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v) for k, v in args.items()})
        # primal: every array the kernel writes that the JAX function also returns
        errs = {k: rel(np.asarray(ref[k]), out[k]) for k in out if k in ref}
        missing = [k for k in out if k not in ref]
        e_primal = max(errs.values())
        # gradient: vjp over the independent float arrays that exist in the JAX inputs
        keys = [k for k in grad if k in args]
        def g(*xs):
            a = dict(args); a.update({k: x for k, x in zip(keys, xs)})
            r = f({k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v) for k, v in a.items()})
            return {k: r[k] for k in seed if k in r}
        _, vjp = jax.vjp(g, *[jnp.asarray(args[k]) for k in keys])
        cot = {k: jnp.asarray(seed[k]) for k in seed if k in ref}
        gr = vjp(cot)
        gerr = {k: rel(np.asarray(v), grad[k]) for k, v in zip(keys, gr)}
        e_grad = max(gerr.values())
        ok = e_primal < 1e-10 and e_grad < 1e-9
        bad += (not ok)
        print(f"{'PASS' if ok else 'FAIL'} {meta['kernel']:18s} {meta['size']:8s} primal {e_primal:.2e}  gradient {e_grad:.2e}"
              + (f"   (not in JAX output: {missing})" if missing else "")
              + (f"   (STADE gradient without JAX input: {[k for k in grad if k not in args]})" if [k for k in grad if k not in args] else ""))
        if not ok:
            print("   primal errors:", {k: f"{v:.1e}" for k, v in errs.items()})
            print("   gradient errors:", {k: f"{v:.1e}" for k, v in gerr.items()})
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main(sys.argv[1:])
