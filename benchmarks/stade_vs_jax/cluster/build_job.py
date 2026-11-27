"""Builds the payload of one benchmark job: the STADE runs and the JAX runs of the chosen phases, in one V100 job.
Usage: STADE_SRC=/path/to/src/STADE.jl python3 build_job.py OUTDIR --stade-dir DIR --roles DIR [options]
  --stade-dir DIR   the OUTDIR of gen_stade_scripts.sh (it holds all/KERNEL_C.jl and all/KERNEL_A.jl)
  --roles DIR       the OUTDIR of make_roles.sh (it holds roles/KERNEL.small/meta.json)
  --phases LIST     comma list of time, mem_primal, mem_adjoint (default: all three)
  --kernels K ...   only these kernels (default: all of gpu/specs.py)
  --max-sizes N     only the first N benchmark sizes of each kernel (a smoke job or a local test). Use the same N in gen_stade_scripts.sh (MAX_SIZES).
  --configs LIST    comma list of STADE configs (default C,A). P and G are the ablation configs of gen_stade_scripts.sh.
  --plan-file FILE  use the lines of this file as the plan (fw, config, kernel, size, phases; tab separated) instead of the plan made from the options above.
                    Used for the profile job (phases prof_primal, prof_adjoint, ncu_adjoint).
  --no-stade, --no-jax   leave one framework out
  --limit-min N     the time limit of the V100 job (default 120). No new run starts 15 min before it.
  --stade-tmo S, --jax-tmo S   the time limit of one run (default 1800 and 500)
Order of the runs: every STADE kernel (config C, then config A; the phases in the order given), then every JAX case (the phases in the order given).
Config C runs every phase. Every other config skips mem_primal, because it has the same primal code as config C. JAX runs every phase.
Result: OUTDIR/job.json (the payload), OUTDIR/plan.tsv and OUTDIR/versions.json. Submit job.json with go.sh."""
import argparse, base64, datetime, hashlib, json, os, shutil, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PKG, "gpu"))
from specs import SZ

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--stade-dir"); ap.add_argument("--roles", required=True)
ap.add_argument("--phases", default="time,mem_primal,mem_adjoint"); ap.add_argument("--kernels", nargs="*")
ap.add_argument("--max-sizes", type=int, default=0); ap.add_argument("--no-stade", action="store_true"); ap.add_argument("--no-jax", action="store_true")
ap.add_argument("--configs", default="C,A"); ap.add_argument("--plan-file"); ap.add_argument("--limit-min", type=int, default=120); ap.add_argument("--stade-tmo", type=int, default=1800); ap.add_argument("--jax-tmo", type=int, default=500)
o = ap.parse_args()
phases = o.phases.split(",")
bad = [p for p in phases if p not in ("time", "mem_primal", "mem_adjoint", "prof_primal", "prof_adjoint", "ncu_adjoint")]
if bad: sys.exit(f"unknown phase {bad}")
fplan = [l.rstrip("\n").split("\t") for l in open(o.plan_file) if l.strip() and not l.startswith("#")] if o.plan_file else None
if fplan:
    for r in fplan:
        if len(r) != 5 or r[0] not in ("stade", "jax"): sys.exit(f"bad plan line {r}")
    o.no_stade = not any(r[0] == "stade" for r in fplan); o.no_jax = not any(r[0] == "jax" for r in fplan)
kernels = o.kernels or (sorted({r[2] for r in fplan}) if fplan else list(SZ))
for k in kernels:
    if k not in SZ: sys.exit(f"unknown kernel {k}")
if not o.no_stade and not o.stade_dir: sys.exit("--stade-dir is required unless --no-stade")

out = os.path.abspath(o.out); B = out + "/bundle"
shutil.rmtree(B, ignore_errors=True)
for d in ("tools", "roles", "stade", "cluster"): os.makedirs(f"{B}/{d}")
for f in ("inputs.jl", "dump_inputs.jl"): shutil.copy(f"{PKG}/tools/{f}", f"{B}/tools/{f}")
for f in ("gpu_state.sh", "job_run.sh", "prof_cmd.sh", "prof_pack.jl"): shutil.copy(f"{HERE}/{f}", f"{B}/cluster/{f}")
if not o.no_jax:
    shutil.copytree(f"{PKG}/kernels", f"{B}/kernels")
    shutil.copytree(f"{PKG}/reference", f"{B}/reference", ignore=shutil.ignore_patterns("__pycache__"))
    for k in kernels:
        os.makedirs(f"{B}/roles/{k}.small")
        shutil.copy(f"{o.roles}/roles/{k}.small/meta.json", f"{B}/roles/{k}.small/meta.json")

plan = []
if fplan:
    for fw, cfg, k, sz, ph in fplan:
        if fw == "stade":
            src = f"{o.stade_dir}/all/{k}_{cfg}.jl"
            if not os.path.isfile(src): sys.exit(f"missing {src}")
            shutil.copy(src, f"{B}/stade/{k}_{cfg}.jl")
            # a script holds the runner of the time it was generated: a profile phase needs a script that knows it
            if any(p.startswith(("prof", "ncu")) for p in ph.split(",")) and "prof_adjoint" not in open(src).read():
                sys.exit(f"{src} was generated before the profile phases existed. Generate it again (gen_stade_scripts.sh).")
        plan.append((fw, cfg, k, sz, ph))
elif not o.no_stade:
    for k in kernels:
        for cfg in o.configs.split(","):
            src = f"{o.stade_dir}/all/{k}_{cfg}.jl"
            if not os.path.isfile(src): sys.exit(f"missing {src}")
            shutil.copy(src, f"{B}/stade/{k}_{cfg}.jl")
            ph = [p for p in phases if not (cfg != "C" and p == "mem_primal")]
            if ph: plan.append(("stade", cfg, k, "-", ",".join(ph)))
if not o.no_jax and not fplan:
    for k in kernels:
        for s in SZ[k][:o.max_sizes or None]: plan.append(("jax", "J", k, s, ",".join(phases)))
plan_txt = "".join("\t".join(r) + "\n" for r in plan)
open(f"{B}/plan.tsv", "w").write(plan_txt); open(f"{out}/plan.tsv", "w").write(plan_txt)

# versions: the STADE commit (with a dirty flag) and a hash of the kit files
def sh(*a, cwd=None):
    try: return subprocess.check_output(a, cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return ""
src = os.environ.get("STADE_SRC", "")
sdir = os.path.dirname(src) if src else PKG
commit = sh("git", "rev-parse", "HEAD", cwd=sdir)
dirty = bool(sh("git", "status", "--porcelain", "--", src or ".", PKG, cwd=sdir))
h = hashlib.sha256()
for root, ds, fs in os.walk(PKG):
    ds[:] = sorted(d for d in ds if d not in ("__pycache__", "work", "results", ".git"))
    for f in sorted(fs):
        if f.endswith((".pyc", ".log")): continue
        p = os.path.join(root, f); h.update(os.path.relpath(p, PKG).encode()); h.update(open(p, "rb").read())
if src and os.path.isfile(src): h.update(open(src, "rb").read())
ver = {"stade_commit": commit, "stade_dirty": dirty, "kit_hash": h.hexdigest()[:16], "built_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
       "phases": phases, "kernels": kernels, "limit_min": o.limit_min}
json.dump(ver, open(f"{B}/versions.json", "w")); json.dump(ver, open(f"{out}/versions.json", "w"), indent=1)
if dirty: print("WARNING: the STADE tree or the kit has uncommitted changes (recorded as stade_dirty = true)", file=sys.stderr)

subprocess.check_call(["tar", "czf", out + "/bundle.tgz", "-C", B, "."])
budget = (o.limit_min - 15) * 60; hard = budget + 600
drv = open(f"{HERE}/job_driver.jl").read()
for t, v in (("__TGZ__", base64.b64encode(open(out + "/bundle.tgz", "rb").read()).decode()), ("__HARD_S__", str(hard)), ("__BUDGET_S__", str(budget)),
             ("__STADE_TMO__", str(o.stade_tmo)), ("__JAX_TMO__", str(o.jax_tmo))):
    drv = drv.replace(t, v)
json.dump({"input": {"_script": drv, "_topology": "gpu_v100"}}, open(out + "/job.json", "w"))
nr = sum(len(r[4].split(",")) for r in plan)
print(f"{nr} runs ({sum(1 for r in plan if r[0]=='stade')} STADE entries, {sum(1 for r in plan if r[0]=='jax')} JAX cases), phases {phases}, budget {budget//60} min ->", out + "/job.json")
