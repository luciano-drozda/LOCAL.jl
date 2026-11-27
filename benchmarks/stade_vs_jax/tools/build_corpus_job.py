"""Builds the payload of one GPU corpus-validation job: STADE.jl and the tests with the corpus in a bundle, and the call of validate_corpus_gpu to run.
Usage: python3 build_corpus_job.py OUTDIR "CALL" [--hard-s N]     (CALL is the keyword part, for example "fuse_ii_loops = true")
Writes OUTDIR/job.json. Submit it with cluster/go.sh. The test/ files are copied from the repository; the corpus baselines (.yaml) are generated on the device host."""
import argparse, base64, os, shutil, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(PKG, "..", ".."))
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("call"); ap.add_argument("--hard-s", type=int, default=6600); ap.add_argument("--stade-src", default=os.path.join(REPO, "src", "STADE.jl"))
o = ap.parse_args()
out = os.path.abspath(o.out); B = out + "/bundle"
shutil.rmtree(B, ignore_errors=True); os.makedirs(B + "/src"); os.makedirs(B + "/test")
shutil.copy(o.stade_src, B + "/src/STADE.jl")
for f in os.listdir(f"{REPO}/test"):
    p = f"{REPO}/test/{f}"
    if os.path.isfile(p) and f.endswith(".jl"): shutil.copy(p, f"{B}/test/{f}")
shutil.copytree(f"{REPO}/test/val-corpus", f"{B}/test/val-corpus", ignore=shutil.ignore_patterns("*.gpu.yaml", "*.out", "*.log"))
subprocess.check_call(["tar", "czf", out + "/bundle.tgz", "-C", B, "."])
drv = open(f"{HERE}/../cluster/corpus_driver.jl").read()
for t, v in (("__TGZ__", base64.b64encode(open(out + "/bundle.tgz", "rb").read()).decode()), ("__HARD_S__", str(o.hard_s)), ("__CALL__", o.call.replace("\\", "\\\\").replace('"', '\\"'))):
    drv = drv.replace(t, v)
import json
json.dump({"input": {"_script": drv, "_topology": "gpu_v100"}}, open(out + "/job.json", "w"))
print("job.json", os.path.getsize(out + "/job.json") // 1024, "KB; call:", o.call)
