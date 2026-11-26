"""Checks the output of the three phases of one STADE runner script that ran on the CPU stand-in (tools/cpu_standin.py).
Usage: python3 check_phase_results.py TIME.json MEM_PRIMAL.json MEM_ADJOINT.json
On the stand-in the device memory is the exact sum of the registered arrays, so the checks are exact:
  time         both timers, no memory field, tiny agrees with the CPU
  mem_primal   the peak equals the inputs: no gradient buffer and no stack is on the device
  mem_adjoint  the setup equals inputs + gradient buffers + stacks, the peak is not below the setup, no timing field
Exit code 0 if all pass, 1 otherwise."""
import json, sys
fails = []
def load(p): return json.loads(open(p).read().strip().splitlines()[-1])
T, MP, MA = (load(p) for p in sys.argv[1:4])
for d, ph in ((T, "time"), (MP, "mem_primal"), (MA, "mem_adjoint")):
    if d.get("phase") != ph: fails.append(f"{ph}: phase field is {d.get('phase')}")
    for x in d["results"]:
        if "error" in x: fails.append(f"{ph} {x['size']}: error {x['error']}")
timing = ("primal", "primal_wall", "adjoint", "adjoint_wall")
for x in T["results"]:
    if "error" in x: continue
    if any(k.startswith("mem_") for k in x): fails.append(f"time {x['size']}: has a memory field")
    if x["size"] != "tiny" and not all(k in x for k in timing): fails.append(f"time {x['size']}: a timer is missing")
    if x["size"] == "tiny" and not (x.get("primal_relerr", 1) <= 1e-10 and x.get("grad_relerr", 1) <= 1e-10): fails.append("time tiny: CPU and device differ")
for x in MP["results"]:
    if "error" in x: continue
    if any(k in x for k in timing): fails.append(f"mem_primal {x['size']}: has a timing field")
    if x["size"] == "tiny": fails.append("mem_primal: a verify size ran")
    if abs(x["mem_inputs_mb"] - x["input_mb"]) > 1e-9: fails.append(f"mem_primal {x['size']}: inputs {x['mem_inputs_mb']} against {x['input_mb']}")
    if x["mem_peak_primal_mb"] - x["mem_inputs_mb"] > 1e-9: fails.append(f"mem_primal {x['size']}: peak {x['mem_peak_primal_mb']} is above the inputs {x['mem_inputs_mb']} (adjoint data on the device)")
for x in MA["results"]:
    if "error" in x: continue
    if any(k in x for k in timing): fails.append(f"mem_adjoint {x['size']}: has a timing field")
    if abs(x["mem_setup_mb"] - x["device_mb_estimate"]) > 1e-9: fails.append(f"mem_adjoint {x['size']}: setup {x['mem_setup_mb']} against parts {x['device_mb_estimate']}")
    if x["mem_peak_adjoint_mb"] < x["mem_setup_mb"] - 1e-9: fails.append(f"mem_adjoint {x['size']}: peak below the setup")
print("FAIL: " + "; ".join(fails) if fails else "PASS")
sys.exit(1 if fails else 0)
