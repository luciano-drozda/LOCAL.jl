"""Builds the timing-only experiment of job ATOM2: a copy of a generated STADE runner script in which every atomic add is a plain add.
Usage: python3 patch_noatomic.py IN.jl OUT.jl
The result of the adjoint is WRONG (lost updates, races on shared memory). Use the script only to time the kernels without atomic adds: the time is
a lower bound for the time of any version that keeps the atomic adds, and it can be too low if the compiler keeps a sum in a register."""
import sys
src = open(sys.argv[1]).read()
n = src.count("CUDA.@atomic ")
assert n > 0, "no atomic add in the script"
open(sys.argv[2], "w").write(src.replace("CUDA.@atomic ", ""))
print(n, "atomic adds replaced")
