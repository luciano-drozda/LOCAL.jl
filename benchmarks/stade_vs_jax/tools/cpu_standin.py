"""Turns a generated STADE runner script (KERNEL.cuda.f1.jl or KERNEL_C.jl) into a script that runs on the CPU.
Usage: python3 cpu_standin.py IN.jl OUT.jl
It removes the GPU code and puts tools/cpu_standin.jl in its place. The device functions become the CPU functions."""
import os, re, sys
s = open(sys.argv[1]).read()
a = s.index("using CUDA\n"); b = s.index("# ---- CPU primal and adjoint"); c = s.index("# ---- inputs ----")
name = re.search(r'const NAME = "([^"]+)"', s).group(1)
pre = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpu_standin.jl")).read()
out = s[:a] + pre + "\nusing .CUDA\nusing LinearAlgebra\n\n" + s[b:c] + f'\nstandin_alias!("{name}")\n\n' + s[c:]
open(sys.argv[2], "w").write(out)
