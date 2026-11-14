# LOCAL.jl

**A fork of [STADE.jl](https://github.com/cerfacs/STADE.jl/)**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22061808.svg)](https://doi.org/10.5281/zenodo.22061808)

## Public API

- `stade_tangent/adjoint/hvp_file(in_path::String, out_path::String; ...)`: writes to `out_path` a Julia source result of differentiation by STADE of the (multi-)kernel file at `in_path`. In the multi-kernel case, inlining is performed in the kernel which is not called by any other one in the file (named root kernel). The inlined root kernel is the one subject to differentiation.

- `stade_cuda/amdgpu/metal/jacc_file(in_path::String, out_path::String; ...)`: writes to `out_path` a Julia source result of GPU porting by STADE of the (multi-)kernel file at `in_path`.

- `stade_ddp_file(in_path::String, out_path::String; ...)`: writes to `out_path` a Julia source epilogue that runs the kernel at `in_path` data-parallel over a mini-batch with GPU-aware MPI.

## Wishlist 💡

- [x] ✅ [v0.2.2] Wrap common subexpressions into auxiliary variables
- [x] ✅ [v0.5.0] Add `ddpgen_` stage for data-parallel mini-batch execution code via GPU-aware MPI
- [x] ✅ [v0.4.3] Add support to `:while` statement
- [ ] Add support to efficient differentiation of fixed-point loops 
- [ ] Benchmark against mainstream frameworks (e.g., JAX, PyTorch)

### Acknowledgements

This project has been mainly inspired by [Inria Tapenade AD engine](https://team.inria.fr/ecuador/en/tapenade/) and received funding support from [ANITI EXPLEARTH](https://aniti.univ-toulouse.fr/en/explainable-and-physics-informed-ai-for-regional-weatherprediction/), [ROSAS Horizon Europe](https://www.rosas-project.eu/), and [PHLUSIM ANR](https://anr.fr/Project-ANR-23-CE23-0025).

#### If you use or build upon this software, please cite us:

```bibtex
@software{drozda_STADE,
  author       = {Drozda, Luciano},
  title        = {STADE.jl: Source Transformation Automatic Differentiation Engine},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {https://doi.org/10.5281/zenodo.22061808},
  url          = {https://github.com/cerfacs/STADE.jl},
}
```
