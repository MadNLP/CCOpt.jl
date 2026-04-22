# CCOpt

A Julia package for solving Mathematical Programs with Complementarity Constraints (MPCCs).
For details check out the [implementation paper](https://arxiv.org/abs/2604.18726).

## Installation

To install CCOpt, simply proceed to
```julia
pkg> add https://github.com/MadNLP/CCOpt.jl
```

## Usage

CCOpt takes as input a nonlinear program formulated with [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl/). Taking a `nlp` as input, a MPCC is defined as
```julia
using CCOpt
mpcc = CCOpt.MPCCModelVarVar(nlp, ind_x1, ind_x2)
```
with `ind_x1` (resp. `ind_x2`) the indices of the variables appearing in the left-hand complementarity
(resp. right-hand complementarity).

### Relaxation method

Once specified, you can solve the MPCC problem implemented in `mpcc` using the relaxation method as
```julia
solver = CCOpt.RelaxationSolver(mpcc)
stats = CCOpt.solve_homotopy!(solver)
```
All the results (primal and dual solutions, objective, etc.) are stored in `stats`.

### Penalty method

Alternatively, you can solve `mpcc` using the penalty method as
```julia

solver = CCOpt.PenaltySolver(mpcc)
stats = CCOpt.solve_homotopy!(solver)
```

## Citation

If you use `CCOpt.jl` in your work, please cite:

```bibtex
@article{Pozharskiy2026,
  title={CCOpt: an Open-Source Solver for Large-Scale Mathematical Programs with Complementarity Constraints},
  author={Pozharskiy, Anton and Pacaud, Fran{\c{c}}ois and Diehl, Moritz and Nurkanovi{\'c}, Armin},
  journal={arXiv preprint arXiv:2604.18726},
  year={2026}
}