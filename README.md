# CCOpt
A Julia package for solving Mathematical Programs with Complementarity Constraints, MPCCs.
CCOpt contains several efficiently implemented algorithms for the special handling of the complementarity constraints found in the problem:
```
min f(x)
x, s.t
   lg <= g(x) <= ug
   lx <=  x   <= ux
   lG <= G(x) _|_ H(x) >= lH
```
It is built on top of the interior point method implemented in [`MadNLP.jl`](https://github.com/MadNLP/MadNLP.jl).
## Installation
CCOpt can be installed directly via the Julia package manager:
```julia
pkg> add CCOpt
```

## Usage
### Modeling
In order to model MPCCs, CCOpt provides it's own extention to the `AbstractNLPModel` concept from [`NLPModels.jl`](https://github.com/JuliaSmoothOptimizers/NLPModels.jl), called `MPCCModel`.
An `MPCCModel` can be constructed by wrapping an underlying `AbstractNLPModel` (here using `NLPModelsJuMP.jl`) and providing the correct complementarity indices:
```julia
using JuMP, NLPModelsJuMP, CCOpt
model = Model()
@variable(model, 0 <= x1)
@variable(model, 0 <= x2)
@objective(model, Min, 100.0 * (x2 - x1^2)^2 + (1.0 - x1)^2)
nlp = MathOptNLPModel(model)
mpcc = CCOpt.MPCCModelVarVar(nlp, [1], [2])
```
### Solvers
CCopt provides two main algorithms for solving MPCCs: `RelaxationSolver`, which relaxes the complementarity constraints, and `PenaltySolver`, which moves the complementarities into the objective.