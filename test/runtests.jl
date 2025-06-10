using LinearAlgebra, SparseArrays, Test
using NLPModels, ADNLPModels, MadMPEC
using MadNLP, NLPModelsIpopt

include("models/simple-model.jl")
include("models/lifted-nlp-model.jl")
include("models/simple-mpcc-model.jl")
include("models/scholtes-relaxation.jl")
include("solvers/homotopy-solver.jl")
include("solvers/madnlp-c.jl")
include("solvers/madnlp-ell1.jl")
