using LinearAlgebra, SparseArrays, Test
using NLPModels, ADNLPModels, MadMPEC

include("models/simple-model.jl")
include("models/lifted-nlp-model.jl")
include("models/simple-mpcc-model.jl")
include("models/scholtes-relaxation.jl")
