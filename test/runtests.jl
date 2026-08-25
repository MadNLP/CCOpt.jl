using LinearAlgebra, SparseArrays, Test
using NLPModels, MPCCModels, ADNLPModels, CCOpt
using MadNLP

include("models/simple-model.jl")
include("models/simple-mpcc-model.jl")
include("models/scholtes-relaxation.jl")
include("solvers/homotopy-solver.jl")
include("solvers/relaxation.jl")
include("solvers/penalty.jl")

if "MOI" in ARGS
    # only run MOI if requested!
    @testset "MOI interface" begin
        include("MOI_wrapper.jl")
    end
end
