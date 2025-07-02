module MadMPEC
using NLPModels, SolverCore, LinearAlgebra, SparseArrays
using Serialization
using DataStructures: CircularBuffer
import MadNLP

include("Types.jl")
include("MappedVector.jl")
include("MPCCModelMeta.jl")
include("MPCCModel.jl")
include("ScholtesRelaxation.jl")
include("models/Ell1Relaxation.jl")
include("LiftedNLPModel.jl")
include("solver/types.jl")
include("solver/AbstractMPCCSolver.jl")
include("solver/HomotopySolver.jl")
include("lpcc/LpccSolver.jl")
include("madnlpc/MadNLPC.jl")
include("exact_penalty/ExactPenalty.jl")

end # module MadMPEC
