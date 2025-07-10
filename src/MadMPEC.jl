module MadMPEC
using NLPModels, SolverCore, LinearAlgebra, SparseArrays, HiGHS
using Printf
using Serialization
using DataStructures: CircularBuffer
import MadNLP

include("Types.jl")
include("enums.jl")
include("MPCCModelMeta.jl")
include("MPCCModel.jl")
include("ScholtesRelaxation.jl")
include("BranchNLP.jl")
include("models/Ell1Relaxation.jl")
include("LiftedNLPModel.jl")
include("solver/AbstractMPCCSolver.jl")
include("solver/HomotopySolver.jl")
include("lpcc/LpccSolver.jl")
include("madnlpc/MadNLPC.jl")
include("exact_penalty/ExactPenalty.jl")

end # module MadMPEC
