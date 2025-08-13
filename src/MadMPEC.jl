module MadMPEC
using NLPModels, SolverCore, LinearAlgebra, SparseArrays, HiGHS, JuMP, Gurobi
using MathOptInterface
const MOI = MathOptInterface
using Serialization
using DataStructures: CircularBuffer
import MadNLP

include("Types.jl")
include("enums.jl")
include("MappedVector.jl")
include("MPCCModelMeta.jl")
include("MPCCModel.jl")
include("MPCCRelaxationModel.jl")
include("ScholtesRelaxation.jl")
include("FischerBurmeisterRelaxation.jl")
include("ChenChenKanzowRelaxation.jl")
include("NaturalResidualRelaxation.jl")
include("BranchNLP.jl")
include("models/Ell1Relaxation.jl")
include("LiftedNLPModel.jl")
include("solver/AbstractMPCCSolver.jl")
include("solver/HomotopySolver.jl")
include("lpcc/LpccSolver.jl")
include("madnlpc/MadNLPC.jl")
include("exact_penalty/ExactPenalty.jl")

end # module MadMPEC
