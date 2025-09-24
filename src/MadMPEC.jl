module MadMPEC
using NLPModels, SolverCore
using LinearAlgebra, SparseArrays, SparseMatricesCOO
using HiGHS, JuMP, Gurobi, QuadraticModels
using MathOptInterface
const MOI = MathOptInterface
using Serialization
using Printf
using DataStructures: CircularBuffer
import MadNLP

include("Types.jl")
include("enums.jl")
include("models/MPCCModelMeta.jl")
include("models/MPCCModel.jl")
include("models/MPCCRelaxationModel.jl")
include("models/ScholtesRelaxation.jl")
include("models/FischerBurmeisterRelaxation.jl")
include("models/ChenChenKanzowRelaxation.jl")
include("models/NaturalResidualRelaxation.jl")
include("models/BranchNLP.jl")
include("models/Ell1Relaxation.jl")
include("models/LiftedNLPModel.jl")
include("models/LPCCModel.jl")
include("models/BigMModel.jl")
include("solver/AbstractMPCCSolver.jl")
include("solver/HomotopySolver.jl")
include("lpcc/LpccSolver.jl")
include("madnlpc/MadNLPC.jl")
include("exact_penalty/ExactPenalty.jl")

end # module MadMPEC
