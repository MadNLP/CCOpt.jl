module MadMPEC
using NLPModels, SolverCore, LinearAlgebra, SparseArrays, HiGHS, JuMP, Gurobi
using MathOptInterface
const MOI = MathOptInterface
using Serialization
using Printf
using DataStructures: CircularBuffer
import MadNLP

include("types.jl")
include("enums.jl")
include("MPCCModelMeta.jl")
include("MPCCModel.jl")
include("MPCCRelaxationModel.jl")

include("Models/scholtes_relaxation.jl")
include("Models/fischer_burmeister_relaxation.jl")
include("Models/chen_chen_kanzow_relaxation.jl")
include("Models/natural_residual_relaxation.jl")
include("Models/branch_nlp.jl")
include("Models/l1_exact_penalty.jl")
include("Models/lifted_model.jl")

include("Solvers/AbstractMPCCSolver.jl")
include("Solvers/homotopy_solver.jl")
include("Solvers/lpcc_solver.jl")
include("Solvers/madnlpc/MadNLPC.jl")
include("Solvers/exact_penalty/ExactPenalty.jl")

end # module MadMPEC
