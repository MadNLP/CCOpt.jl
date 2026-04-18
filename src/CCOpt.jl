module CCOpt
using NLPModels, SolverCore, LinearAlgebra, SparseArrays
using Serialization
using Printf
using DataStructures: CircularBuffer
import MadNLP

include("types.jl")
include("LinearModel.jl")
include("MPCCModelMeta.jl")
include("MPCCModel.jl")
include("LPCCModel.jl")
include("MPCCRelaxationModel.jl")
include("MPCCPenaltyModel.jl")

include("Models/scholtes_relaxation.jl")
include("Models/fischer_burmeister_relaxation.jl")
include("Models/chen_chen_kanzow_relaxation.jl")
include("Models/natural_residual_relaxation.jl")
include("Models/branch_nlp.jl")
include("Models/l1_exact_penalty.jl")
include("Models/lasry_lions_penalty.jl")
include("Models/lifted_model.jl")
include("Models/big_m_model.jl")

include("utils.jl")
include("Solvers/common.jl")
include("Solvers/AbstractMPCCSolver.jl")
include("Solvers/homotopy_solver.jl")
include("Solvers/relaxation/Relaxation.jl")
include("Solvers/penalty/Penalty.jl")
include("Solvers/crossover.jl")

# Modelling
export MPCCModel, MPCCModelVarVar, MPCCModelConCon, MPCCModelVarCon,# struct
    is_vertical, vertical_form #Verticality

# Function eval for MPCCs
export comp_left, comp_left!, comp_right, comp_right!,
    comp_res_left, comp_res_left!, comp_res_right, comp_res_right!,
    jac_comp_left_structure,  jac_comp_left_structure!,
    jac_comp_right_structure,  jac_comp_right_structure!,
    jac_comp_left_coord,  jac_comp_left_coord!,
    jac_comp_right_coord,  jac_comp_right_coord!,
    comp_residual, comp_residual_product,

# Solve driver
export solve_homotopy!

# Shared structs
export CCOptExecutionStats

# Relaxation Solver
export RelaxationSolver, RelaxationOptions, # Core structs
    ProportionalRelaxationUpdate, RolloffRelaxationUpdate, LOQORelaxationUpdate, # relaxation options structs
    NoEndgameStrategy, RelaxLBEndgameStrategy # endgame

# Penalty Solver
export PenaltySolver, PenaltyOptions # Core structs

# Crossover
export CrossoverOptions, CrossoverExecutionStats, # Structs
    GurobiLPCCSolverOptions, HiGHSLPCCSolverOptions, # MILP options,
    crossover # driver

end # module CCOpt
