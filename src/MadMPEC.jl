module MadMPEC
using NLPModels, SolverCore, LinearAlgebra
using Serialization
import MadNLP

include("Types.jl")
include("MappedVector.jl")
include("MPCCModelMeta.jl")
include("MPCCModel.jl")
include("ScholtesRelaxation.jl")
include("LiftedNLPModel.jl")
include("solver/types.jl")
include("solver/AbstractMPCCSolver.jl")
include("solver/HomotopySolver.jl")
include("madnlpc/MadNLPC.jl")

end # module MadMPEC
