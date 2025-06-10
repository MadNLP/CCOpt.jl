module MadMPEC
using NLPModels, SolverCore, LinearAlgebra
using Serialization
using DataStructures: CircularBuffer
using AmplNLReader: AmplException
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
include("madnlpc/MadNLPC.jl")
include("madnlp_ell1/MadNLP_ell1.jl")

end # module MadMPEC
