module MadMPEC
using NLPModels, SolverCore
import MadNLP: regular!

include("Types.jl")
include("MappedVector.jl")
include("MPCCModelMeta.jl")
include("MPCCModel.jl")
include("ScholtesRelaxation.jl")
include("LiftedNLPModel.jl")
include("solver/types.jl")
include("solver/AbstractMPCCSolver.jl")
include("solver/HomotopySolver.jl")
include("ipoptc/regular_patch.jl")

end # module MadMPEC
