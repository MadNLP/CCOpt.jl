module MadMPEC
using NLPModels

include("Types.jl")
include("MappedVector.jl")
include("MPCCModelMeta.jl")
include("MPCCModel.jl")
include("ScholtesRelaxation.jl")
include("LiftedNLPModel.jl")
include("solver/AbstractMPCCSolver.jl")
include("solver/HomotopySolver.jl")

end # module MadMPEC
