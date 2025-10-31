######################### Type Definitions #########################
# Typealias for index set
# TODO(@anton) Perhaps this should be Set{Int}
const IndexSet = Vector{Int}

abstract type CCType end
struct VarCon <: CCType end
struct VarVar <: CCType end
struct ConCon <: CCType end
