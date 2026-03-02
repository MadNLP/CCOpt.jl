######################### Type Definitions #########################
# Typealias for index set
# TODO(@anton) Perhaps this should be Set{Int}
const IndexSet = Vector{Int}

# abstract type CCType end
# struct VarCon <: CCType end
# struct VarVar <: CCType end
# struct ConCon <: CCType end

# TODO(@anton) this is probably how we should do it rather than with the structs above. refactor.
@enum CCType::Clonglong begin
    VarVar=Clonglong(0)
    VarCon=Clonglong(1)
    ConVar=Clonglong(2)
    ConCon=Clonglong(3)
end
