
using JuMP

#=
    Collocation scheme.
=#

"""
    RKScheme

Store Butcher tableau used in the collocation.

"""
struct RKScheme
    c::Vector{Float64}
    b::Vector{Float64}
    a::Matrix{Float64}
end

function GaussLegendre()
    p = √3/6
    return RKScheme([0.5 - p, 0.5 + p], [0.5, 0.5], [0.25 0.25-p; 0.25+p 0.25])
end
ImplicitEuler() = RKScheme([1.0], [1.0], ones(1, 1))
CrankNicolson() = RKScheme([0.0, 1.0], [0.5, 0.5], [0.0 0.0; 0.5 0.5])


#=
    Parser for JuMP problems with complementarity constraints.
=#

function _is_single_variable(term::Vector{MOI.ScalarAffineTerm{T}}, c::T) where T
    return length(term) == 1 && term[1].coefficient == 1.0 && c == 0.0
end

function _vertical_formulation!(model, terms::Vector{MOI.ScalarAffineTerm{T}}, c::T) where T
    x = MOI.add_variable(model)
    # Add constraint x >= 0
    MOI.add_constraint(model, x, MOI.GreaterThan{T}(zero(T)))
    push!(terms, MOI.ScalarAffineTerm{T}(-one(T), x))

    func = MOI.ScalarAffineFunction{T}(terms, zero(T))
    MOI.add_constraint(model, func, MOI.EqualTo{T}(-c))
    return x.value
end

function parse_ccons!(model)
    moimodel = JuMP.backend(model)
    ind_cc1, ind_cc2 = Int[], Int[]

    contypes = MOI.get(moimodel, MOI.ListOfConstraintTypesPresent())
    for (F, S) in contypes
        # Parse only complementarity constraints
        if S == MOI.Complements
            conindices = MOI.get(moimodel, MOI.ListOfConstraintIndices{F, S}())
            for cidx in conindices
                fun = MOI.get(moimodel, MOI.ConstraintFunction(), cidx)
                set = MOI.get(moimodel, MOI.ConstraintSet(), cidx)
                n_comp = div(set.dimension, 2)
                if isa(fun, MOI.VectorOfVariables)
                    indv = [v.value for v in fun.variables]
                    append!(ind_cc1, indv[1:n_comp])
                    append!(ind_cc2, indv[n_comp+1:end])
                elseif isa(fun, MOI.VectorAffineFunction)
                    # Parse all affine terms in `fun`
                    expr_cc = [MOI.ScalarAffineTerm{Float64}[] for i in 1:2*n_comp]
                    for i in eachindex(fun.terms)
                        term = fun.terms[i]
                        push!(expr_cc[term.output_index], term.scalar_term)
                    end

                    # Read each complementarity constraint and get corresponding indices
                    for i in 1:n_comp
                        t1 = expr_cc[i]
                        c1 = fun.constants[i]
                        t2 = expr_cc[i + n_comp]
                        c2 = fun.constants[i + n_comp]
                        # If the variable is isolated, we don't reformulate the
                        # complementarity constraint using a slack
                        isvar1 = _is_single_variable(t1, c1)
                        isvar2 = _is_single_variable(t2, c2)
                        if isvar1 && !isvar2
                            # Reformulate only RHS using vertical form
                            ind = _vertical_formulation!(moimodel, t2, c2)
                            push!(ind_cc1, t1[1].variable.value)
                            push!(ind_cc2, ind)
                        elseif !isvar1 && isvar2
                            # Reformulate only LHS using vertical form
                            ind = _vertical_formulation!(moimodel, t1, c1)
                            push!(ind_cc1, t2[1].variable.value)
                            push!(ind_cc2, ind)
                        elseif !isvar1 && !isvar2
                            # Reformulate LHS and RHS
                            ind1 = _vertical_formulation!(moimodel, t1, c1)
                            ind2 = _vertical_formulation!(moimodel, t2, c2)
                            push!(ind_cc1, ind1)
                            push!(ind_cc2, ind2)
                        else
                            push!(ind_cc1, t1[1].variable.value)
                            push!(ind_cc2, t2[1].variable.value)
                        end
                    end
                else
                    error("Complementary constraints formulated with $(typeof(fun)) are not yet supported")
                end
                # We delete the complementarity constraints before passing them to NLPModelsJuMP
                MOI.delete(moimodel, cidx)
            end
        end
    end
    return ind_cc1, ind_cc2
end

