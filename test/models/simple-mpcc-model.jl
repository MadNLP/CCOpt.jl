"""
    SimpleMPCCModel <: AbstractMPCCModel

Simple model for testing purposes.

     min   (x₁ - 1)² + (x₂ - 1)²
     s.t   0 ≤ x₁ ⟂ x₂ ≥ 0

x₀ = [2.0, 2.0].
"""
function SimpleMPCCModel(T::Type)
    f(x) = (x[1] - 1)^2 + (x[2] - 1)^2
    x0 = T[2.0, 2.0];
    ind_vcc1 = [1];
    ind_vcc2 = [2];
    ind_x = Vector{Int}();
    lvar_vv = T[0.0, 0.0]
    uvar_vv = T[Inf, Inf]

    nlp_vv = ADNLPModels.ADNLPModel(f, x0, lvar_vv, uvar_vv)

    # Test MPCCVarVar
    return MPCCModel(nlp_vv, ind_vcc1, ind_vcc2)
end

"""
    SimpleMPCCModel <: AbstractMPCCModel

Simple(r) model for testing purposes.

     min   (x₁ - 0.1)² + (x₂ - 1)²
     s.t   0 ≤ x₁ ⟂ x₂ ≥ 0

x₀ = [2.0, 2.0].
"""
function SimpleMPCCModel2(T::Type)
    f(x) = (x[1] - 0.1)^2 + (x[2] - 1)^2
    x0 = T[2.0, 2.0];
    ind_vcc1 = [1];
    ind_vcc2 = [2];
    ind_x = Vector{Int}();
    lvar_vv = T[0.0, 0.0]
    uvar_vv = T[Inf, Inf]

    nlp_vv = ADNLPModels.ADNLPModel(f, x0, lvar_vv, uvar_vv)

    # Test MPCCVarVar
    return MPCCModel(nlp_vv, ind_vcc1, ind_vcc2)
end
