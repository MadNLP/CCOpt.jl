using MadMPEC
using ADNLPModels
function SimpleMPCCModel(T::Type=Float64; x1=1, x2=1)
    f(x) = (x[1] - x1)^2 + (x[2] - x2)^2
    x0 = T[0.0, 0.0];
    ind_vcc1 = [1];
    ind_vcc2 = [2];
    ind_x = Vector{Int}();
    lvar_vv = T[0.0, 0.0]
    uvar_vv = T[Inf, Inf]

    nlp_vv = ADNLPModels.ADNLPModel(f, x0, lvar_vv, uvar_vv)

    # Test MPCCVarVar
    return MadMPEC.MPCCModelVarVar(nlp_vv, ind_vcc1, ind_vcc2)
end
