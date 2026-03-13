using CCOpt, MadNLP, ADNLPModels
function SimpleMPCCModel(T::Type)
    f(x) = (x[1] - 1)^2 + (x[2] - 1)^2
    x0 = T[2.0, 1];
    ind_vcc1 = [1];
    ind_vcc2 = [2];
    ind_x = Vector{Int}();
    lvar_vv = T[0.0, 0.0]
    uvar_vv = T[Inf, Inf]

    nlp_vv = ADNLPModels.ADNLPModel(f, x0, lvar_vv, uvar_vv)

    # Test MPCCVarVar
    return CCOpt.MPCCModelVarVar(nlp_vv, ind_vcc1, ind_vcc2)
end

mpcc = SimpleMPCCModel(Float64)
madnlpc_opts = CCOpt.MadNLPCOptions(; use_magic_step=true, plot_iterates=true)
solver = CCOpt.MadNLPCSolver(
    mpcc;
    madnlpc_opts=madnlpc_opts,
    print_level=MadNLP.INFO,
    bound_relax_factor=1e-10,
)
CCOpt.solve_homotopy!(solver)
