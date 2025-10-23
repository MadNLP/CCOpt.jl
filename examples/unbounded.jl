using NLPModels, ADNLPModels, MadMPEC

f(x) = x[1]*x[3]
x0 = [1.0, 1.0, 1.0];
ind_vcc1 = [1];
ind_vcc2 = [2];
ind_x = Vector{Int}();
lvar_vv = [0.0, 0.0, 0.0]
uvar_vv = [Inf, Inf, Inf]

nlp_vv = ADNLPModels.ADNLPModel(f, x0, lvar_vv, uvar_vv)

mpcc_vv = MPCCModelVarVar(nlp_vv, ind_vcc1, ind_vcc2)

opts = MadMPEC.ExactPenaltyOptions(
    dynamic_tau_update=false,
    kkt_regularization=:eigenvalue_decomposition,
)
solver = MadMPEC.ExactPenaltySolver(
    probs[10];
    solver_opts=opts,
    bound_relax_factor=0.0,
    barrier=MadNLP.MonotoneUpdate(),
)
