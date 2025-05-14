#! /usr/bin/which julia

using NLPModels,ADNLPModels, MadMPEC

f(x) = (x[1] - 1)^2 + (x[2] + 1)^2
x0 = [1.,1.];

# Test MPCCConCon
c_cc(x) = [x[2]^2;x[1]^2]
lvar_cc = [-Inf,-Inf]
uvar_cc = [Inf, Inf]
lcon_cc = [0., 0.];
ucon_cc = [Inf, Inf];

nlp_cc = ADNLPModels.ADNLPModel(f, [1.,1.0], lvar_cc, uvar_cc, c_cc, lcon_cc, ucon_cc)

lifted = LiftedNLPModel(nlp_cc, [1,2])

cons(lifted, [2.,2.,0.,0.])
