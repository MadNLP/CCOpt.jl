#! /usr/bin/which julia

using NLPModels,ADNLPModels, MadMPEC

f(x) = (x[1] - 1)^2 + (x[2] + 1)^2
x0 = [1.,1.];
ind_vcc1 = [1];
ind_vcc2 = [2];
ind_x = Vector{Int}();
lvar_vv = [0.,0.]
uvar_vv = [Inf,Inf]

nlp_vv = ADNLPModels.ADNLPModel(f, [1.,1.0], lvar_vv, uvar_vv)

# Test MPCCVarVar
mpcc_vv = MPCCModelVarVar(nlp_vv, ind_vcc1, ind_vcc2)

# Test MPCCVarCon
c_vc(x) = [x[2]]
lvar_vc = [0.,-Inf]
uvar_vc = [Inf, Inf]
lcon_vc = [0.];
ucon_vc = [Inf];
ind_ccc2 = [1];

nlp_vc = ADNLPModels.ADNLPModel(f, [1.,1.0], lvar_vc, uvar_vc, c_vc, lcon_vc, ucon_vc)

mpcc_vc = MPCCModelVarCon(nlp_vc, ind_vcc1, ind_ccc2)
# Test MPCCConCon
c_cc(x) = [x[2];x[1]]
lvar_cc = [-Inf,-Inf]
uvar_cc = [Inf, Inf]
lcon_cc = [0., 0.];
ucon_cc = [Inf, Inf];
ind_ccc1 = [2];

nlp_cc = ADNLPModels.ADNLPModel(f, [1.,1.0], lvar_cc, uvar_cc, c_cc, lcon_cc, ucon_cc)

mpcc_cc = MPCCModelConCon(nlp_cc, ind_ccc1, ind_ccc2)
#vmpcc = MadMPEC.MPCCModelVerticalForm(mpcc);

cons(mpcc_cc, [1.0, 1.0])

# Test Scholtes relaxation
scholtes = ScholtesRelaxation(mpcc_vv)

#cons(scholtes, [1.,1.])
jac_structure(scholtes)
jac_coord(scholtes, [1.,1.])

jprod(scholtes, [1.,1.],[1.,1.])

jtprod(scholtes, [1.,2.], [1.])

hess(scholtes, [1.,1.])

hess(scholtes, [1.,1.], [1.])

hprod(scholtes, [1.,1.], [1.], [1.,1.])

lifted_mpcc_cc = vertical_form(mpcc_cc)

vscholtes = ScholtesRelaxation(lifted_mpcc_cc)
