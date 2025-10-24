include("common.jl")

problem_path = "data/macMPEC/nls/bem-milanc30-s.nl"
model = AmplNLReader.AmplModel(problem_path)
mpcc = MadMPEC.vertical_form(mpcc_from_ampl(model))

opts1 = MadMPEC.ExactPenaltyOptions(kkt_regularization=:eigenvalue_decomposition);
solver1 = MadMPEC.ExactPenaltySolver(
    mpcc;
    solver_opts=opts1,
    bound_relax_factor=0.0,
    barrier=MadNLP.MonotoneUpdate(),
    max_iter=1000,
    print_level=MadNLP.INFO,
);
st1 = MadMPEC.solve_homotopy!(solver1)

opts2 = MadMPEC.ExactPenaltyOptions(kkt_regularization=:eigenvalue_decomposition);
solver2 = MadMPEC.ExactPenaltySolver(
    mpcc;
    solver_opts=opts2,
    bound_relax_factor=0.0,
    max_iter=1000,
    print_level=MadNLP.INFO,
);
st2 = MadMPEC.solve_homotopy!(solver2)
