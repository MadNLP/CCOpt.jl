using NLPModels, Profile, ProfileView
includet("./macmpec/run_macmpec.jl")
names, probs = load_ampl_benchmark("data/macMPEC/nls/")
opts = MadMPEC.MadNLPCOptions(;
    use_mpecopt=true,
    lpcc_solver_opts=MadMPEC.MadNLPCOptions(; print_level=MadNLP.INFO),
);
solver = MadMPEC.MadNLPCSolver(
    probs[100];
    solver_opts=opts,
    print_level=MadNLP.INFO,
    bound_relax_factor=0.0,
    max_iter=3000,
    tol=1e-8,
    barrier=MadNLP.MonotoneUpdate(),
);
st = MadMPEC.solve_homotopy!(solver)
c = Vector{Float64}(undef, solver.lpcc_solver.rnlp.meta.ncon)
@profview for i in 1:100
    cons!(solver.lpcc_solver.rnlp, 10*solver.lpcc.meta.x0, c)
end
