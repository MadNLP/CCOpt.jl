include("macmpec/run_macmpec.jl")
using CUDA, MadNLPGPU

idx = 7

names, probs = load_ampl_benchmark("data/macMPEC/nls/");
println(names[idx])
gpu_nlp = MadNLP.SparseWrapperModel(CuArray, probs[idx].nlp)
gpu_mpcc =
    MadMPEC.MPCCModelVarVar(gpu_nlp, get_ind_cc1(probs[idx]), get_ind_cc2(probs[idx]))

opts = MadMPEC.MadNLPCOptions(;
    center_complementarities=false,
    sigma_min=1e-8,
    respect_comp_bounds=false,
)
solver = MadMPEC.MadNLPCSolver(
    gpu_mpcc;
    solver_opts=opts,
    print_level=MadNLP.INFO,
    bound_relax_factor=1e-8,
    bound_push=1e-2,
    max_iter=3000,
    linear_solver=CUDSSSolver,
    equality_treatment=MadNLP.RelaxEquality,
    tol=1e-6,
)
st=MadMPEC.solve_homotopy!(solver)
