include("readlog.jl")
include("macmpec/run_macmpec.jl")
include("simple_model.jl")
include("plotting/plot_problem_iters.jl")

names, probs = load_ampl_benchmark("data/macMPEC/nls/");

x1_star=1.0
x2_star=2.0
q1 = 1000.0
q2 = 1.0
#mpcc = probs[47]
mpcc = SimpleMPCCModel(; x1=1.0, x2=2.0, q1=q1, q2=q2, x0=[0.0, 2])

#opts = MadMPEC.MadNLPCOptions(;iterates_fname="wow_multipliers.iters", kkt_regularization=:none, center_complementarities=true, centering_factor=0.9, relaxation_update=MadMPEC.RelaxLBUpdate(tau=0.45));
opts = MadMPEC.MadNLPCOptions(;
    iterates_fname="wow_multipliers.iters",
    kkt_regularization=:none,
    center_complementarities=false,
    centering_factor=0.9,
);

solver = MadMPEC.MadNLPCSolver(
    mpcc;
    solver_opts=opts,
    print_level=MadNLP.INFO,
    bound_relax_factor=0.0,
    bound_push=1e-4,
    max_iter=20,
    barrier=MadNLP.MonotoneUpdate(mu_init=1e-4, mu_min=5e-9),
);

st = MadMPEC.solve_homotopy!(solver)
ipm = solver.ipm
step = :magic
kappa = 1e5

x1 = st.solution[1]
x2 = st.solution[2]
s = MadNLP.slack(ipm.x)[1]
z1 = st.stats.multipliers_L[1]
z2 = st.stats.multipliers_L[2]
zs = ipm.y[1]
mu = ipm.mu
delta = kappa*mu

r1 = -2*q1*(x1-x1_star)*ipm.cb.obj_scale[]
r2 = -2*q2*(x2-x2_star)*ipm.cb.obj_scale[]

if step == :magic
    z1_hat = inv(x1+delta)*(mu)
    zs_hat = inv(x2)*(r1 + z1_hat)
    z2_hat = x1*zs_hat - r2
elseif step == :halfmagic
    z1_hat = inv(x1+delta)*(mu)
    zs_hat = zs
    z2_hat = z2
else
    step == :none
    z1_hat = z1
    zs_hat = zs
    z2_hat = z2
end
println("Calculations:")
println("z1: $(z1), z1_hat: $(z1_hat)")
println("z2: $(z2), z2_hat: $(z2_hat)")
println("zs: $(zs), zs_hat: $(zs_hat)")
println()

println("MPEC multipliers")
println("old z1 - x2*zs: $(z1 - x2*zs)")
println("new z1 - x2*zs: $(z1_hat - x2*zs_hat)")
println("old z2 - x1*zs: $(z2 - x1*zs)")
println("new z2 - x1*zs: $(z2_hat - x1*zs_hat)")
println()

println("New complementarity residuals")
println("(x1+δ)*z1_hat - μ = $((x1+delta)*z1_hat - mu)")
println("x2*z2_hat - μ = $(x2*z2_hat - mu)")
println("s*zs_hat - μ = $(-s*zs_hat - mu)")

println("New dual residuals")
println("∇L1 = $(-r1 + x2*zs_hat - z1_hat)")
println("∇L2 = $(-r2 + x1*zs_hat - z2_hat)")
