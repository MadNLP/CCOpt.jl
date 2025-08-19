#iters = 229
#iters = 270
#iters = 286
#iters = 500
iters = 3000
idx_prob = 173

opts = MadMPEC.MadNLPCOptions(; print_level=MadNLP.ERROR, reset_slacks_on_update=true);
solver = MadMPEC.MadNLPCSolver(
    probs[idx_prob];
    solver_opts=opts,
    print_level=MadNLP.INFO,
    bound_relax_factor=0.0,
    max_iter=iters,
    tol=1e-8,
    barrier=MadNLP.MonotoneUpdate(),
);

stat = MadMPEC.solve_homotopy!(solver)
mpcc = solver.mpcc
ind_cc1 = mpcc.meta.ind_cc1
ind_cc2 = mpcc.meta.ind_cc2
ncc=length(ind_cc1)

x = MadNLP.variable(solver.ipm.x)
zx = MadNLP.variable(solver.ipm.zl)
x1 = x[ind_cc1]
x2 = x[ind_cc2]
z1 = zx[ind_cc1]
z2 = zx[ind_cc2]
s = MadNLP.slack(solver.ipm.x)[(end-ncc+1):end]
zs = MadNLP.slack(solver.ipm.zu)[(end-ncc+1):end]
hcat(x1, x2, z1, z2, s, zs)
