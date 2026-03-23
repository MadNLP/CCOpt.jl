using ADNLPModels
using PolynomialRoots
# TODO(@anton) This is 100% inefficient but fine for testing
function magic_step_nlp(w_old, mu, tau, eta)
    f(x) = 0.5*((x[1]-w_old[1])^2 + (x[2] - w_old[2])^2)# + 1e-6*(x[4])^2 + 1e-6*(x[5])^2 + 1e-6*(x[6])^2)
    #f(x) = 0.5*(sum((x.-w_old).^2))
    c(x) = [
        x[1]*x[2] - tau + x[3],
        x[1]*x[4] - mu,
        x[2]*x[5] - mu,
        x[3]*x[6] - mu,
        x[5] - x[6]*x[1] - eta,
    ]
    x0 = w_old
    lvar = zeros(6);
    uvar = Inf*ones(6);
    lcon = zeros(5);
    ucon = zeros(5);
    nlp = ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon)
    sol = MadNLP.madnlp(
        nlp;
        print_level=MadNLP.ERROR,
        bound_push=1e-8,
        barrier=MadNLP.MonotoneUpdate(mu_init=1e-6),
    )
    println("iter = $(sol.iter) status = $(sol.status)")
    return sol.solution
end

function magic_cubic(mu, tau, zeta1, zeta2)
    d = mu*mu*zeta1*zeta2
    c = -2*mu*mu*zeta2
    b = -tau*zeta2*zeta2
    a = tau*zeta2*zeta2/(zeta1)
    z1_candidates = roots([d, c, b, a])

    candidate_solutions = []

    # TODO(@anton) do checks here.
    z1 = real(z1_candidates[1])
    x1 = mu/z1
    zs = z1*(zeta2*z1/(mu*zeta1) - zeta2/mu)
    s = mu/zs
    x2 = (tau-s)/(x1)
    z2 = mu/x2
    push!(candidate_solutions, (x1=x1, x2=x2, s=s, z1=z1, z2=z2, zs=zs))

    z1 = real(z1_candidates[2])
    x1 = mu/z1
    zs = z1*(zeta2*z1/(mu*zeta1) - zeta2/mu)
    s = mu/zs
    x2 = (tau-s)/(x1)
    z2 = mu/x2
    push!(candidate_solutions, (x1=x1, x2=x2, s=s, z1=z1, z2=z2, zs=zs))

    z1 = real(z1_candidates[3])
    x1 = mu/z1
    zs = z1*(zeta2*z1/(mu*zeta1) - zeta2/mu)
    s = mu/zs
    x2 = (tau-s)/(x1)
    z2 = mu/x2
    push!(candidate_solutions, (x1=x1, x2=x2, s=s, z1=z1, z2=z2, zs=zs))
    return candidate_solutions
end

function take_magic_step!(solver::RelaxationSolver{T}) where {T}
    ncc = get_ncc(solver.mpcc)
    ipm = solver.ipm
    if get_relaxation(solver.rnlp)[1] < 1e-10
        return
    end
    for ii in 1:ncc
        cc1 = solver.ind_cc1[ii]
        cc2 = solver.ind_cc2[ii]
        x1 = MadNLP.variable(ipm.x)[cc1] - MadNLP.variable(ipm.xl)[cc1]
        x2 = MadNLP.variable(ipm.x)[cc2] - MadNLP.variable(ipm.xl)[cc2]
        z1 = MadNLP.variable(ipm.zl)[cc1]
        z2 = MadNLP.variable(ipm.zl)[cc2]
        s = -MadNLP.slack(ipm.x)[end-ncc+ii]
        zs = MadNLP.slack(ipm.zu)[end-ncc+ii]

        f1 = MadNLP.variable(ipm.f)[cc1]
        f2 = MadNLP.variable(ipm.f)[cc2]
        jac1 = ipm.jacl[cc1]
        jac2 = ipm.jacl[cc2]

        eta1 = z2 - zs*x1
        eta2 = z1 - zs*x2
        println("eta1 = $(eta1), eta2 = $(eta2)")
        println("eta1_alt = $(f2 + jac2 - zs*x1), eta2_alt = $(f1 + jac1 - zs*x2)")

        res = x1*x2 - get_relaxation(solver.rnlp)[ii]

        if res < 0
            continue
        end

        if abs(eta1) > abs(eta2)
            mu = solver.ipm.mu
            w_old = [x1, x2, mu, z1, z2, zs]
            tau = get_relaxation(solver.rnlp)[ii]

            sol = magic_step_nlp(w_old, mu, tau, eta1)
        else
            w_old = [x2, x1, s, z2, z1, zs]
            mu = solver.ipm.mu
            tau = get_relaxation(solver.rnlp)[ii]

            sol = magic_step_nlp(w_old, mu, tau, eta2)
            w_old[:] = w_old[[2, 1, 3, 5, 4, 6]]
            sol[:] = @view sol[[2, 1, 3, 5, 4, 6]]
        end

        # Save back:
        # println("sd_old = $(MadNLP.get_sd(ipm.y, ipm.zl_r, ipm.zu_r, T(ipm.opt.s_max)))")
        # println("x1_old = $(MadNLP.variable(ipm.x)[cc1]) x1_new = $(MadNLP.variable(ipm.xl)[cc1] + sol[1])")
        # println("x2_old = $(MadNLP.variable(ipm.x)[cc2]) x2_new = $(MadNLP.variable(ipm.xl)[cc2] + sol[2])")
        # println("s_old = $(MadNLP.slack(ipm.x)[end-ncc+ii]) s_new = $(-sol[3])")
        # println("z1_old = $(MadNLP.variable(ipm.zl)[cc1]) z1_new = $(sol[4])")
        # println("z2_old = $(MadNLP.variable(ipm.zl)[cc2]) z2_new = $(sol[5])")
        # println("zs_old = $(MadNLP.slack(ipm.zu)[end-ncc+ii]) zs_new = $(sol[6])")
        MadNLP.variable(ipm.x)[cc1] = MadNLP.variable(ipm.xl)[cc1] + sol[1]
        MadNLP.variable(ipm.x)[cc2] = MadNLP.variable(ipm.xl)[cc2] + sol[2]
        MadNLP.variable(ipm.zl)[cc1] = sol[4]
        MadNLP.variable(ipm.zl)[cc2] = sol[5]
        MadNLP.slack(ipm.x)[end-ncc+ii] = -sol[3]
        MadNLP.slack(ipm.zu)[end-ncc+ii] = sol[6]
        ipm.y[end-ncc+ii] = sol[6]
        # Re-eval everything out of paranoia

        # println("sd_new = $(MadNLP.get_sd(ipm.y, ipm.zl_r, ipm.zu_r, T(ipm.opt.s_max)))")
        # println(ipm.inf_du)
        # println(MadNLP.get_inf_du(
        #     MadNLP.full(ipm.f),
        #     MadNLP.full(ipm.zl),
        #     MadNLP.full(ipm.zu),
        #     ipm.jacl,
        #     MadNLP.get_sd(ipm.y, ipm.zl_r, ipm.zu_r, T(ipm.opt.s_max)),
        # ))

        # println("f1_old = $(f1) f1_new = $(MadNLP.variable(ipm.f)[cc1])")
        # println("f2_old = $(f2) f2_new = $(MadNLP.variable(ipm.f)[cc2])")
        # println("jac1_old = $(jac1) jac1_new = $(ipm.jacl[cc1])")
        # println("jac2_old = $(jac2) jac2_new = $(ipm.jacl[cc2])")
        # f1 = MadNLP.variable(ipm.f)[cc1]
        # f2 = MadNLP.variable(ipm.f)[cc2]
        # jac1 = ipm.jacl[cc1]
        # jac2 = ipm.jacl[cc2]

        #@printf("mu: %1.4e, tau: %1.4e, eta: %1.4e\n", mu, tau, eta)
        println(
            "----------------------------------------------------------------------------------",
        )
        println("w_old[$(ii)]: $(w_old)")
        println("w_new[$(ii)]: $(sol)")
        println(
            "----------------------------------------------------------------------------------",
        )
    end
    MadNLP.eval_cons_wrapper!(ipm, ipm.c, ipm.x)
    MadNLP.eval_f_wrapper(ipm, ipm.x)
    MadNLP.eval_grad_f_wrapper!(ipm, ipm.f, ipm.x)
    MadNLP.eval_jac_wrapper!(ipm, ipm.kkt, ipm.x)
    MadNLP.eval_lag_hess_wrapper!(ipm, ipm.kkt, ipm.x, ipm.y)
    return MadNLP.jtprod!(ipm.jacl, ipm.kkt, ipm.y)
end
