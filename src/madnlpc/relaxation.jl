function update_sigma!(
    relax::ProportionalRelaxationUpdate{T},
    solver::MadNLPCSolver{T},
) where {T}
    if relax.monotone
        solver.scholtes.𝜎[] = max(
            min(solver.scholtes.𝜎[], relax.sigma_mu_ratio*solver.ipm.mu),
            solver.opts.sigma_min,
        )
    else
        solver.scholtes.𝜎[] = relax.sigma_mu_ratio*solver.ipm.mu
    end
    # Here we assume the barrier update handles whether we throw out the filter.
    return nothing
end

function update_sigma!(relax::LOQORelaxationUpdate{T}, solver::MadNLPCSolver{T}) where {T}
    ipm = solver.ipm
    mpcc = solver.mpcc
    ncc = mpcc.meta.ncc
    # Calculate mean primal complementarity
    cc_pr = @views dot(
        MadNLP.variable(ipm.x)[mpcc.meta.ind_cc1] -
        MadNLP.variable(ipm.xl)[mpcc.meta.ind_cc1],
        MadNLP.variable(ipm.x)[mpcc.meta.ind_cc2] -
        MadNLP.variable(ipm.xl)[mpcc.meta.ind_cc2],
    )
    mean_cc = cc_pr/ncc
    # Calculate minimum primal complementarity
    min_cc_pr = @views mapreduce(
        (x1, xl1, x2, x2l) -> (x1-xl1)*(x2-x2l),
        min,
        MadNLP.variable(ipm.x)[mpcc.meta.ind_cc1],
        MadNLP.variable(ipm.xl)[mpcc.meta.ind_cc1],
        MadNLP.variable(ipm.x)[mpcc.meta.ind_cc2],
        MadNLP.variable(ipm.xl)[mpcc.meta.ind_cc2],
        init=T(Inf),
    )
    # Calculate the factor to multiply the mean complementarity by.
    xi = min_cc_pr/mean_cc
    gamma_sigma = max(relax.gamma_min, relax.gamma*min((1-relax.r)*((1-xi)/xi), 2)^3)
    # TODO(@anton) in principle we would like to not reduce this too much depending on how close we are to the KKT conds
    solver.scholtes.𝜎[] =
        max(gamma_sigma*mean_cc, solver.opts.sigma_min, relax.mu_factor*ipm.mu)
    # Throw out the filter as the barrier problem has changed
    empty!(ipm.filter)
    push!(ipm.filter, (ipm.theta_max, -Inf))
    return nothing
end
