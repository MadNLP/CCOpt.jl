function MadNLP.update_barrier!(
    barrier::MadNLP.LOQOUpdate{T},
    solver::MadNLPCSolver{T},
    sc::T,
) where {T}
    ipm = solver.ipm
    if !solver.opts.use_specialized_barrier_update
        return MadNLP.update_barrier!(barrier, ipm, sc)
    end
    mu = MadNLP.get_average_complementarity(solver) # get average complementarity.
    ncc = ipm.nlb + ipm.nub
    min_cc = MadNLP.get_min_complementarity(solver)
    xi = min_cc/mu
    sigma = barrier.gamma*min((1-barrier.r)*((1-xi)/xi), 2)^3
    new_mu = max(sigma*mu, barrier.mu_min)
    if ipm.mu != new_mu # Stop clearing if we reached mu_min
        empty!(ipm.filter)
        push!(ipm.filter, (ipm.theta_max, -Inf))
    end
    ipm.mu = max(sigma*mu, barrier.mu_min)

    return nothing
end

function MadNLP.update_barrier!(
    barrier::MadNLP.MonotoneUpdate{T},
    solver::MadNLPCSolver{T},
    sc::T,
) where {T}
    return MadNLP.update_barrier!(barrier, solver.ipm, sc)
end

function MadNLP.update_barrier!(
    barrier::MadNLP.QualityFunctionUpdate{T},
    solver::MadNLPCSolver{T},
    sc::T,
) where {T}
    # TODO(@anton) also specialize this for mpcc
    return MadNLP.update_barrier!(barrier, solver.ipm, sc)
end
