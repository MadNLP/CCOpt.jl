######################### Types #########################
mutable struct HomotopySolverStats
  # TODO(@anton) what needs to live here
end

mutable struct HomotopySolverOptions{T}
  𝜎₀::T = 1
  𝛼::T = 0.1
  𝛽::T = 1.5

  comp_tol::T = 1e-8

end
# TODO(@anton) mutable?
mutable struct HomotopySolver{M} <: AbstractMPCCSolver{M<:AbstactMPCCModel{T,VT}}
  mpcc::M
  nlp::ScholtesRelaxation{T,VT}
  solver::IpoptSolver

  opts::HomotopySolverOptions
  stats::HomotopySolverStats

  x_k::VT
  y_k::VT

  𝜎::T
end

function HomotopySolver(mpcc::AbstractMPCCModel, opts::HomotopySolverOptions)
  nlp = ScholtesRelaxation(mpcc)
  stats = HomotopySolverStats

  solver = IpoptSolver(nlp)

  x_k = nlp.meta.x0
  y_k = nlp.meta.y0

  𝜎 = opts.𝜎₀

  HomotopySolver(mpcc, nlp, opts, stats, x_k, y_k, 𝜎)
end
######################### Main loop #########################

function solve(mpcc::AbstractMPCCModel, solver::HomotopySolver, stats::HomotopySolverStats;
               x=nothing, y=nothing, kwargs...)
  if x ≠ nothing
    solver.x_k .= x
  end
  if y ≠ nothing
    solver.y_k .= y
  end

  converged = false
  solver.nlp.𝜎[] = solver.𝜎

  while !converged
    stats = solve!(solver, solver.nlp, )
    cc_res = comp_resudual(mpcc, stats.solution; x0=solver.x_k, y0=solver.y_k)

    if stats.status ∈ { :first_order, :acceptable, :small_step } && cc_res ≤ mpcc.opts.comp_tol
      converged = true
    end
    solver.𝜎 = min(𝛼*solver.𝜎, solver.𝜎^𝛽)
    solver.nlp.𝜎[] = solver.𝜎
    solver.x_k = stats.solution
    solver.y_k = stats.multipliers
  end
end
