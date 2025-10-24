function phase_II!(solver::Mpecopt{T, VT}) where {T, VT}
    # TODO(@anton) this is unoptimized for now as we generate a new solver at each iteration :)
    tr = solver.opts.tr0
    prev_obj = solver.ipm.obj_val / solver.ipm.cb.obj_scale[]
    mpcc = solver.mpcc
    opts = solver.opts
    MadMPEC.linearize_lpec!(solver, tr)

    # Check for S-stationarity
    @views begin
        if ~any(
            (
                solver.x[mpcc.meta.ind_cc1] .- mpcc.meta.lvar[mpcc.meta.ind_cc1] .<
                opts.s_stationarity_tol
            ) .& (
                solver.x[mpcc.meta.ind_cc2] .- mpcc.meta.lvar[mpcc.meta.ind_cc2] .<
                opts.s_stationarity_tol
            ),
        )
            return
        end
    end
    for i=1:opts.max_iter
        # Solve the corresponding LPCC
        optimal, d, b, obj =
            MadMPEC.solve_lpec!(solver; x0=vcat(zeros(mpcc.meta.nvar), solver.b))
        if optimal
            if norm(@view d[1:mpcc.meta.nvar]) <= opts.b_stationarity_tol  # TODO(@anton) make option
                solver.status = B_STATIONARY
                return
            elseif abs(obj) <= opts.b_stationarity_tol
                solver.status = B_STATIONARY
                return
            elseif (mpcc.meta.minimize && obj > 0) || (!mpcc.meta.minimize && obj < 0)
                # TODO(@anton) is this reasonable? this technically implies an infeasible point?
                solver.status = B_STATIONARY
                return
            elseif all(solver.b .== b) # TODO(@anton) this should maybe also check for "acceptable" tolerance
                #solver.status = B_STATIONARY
                #return
                tr = opts.phase_II_alpha_tr*tr # TODO(@anton) Options
                if tr <= opts.phase_II_tr_min
                    # Search direction too small
                    solver.status = SMALL_STEP
                end
                MadMPEC.update_lpec_tr!(solver, tr)
                continue
            else
                solver.b .= b
            end
        else
            solver.status = LPCC_ERROR
            return
        end
        # Build bnlp
        MadMPEC.build_bnlp_solver!(solver, b)

        # Solve bnlp
        ipm_stats = MadMPEC.solve_bnlp!(solver, stats.stats)

        # Check if BNLP succeeded
        if ipm_stats.status ∈ [
            MadNLP.SOLVE_SUCCEEDED,
            MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL,
            MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL,
        ]
            if ipm_stats.objective < prev_obj # Accept step
                println("ACCEPTED PHASE II STEP")
                # update current values
                prev_obj = ipm_stats.objective
                solver.x .= ipm_stats.solution
                # Check if we even need to solve an LPCC by checking for biactives:
                @views begin # TODO(@anton) add tolerance as option or maybe use tr?
                    if ~any(
                        (solver.x[mpcc.meta.ind_cc1] .< 1e-8) .&
                        (solver.x[mpcc.meta.ind_cc2] .< 1e-8),
                    )
                        solver.status = B_STATIONARY
                        continue
                    end
                end
                # Reset the trust region
                tr = opts.phase_II_tr0
                # Linearize at the current point
                MadMPEC.linearize_lpec!(solver, tr)
            else # Otherwise we did not get descent in the BNLP, reuse linearization and a smaller tr
                tr = opts.phase_II_alpha_tr*tr # TODO(@anton) Options
                println("REJECTED PHASE II STEP")
                if tr <= opts.phase_II_tr_min
                    # Search direction too small
                    solver.status = SEARCH_DIRECTION_BECOMES_TOO_SMALL
                    continue
                end
                MadMPEC.update_lpec_tr!(solver, tr)
            end
        else
            if ipm_stats.status == MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
                solver.status = MAXIMUM_ITERATIONS_EXCEEDED
            elseif ipm_stats.status == MadNLP.MAXIMUM_WALLTIME_EXCEEDED
                solver.status = MAXIMUM_WALLTIME_EXCEEDED
            else
                solver.status = IPM_ERROR
            end
        end
    end
end

function crossover!(mpcc::AbstractMPCCModel{T, VT}, x0::VT)

end
