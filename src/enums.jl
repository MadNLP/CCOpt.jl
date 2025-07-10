@enum(
    Status::Int,
    B_STATIONARY = 1,
    NLP_STATIONARY = 2,
    SEARCH_DIRECTION_BECOMES_TOO_SMALL = 3,
    DIVERGING_ITERATES = 4,
    INFEASIBLE_PROBLEM_DETECTED = 5,
    MAXIMUM_ITERATIONS_EXCEEDED = 6,
    MAXIMUM_WALLTIME_EXCEEDED = 7,
    INITIAL = 11,
    PHASE_I = 12,
    PHASE_II = 13,
    IPM_ERROR = -1,
    LPCC_ERROR = -2,
    INTERNAL_ERROR = -6
)

function get_status_output(status, opt, ipm_opt)
    if status == B_STATIONARY
        return "B-Stationary point found (tol = 1e-7)."
    elseif status == NLP_STATIONARY
        return @sprintf "NLP stationary point found (tol = %5.1e)." ipm_opt.tol
    elseif status == SEARCH_DIRECTION_BECOMES_TOO_SMALL
        return "Search Direction is becoming Too Small."
    elseif status == DIVERGING_ITERATES
        return "Iterates divering; problem might be unbounded."
    elseif status == MAXIMUM_ITERATIONS_EXCEEDED
        return "Maximum Number of Iterations Exceeded."
    elseif status == MAXIMUM_WALLTIME_EXCEEDED
        return "Maximum wall-clock Time Exceeded."
    elseif status == INFEASIBLE_PROBLEM_DETECTED
        return "Converged to a point of local infeasibility. Problem may be infeasible."
    elseif status == INTERNAL_ERROR
        return "Internal Error."
    elseif status == IPM_ERROR
        return "Error in interior point solver."
    elseif status == LPCC_ERROR
        return "Error in  lpcc solver."
    else
        error("status code is not valid")
    end
end
