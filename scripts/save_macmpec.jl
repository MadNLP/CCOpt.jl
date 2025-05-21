using CSV
using DataFrames
using Plots, BenchmarkProfiles

function save_madnlp_df(
    names::Vector{AbstractString},
    stats_madnlp::Vector{MadMPEC.HomotopySolverStats{T, VT}},
    name::AbstractString,
) where {T, VT}
    df_madnlp = DataFrame(
        name=names,
        success=[s.status == MadMPEC.NLP_STATIONARY for s in stats_madnlp],
        status=[s.status for s in stats_madnlp],
        objective=[s.objective for s in stats_madnlp],
        inf_cc=[s.inf_cc for s in stats_madnlp],
        wall_time=[s.wall_time for s in stats_madnlp],
        iter=[s.iter for s in stats_madnlp],
        eval_function_time=[s.eval_function_time for s in stats_madnlp],
        linear_solver_time=[s.linear_solver_time for s in stats_madnlp],
        homotopy_iter=[length(s.nlp_stats) for s in stats_madnlp],
    )
    CSV.write(name, df_madnlp)

    return df_madnlp
end

function save_ipopt_df(
    names::Vector{AbstractString},
    stats_ipopt::Vector{MadMPEC.HomotopySolverStats{T, VT}},
    name::AbstractString,
) where {T, VT}
    df_ipopt = DataFrame(
        name=names,
        success=[s.status == MadMPEC.NLP_STATIONARY for s in stats_ipopt],
        status=[s.status for s in stats_ipopt],
        objective=[s.objective for s in stats_ipopt],
        inf_cc=[s.inf_cc for s in stats_ipopt],
        wall_time=[s.wall_time for s in stats_ipopt],
        iter=[s.iter for s in stats_ipopt],
        homotopy_iter=[length(s.nlp_stats) for s in stats_ipopt],
    )
    CSV.write(name, df_ipopt)

    return df_ipopt
end

function perf_plot(df_ipopt::DataFrame, df_madnlp::DataFrame, title::AbstractString)
    cost_ipopt = df_ipopt.wall_time
    cost_madnlp = df_madnlp.wall_time

    cost_ipopt[findall(.!df_ipopt.success)] .= Inf
    cost_madnlp[findall(.!df_madnlp.success)] .= Inf
    return performance_profile(
        PlotsBackend(),
        hcat(cost_ipopt, cost_madnlp),
        ["Ipopt", "madNLP"],
        title=title,
    )
end
