include("../common.jl")
include("generate_benchmark.jl")

function run_random_benchmark(args...; plot=false, range=:, n_probs=7)
    stats = Dict{String, Any}()
    names, probs = generate_benchmark_jump(n_probs)
    solnames = Vector{String}()
    for (solname::AbstractString, solfun, dffun, opts, solargs) in args
        stats[solname] = run_benchmark(probs[range], solfun, opts, solargs...)
        push!(solnames, solname)
        stats[solname] =
            dffun(names[range], stats[solname], probs, replace(solname, " "=>"_")*".csv")
    end
    if plot
        perf_plot("Performance Plot", solnames, stats)
    end
    return solnames, names, stats
end

function test_random_benchmark(; range=:, n_probs=7)
    opts_ipopt = MadMPEC.HomotopySolverOptions()
    opts_ipopt.print_level = MadNLP.INFO
    opts_ipopt.nlp_solver_options[:print_level] = 0
    opts_ipopt.nlp_solver_options[:max_iter] = 3000
    opts_ipopt.nlp_solver_options[:linear_solver] = "ma27"
    opts_madnlp = MadMPEC.HomotopySolverOptions()
    opts_madnlp.print_level = MadNLP.INFO
    opts_madnlp.nlp_solver_options = Dict(
        :bound_relax_factor=>1e-12,
        :print_level=>MadNLP.ERROR,
        :max_iter=>3000,
        :linear_solver=>Ma27Solver,
    )

    opts_ncl = MadNCL.NCLOptions();

    default_ipopt = (
        "random ma27 Ipopt",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt,
        (NLPModelsIpopt.IpoptSolver,),
    )
    default_madnlp = (
        "random ma27 madNLP",
        solve_benchmark_problem,
        save_madnlp_df,
        opts_madnlp,
        (MadNLP.MadNLPSolver,),
    )
    default_madncl = (
        "random ma27 madNCL",
        solve_benchmark_problem,
        save_ncl_df,
        opts_ncl,
        ((:print_level, MadNLP.ERROR), (:linear_solver, Ma27Solver)),
    )

    solnames, names, stats = run_random_benchmark(
        default_madncl,
        default_madnlp,
        default_ipopt;
        range=range,
        n_probs=n_probs,
    )

    return solnames, names, stats
end
