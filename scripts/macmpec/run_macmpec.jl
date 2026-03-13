include("../common.jl")

function load_ampl_benchmark(nlpath::AbstractString)
    probs = readdir(abspath(nlpath), join=true)
    mpccs::Vector{CCOpt.AbstractMPCCModel} = []
    names = Vector{String}()
    for i in 1:length(probs)
        model = AmplNLReader.AmplModel(probs[i])
        mpcc = CCOpt.vertical_form(mpcc_from_ampl(model))
        push!(mpccs, mpcc)
        push!(names, basename(probs[i]))
    end
    return names, mpccs
end

function run_macmpec(args...; plot=false, range=:)
    stats = Dict{String, Any}()
    names, probs =
        load_ampl_benchmark(joinpath(dirname(@__FILE__), "../../data/macMPEC/nls/"))
    solnames = Vector{String}()
    for (solname::AbstractString, solfun, dffun, opts, solargs) in args
        stats[solname] = run_benchmark(probs[range], solfun, opts, solargs...)
        push!(solnames, solname)
        stats[solname] = dffun(
            names[range],
            stats[solname],
            probs[range],
            replace(solname, " "=>"_")*".csv",
        )
    end
    if plot
        perf_plot("Performance Plot", solnames, stats)
    end
    return solnames, names, stats
end

function test_macmpec(; range=:)
    opts_ipopt = CCOpt.HomotopySolverOptions()
    opts_ipopt.print_level = 4
    opts_ipopt.nlp_solver_options[:print_level] = 5
    opts_ipopt.nlp_solver_options[:max_iter] = 500
    opts_madnlp = CCOpt.HomotopySolverOptions()
    opts_madnlp.nlp_solver_options =
        Dict(:bound_relax_factor=>0.0, :print_level=>MadNLP.DEBUG, :max_iter=>500)

    opts_ncl = MadNCL.NCLOptions();

    default_ipopt = (
        "nosnoc Ipopt",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt,
        (NLPModelsIpopt.IpoptSolver,),
    )
    default_madnlp = (
        "default madNLP",
        solve_benchmark_problem,
        save_madnlp_df,
        opts_madnlp,
        (MadNLP.MadNLPSolver,),
    )
    default_madncl = ("default madNCL", solve_benchmark_problem, save_ncl_df, opts_ncl, ())

    solnames, names, stats =
        run_macmpec(default_ipopt, default_madnlp, default_madncl; range=range)

    return solnames, names, stats
end

function test_ipopt_mu_update(; range=:)
    opts_ipopt_monotone = CCOpt.HomotopySolverOptions()
    opts_ipopt_monotone.print_level = 5
    opts_ipopt_monotone.nlp_solver_options[:print_level] = 5
    opts_ipopt_monotone.nlp_solver_options[:max_iter] = 3000
    opts_ipopt_monotone.nlp_solver_options[:mu_strategy] = "monotone"

    opts_ipopt_adaptive_quality = CCOpt.HomotopySolverOptions()
    opts_ipopt_adaptive_quality.print_level = 5
    opts_ipopt_adaptive_quality.nlp_solver_options[:print_level] = 5
    opts_ipopt_adaptive_quality.nlp_solver_options[:max_iter] = 3000
    opts_ipopt_adaptive_quality.nlp_solver_options[:mu_strategy] = "adaptive"
    opts_ipopt_adaptive_quality.nlp_solver_options[:mu_oracle] = "quality-function"

    opts_ipopt_adaptive_probing = CCOpt.HomotopySolverOptions()
    opts_ipopt_adaptive_probing.print_level = 5
    opts_ipopt_adaptive_probing.nlp_solver_options[:print_level] = 5
    opts_ipopt_adaptive_probing.nlp_solver_options[:max_iter] = 3000
    opts_ipopt_adaptive_probing.nlp_solver_options[:mu_strategy] = "adaptive"
    opts_ipopt_adaptive_probing.nlp_solver_options[:mu_oracle] = "probing"

    opts_ipopt_adaptive_loqo = CCOpt.HomotopySolverOptions()
    opts_ipopt_adaptive_loqo.print_level = 5
    opts_ipopt_adaptive_loqo.nlp_solver_options[:print_level] = 5
    opts_ipopt_adaptive_loqo.nlp_solver_options[:max_iter] = 3000
    opts_ipopt_adaptive_loqo.nlp_solver_options[:mu_strategy] = "adaptive"
    opts_ipopt_adaptive_loqo.nlp_solver_options[:mu_oracle] = "loqo"

    ipopt_monotone = (
        "Ipopt monotone",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt_monotone,
        (NLPModelsIpopt.IpoptSolver,),
    )
    ipopt_adaptive_quality = (
        "Ipopt adaptive quality function",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt_adaptive_quality,
        (NLPModelsIpopt.IpoptSolver,),
    )
    ipopt_adaptive_probing = (
        "Ipopt adaptive probing",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt_adaptive_probing,
        (NLPModelsIpopt.IpoptSolver,),
    )
    ipopt_adaptive_loqo = (
        "Ipopt adaptive loqo",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt_adaptive_loqo,
        (NLPModelsIpopt.IpoptSolver,),
    )

    solnames, names, stats = run_macmpec(
        ipopt_monotone,
        ipopt1,
        ipopt_adaptive_probing,
        ipopt_adaptive_loqo;
        range=range,
    )

    return solnames, names, stats
end

function test_homotopy_bound_push(; range=:)
    opts_ipopt1 = CCOpt.HomotopySolverOptions()
    opts_ipopt1.print_level = 5
    opts_ipopt1.nlp_solver_options[:print_level] = 5
    opts_ipopt1.nlp_solver_options[:max_iter] = 3000

    ipopt1 = (
        "Ipopt bp default",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt1,
        (NLPModelsIpopt.IpoptSolver,),
    )

    opts_ipopt2 = CCOpt.HomotopySolverOptions(warm_start_bound_push=1e-5)
    opts_ipopt2.print_level = 5
    opts_ipopt2.nlp_solver_options[:print_level] = 5
    opts_ipopt2.nlp_solver_options[:max_iter] = 3000

    ipopt2 = (
        "Ipopt bp 1e-5",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt2,
        (NLPModelsIpopt.IpoptSolver,),
    )

    opts_ipopt3 = CCOpt.HomotopySolverOptions(warm_start_bound_push=1e-9)
    opts_ipopt3.print_level = 5
    opts_ipopt3.nlp_solver_options[:print_level] = 5
    opts_ipopt3.nlp_solver_options[:max_iter] = 3000

    ipopt3 = (
        "Ipopt bp 1e-9",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt3,
        (NLPModelsIpopt.IpoptSolver,),
    )

    opts_madnlp1 = CCOpt.HomotopySolverOptions()
    opts_madnlp1.print_level = 5
    opts_madnlp1.nlp_solver_options =
        Dict(:bound_relax_factor=>0.0, :print_level=>MadNLP.ERROR, :max_iter=>3000)

    madnlp1 = (
        "madNLP bp default",
        solve_benchmark_problem,
        save_madnlp_df,
        opts_madnlp1,
        (MadNLP.MadNLPSolver,),
    )

    opts_madnlp2 = CCOpt.HomotopySolverOptions(warm_start_bound_push=1e-5)
    opts_madnlp2.print_level = 5
    opts_madnlp2.nlp_solver_options =
        Dict(:bound_relax_factor=>0.0, :print_level=>MadNLP.ERROR, :max_iter=>3000)

    madnlp2 = (
        "madNLP bp 1e-5",
        solve_benchmark_problem,
        save_madnlp_df,
        opts_madnlp2,
        (MadNLP.MadNLPSolver,),
    )

    opts_madnlp3 = CCOpt.HomotopySolverOptions(warm_start_bound_push=1e-9)
    opts_madnlp3.print_level = 5
    opts_madnlp3.nlp_solver_options =
        Dict(:bound_relax_factor=>0.0, :print_level=>MadNLP.ERROR, :max_iter=>3000)

    madnlp3 = (
        "madNLP bp 1e-9",
        solve_benchmark_problem,
        save_madnlp_df,
        opts_madnlp3,
        (MadNLP.MadNLPSolver,),
    )

    solnames, names, stats =
        run_macmpec(ipopt1, ipopt2, ipopt3, madnlp1, madnlp2, madnlp3; range=range)

    return solnames, names, stats
end

function test_ipopt_norm_type(; range=:)
    opts_ipopt_1_norm = CCOpt.HomotopySolverOptions()
    opts_ipopt_1_norm.print_level = MadNLP.INFO
    opts_ipopt_1_norm.nlp_solver_options[:print_level] = 0
    opts_ipopt_1_norm.nlp_solver_options[:max_iter] = 3000
    opts_ipopt_1_norm.nlp_solver_options[:mu_strategy] = "adaptive"
    opts_ipopt_1_norm.nlp_solver_options[:mu_oracle] = "quality-function"
    opts_ipopt_1_norm.nlp_solver_options[:quality_function_norm_type] = "1-norm"

    opts_ipopt_2_norm_squared = CCOpt.HomotopySolverOptions()
    opts_ipopt_2_norm_squared.print_level = MadNLP.INFO
    opts_ipopt_2_norm_squared.nlp_solver_options[:print_level] = 0
    opts_ipopt_2_norm_squared.nlp_solver_options[:max_iter] = 3000
    opts_ipopt_2_norm_squared.nlp_solver_options[:mu_strategy] = "adaptive"
    opts_ipopt_2_norm_squared.nlp_solver_options[:mu_oracle] = "quality-function"
    opts_ipopt_2_norm_squared.nlp_solver_options[:quality_function_norm_type] = "2-norm-squared"

    opts_ipopt_max_norm = CCOpt.HomotopySolverOptions()
    opts_ipopt_max_norm.print_level = MadNLP.INFO
    opts_ipopt_max_norm.nlp_solver_options[:print_level] = 0
    opts_ipopt_max_norm.nlp_solver_options[:max_iter] = 3000
    opts_ipopt_max_norm.nlp_solver_options[:mu_strategy] = "adaptive"
    opts_ipopt_max_norm.nlp_solver_options[:mu_oracle] = "quality-function"
    opts_ipopt_max_norm.nlp_solver_options[:quality_function_norm_type] = "max-norm"

    opts_ipopt_2_norm = CCOpt.HomotopySolverOptions()
    opts_ipopt_2_norm.print_level = MadNLP.INFO
    opts_ipopt_2_norm.nlp_solver_options[:print_level] = 0
    opts_ipopt_2_norm.nlp_solver_options[:max_iter] = 3000
    opts_ipopt_2_norm.nlp_solver_options[:mu_strategy] = "adaptive"
    opts_ipopt_2_norm.nlp_solver_options[:mu_oracle] = "quality-function"
    opts_ipopt_2_norm.nlp_solver_options[:quality_function_norm_type] = "2-norm"

    ipopt_1_norm = (
        "Ipopt 1-norm",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt_1_norm,
        (NLPModelsIpopt.IpoptSolver,),
    )
    ipopt_2_norm_squared = (
        "Ipopt 2-norm-squared",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt_2_norm_squared,
        (NLPModelsIpopt.IpoptSolver,),
    )
    ipopt_max_norm = (
        "Ipopt max-norm",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt_max_norm,
        (NLPModelsIpopt.IpoptSolver,),
    )
    ipopt_2_norm = (
        "Ipopt 2-norm",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt_2_norm,
        (NLPModelsIpopt.IpoptSolver,),
    )

    solnames, names, stats = run_macmpec(
        ipopt_1_norm,
        ipopt_2_norm_squared,
        ipopt_max_norm,
        ipopt_2_norm;
        range=range,
    )

    return solnames, names, stats
end

function test_macmpec_hsl(; range=:)
    opts_ipopt = CCOpt.HomotopySolverOptions()
    opts_ipopt.print_level = MadNLP.INFO
    opts_ipopt.nlp_solver_options[:print_level] = 0
    opts_ipopt.nlp_solver_options[:max_iter] = 3000
    opts_ipopt.nlp_solver_options[:linear_solver] = "ma27"
    opts_madnlp = CCOpt.HomotopySolverOptions()
    opts_madnlp.print_level = MadNLP.INFO
    opts_madnlp.nlp_solver_options = Dict(
        :bound_relax_factor=>0.0,
        :print_level=>MadNLP.ERROR,
        :max_iter=>3000,
        :linear_solver=>Ma27Solver,
    )

    opts_ncl = MadNCL.NCLOptions();
    #opts_madnlp.nlp_solver_options = Dict(:print_level=>MadNLP.INFO)

    default_ipopt = (
        "ma27 Ipopt",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt,
        (NLPModelsIpopt.IpoptSolver,),
    )
    default_madnlp = (
        "ma27 madNLP",
        solve_benchmark_problem,
        save_madnlp_df,
        opts_madnlp,
        (MadNLP.MadNLPSolver,),
    )
    default_madncl = (
        "ma27 madNCL",
        solve_benchmark_problem,
        save_ncl_df,
        opts_ncl,
        ((:print_level, MadNLP.ERROR), (:linear_solver, Ma27Solver)),
    )

    solnames, names, stats =
        run_macmpec(default_ipopt, default_madnlp, default_madncl; range=range)

    return solnames, names, stats
end

function test_vs_madnlp_c(; range=:)
    opts_ipopt = CCOpt.HomotopySolverOptions()
    opts_ipopt.print_level = MadNLP.INFO
    opts_ipopt.nlp_solver_options[:print_level] = 0
    opts_ipopt.nlp_solver_options[:max_iter] = 3000
    opts_ipopt.nlp_solver_options[:linear_solver] = "ma27"
    opts_madnlp = CCOpt.HomotopySolverOptions()
    opts_madnlp.print_level = MadNLP.INFO
    opts_madnlp.nlp_solver_options = Dict(
        :bound_relax_factor=>0.0,
        :print_level=>MadNLP.ERROR,
        :max_iter=>1000,
        :linear_solver=>Ma27Solver,
    )

    #opts_ncl = MadNCL.NCLOptions(feas_tol=1e-8) # Match tolerance
    #opts_madnlp.nlp_solver_options = Dict(:print_level=>MadNLP.INFO)

    opts_madnlp_c =
        CCOpt.MadNLPCOptions(kkt_regularization=:vicente_wright, print_level=MadNLP.ERROR)
    madnlpc_solver_options = Dict(
        :bound_relax_factor=>0.0,
        :print_level=>MadNLP.ERROR,
        :max_iter=>1000,
        :linear_solver=>Ma27Solver,
    )

    opts_madnlp_c_magic = CCOpt.MadNLPCOptions(
        use_magic_step=true,
        kkt_regularization=:vicente_wright,
        print_level=MadNLP.ERROR,
    )

    opts_exact_penalty = CCOpt.ExactPenaltyOptions(; print_level=MadNLP.ERROR)
    opts_exact_penalty_dynamic =
        CCOpt.ExactPenaltyOptions(; print_level=MadNLP.ERROR, dynamic_rho_update=true)
    exact_penalty_solver_options = Dict(
        :bound_relax_factor=>0.0,
        :print_level=>MadNLP.ERROR,
        :max_iter=>1000,
        :linear_solver=>Ma27Solver,
    )

    default_ipopt = (
        "ma27 Ipopt",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt,
        (NLPModelsIpopt.IpoptSolver,),
    )
    default_madnlp = (
        "ma27 madNLP",
        solve_benchmark_problem,
        save_madnlp_df,
        opts_madnlp,
        (MadNLP.MadNLPSolver,),
    )
    #default_madncl = ("ma27 madNCL", solve_benchmark_problem, save_ncl_df, opts_ncl, ())
    default_madnlp_c = (
        "ma27 madNLP-C",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c,
        ((madnlpc_solver_options...,)),
    )
    magic_madnlp_c = (
        "ma27 madNLP-C magic step",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_magic,
        ((madnlpc_solver_options...,)),
    )
    default_exact_penalty = (
        "ma27 exact penalty classic",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_exact_penalty,
        ((exact_penalty_solver_options...,)),
    )
    dynamic_exact_penalty = (
        "ma27 exact penalty dynamic",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_exact_penalty_dynamic,
        ((exact_penalty_solver_options...,)),
    )

    # solnames, names, stats = run_macmpec(
    #     default_exact_penalty,
    #     dynamic_exact_penalty,
    #     default_madnlp_c,
    #     default_ipopt,
    #     default_madnlp,
    #     range=range,
    # )
    solnames, names, stats = run_macmpec(
        #default_exact_penalty,
        dynamic_exact_penalty,
        default_madnlp_c,
        #default_ipopt,
        #default_madnlp,
        range=range,
    )

    return solnames, names, stats
end

function test_madnlp_c_opts(; range=:)
    opts_ipopt = CCOpt.HomotopySolverOptions()
    opts_ipopt.print_level = MadNLP.INFO
    opts_ipopt.nlp_solver_options[:print_level] = 0
    opts_ipopt.nlp_solver_options[:max_iter] = 3000
    opts_ipopt.nlp_solver_options[:linear_solver] = "ma27"
    opts_madnlp = CCOpt.HomotopySolverOptions()
    opts_madnlp.print_level = MadNLP.INFO
    opts_madnlp.nlp_solver_options = Dict(
        :bound_relax_factor=>0.0,
        :print_level=>MadNLP.ERROR,
        :max_iter=>3000,
        :linear_solver=>Ma27Solver,
    )

    madnlpc_solver_options = Dict(
        :bound_relax_factor=>0.0,
        :print_level=>MadNLP.ERROR,
        :max_iter=>3000,
        :linear_solver=>Ma27Solver,
    )
    opts_madnlp_c = CCOpt.MadNLPCOptions()
    opts_madnlp_c_reg = CCOpt.MadNLPCOptions(kkt_regularization=:vicente_wright)
    opts_madnlp_c_magic = CCOpt.MadNLPCOptions(use_magic_step=true, magic_step_kappa=0.5)
    opts_madnlp_c_magic_reg = CCOpt.MadNLPCOptions(
        use_magic_step=true,
        kkt_regularization=:vicente_wright,
        magic_step_kappa=0.5,
    )

    default_ipopt = (
        "ma27 Ipopt",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt,
        (NLPModelsIpopt.IpoptSolver,),
    )
    default_madnlp_c = (
        "ma27 madNLP-C default",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c,
        ((madnlpc_solver_options...,)),
    )
    reg_madnlp_c = (
        "ma27 madNLP-C vw regularization",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_reg,
        ((madnlpc_solver_options...,)),
    )
    magic_madnlp_c = (
        "ma27 madNLP-C magic step",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_magic,
        ((madnlpc_solver_options...,)),
    )
    magic_reg_madnlp_c = (
        "ma27 madNLP-C magic and vw",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_magic_reg,
        ((madnlpc_solver_options...,)),
    )

    solnames, names, stats = run_macmpec(
        reg_madnlp_c,
        magic_madnlp_c,
        magic_reg_madnlp_c,
        default_madnlp_c,
        default_ipopt,
        range=range,
    )

    return solnames, names, stats
end

function test_magic_opts(; range=:)
    madnlpc_solver_options = Dict(
        :bound_relax_factor=>0.0,
        :print_level=>MadNLP.ERROR,
        :max_iter=>1000,
        :linear_solver=>Ma27Solver,
    )
    opts_madnlp_c_magic_all = CCOpt.MadNLPCOptions(use_magic_step=true)
    opts_madnlp_c_magic_primal = CCOpt.MadNLPCOptions(
        use_magic_step=true,
        magic_step_duals=false,
        magic_step_slack=false,
        magic_step_slack_dual=false,
    )
    opts_madnlp_c_magic_primal_dual = CCOpt.MadNLPCOptions(
        use_magic_step=true,
        magic_step_duals=true,
        magic_step_slack=false,
        magic_step_slack_dual=false,
    )
    opts_madnlp_c_magic_primal_dual_slack = CCOpt.MadNLPCOptions(
        use_magic_step=true,
        magic_step_duals=true,
        magic_step_slack=true,
        magic_step_slack_dual=false,
    )

    magic_madnlp_c = (
        "ma27 madNLP-C magic all",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_magic_all,
        ((madnlpc_solver_options...,)),
    )
    magic_p_madnlp_c = (
        "ma27 madNLP-C magic primal",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_magic_primal,
        ((madnlpc_solver_options...,)),
    )
    magic_pd_madnlp_c = (
        "ma27 madNLP-C magic primal dual",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_magic_primal_dual,
        ((madnlpc_solver_options...,)),
    )
    magic_pds_madnlp_c = (
        "ma27 madNLP-C magic primal dual slack",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_magic_primal_dual_slack,
        ((madnlpc_solver_options...,)),
    )

    solnames, names, stats = run_macmpec(
        magic_madnlp_c,
        magic_p_madnlp_c,
        magic_pd_madnlp_c,
        magic_pds_madnlp_c;
        range=range,
    )

    return solnames, names, stats
end

function test_adaptive(; range=:)
    opts_ipopt = CCOpt.HomotopySolverOptions()
    opts_ipopt.print_level = MadNLP.ERROR
    opts_ipopt.nlp_solver_options[:print_level] = 0
    opts_ipopt.nlp_solver_options[:max_iter] = 3000
    opts_ipopt.nlp_solver_options[:linear_solver] = "ma27"

    default_ipopt = (
        "Ipopt Homotopy",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt,
        (NLPModelsIpopt.IpoptSolver,),
    )

    madnlpc_adaptive_solver_options = Dict(
        :bound_relax_factor=>0.0,
        :print_level=>MadNLP.ERROR,
        :max_iter=>3000,
        :linear_solver=>Ma27Solver,
        :barrier=>MadNLP.QualityFunctionUpdate(),
        :rethrow_error=>false,
    )
    opts_madnlpc_adaptive = CCOpt.MadNLPCOptions()
    opts_madnlpc_adaptive_sigma = CCOpt.MadNLPCOptions()

    madnlpc_loqo_solver_options = Dict(
        :bound_relax_factor=>0.0,
        :print_level=>MadNLP.ERROR,
        :max_iter=>3000,
        :linear_solver=>Ma27Solver,
        :barrier=>MadNLP.LOQOUpdate(gamma=0.05),
    )
    opts_madnlpc_loqo = CCOpt.MadNLPCOptions()

    madnlpc_monotone_solver_options = Dict(
        :bound_relax_factor=>0.0,
        :print_level=>MadNLP.ERROR,
        :max_iter=>3000,
        :linear_solver=>Ma27Solver,
    )
    opts_madnlpc_monotone = CCOpt.MadNLPCOptions()

    monotone_madnlp_c = (
        "ma27 madNLP-C monotone",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlpc_monotone,
        ((madnlpc_monotone_solver_options...,)),
    )

    adaptive_madnlp_c = (
        "ma27 madNLP-C adaptive",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlpc_adaptive,
        ((madnlpc_adaptive_solver_options...,)),
    )
    adaptive_sigma_madnlp_c = (
        "ma27 madNLP-C adaptive sigma",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlpc_adaptive_sigma,
        ((madnlpc_adaptive_solver_options...,)),
    )

    loqo_madnlp_c = (
        "ma27 madNLP-C loqo",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlpc_loqo,
        ((madnlpc_loqo_solver_options...,)),
    )

    solnames, names, stats = run_macmpec(
        loqo_madnlp_c,
        adaptive_madnlp_c,
        #adaptive_sigma_madnlp_c,
        monotone_madnlp_c,
        default_ipopt,
        range=range,
    )

    return solnames, names, stats
end

function test_bound_respect(; range=:)
    madnlpc_default_solver_options = Dict(
        :bound_relax_factor=>1e-10,
        :print_level=>MadNLP.ERROR,
        :max_iter=>1000,
        :linear_solver=>Ma27Solver,
    )
    opts_madnlpc_no_respect = CCOpt.MadNLPCOptions(respect_comp_bounds=false)
    opts_madnlpc_respect = CCOpt.MadNLPCOptions(respect_comp_bounds=true)

    no_respect_madnlp_c = (
        "ma27 madNLP-C no respect",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlpc_no_respect,
        ((madnlpc_default_solver_options...,)),
    )

    respect_madnlp_c = (
        "ma27 madNLP-C respect",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlpc_respect,
        ((madnlpc_default_solver_options...,)),
    )

    solnames, names, stats = run_macmpec(no_respect_madnlp_c, respect_madnlp_c, range=range)

    return solnames, names, stats
end

function test_loqo_sigma(; range=:)
    madnlpc_default_solver_options = Dict(
        :bound_relax_factor=>0.0,
        :print_level=>MadNLP.ERROR,
        :max_iter=>1000,
        :linear_solver=>Ma27Solver,
    )
    madnlpc_loqo_solver_options = Dict(
        :bound_relax_factor=>0.0,
        :print_level=>MadNLP.ERROR,
        :max_iter=>1000,
        :linear_solver=>Ma27Solver,
        :barrier=>MadNLP.LOQOUpdate(gamma=0.05),
    )

    opts_madnlpc_loqo = CCOpt.MadNLPCOptions(
        relaxation_update=CCOpt.LOQORelaxationUpdate(mu_factor=1e-2);
        use_specialized_barrier_update=false,
    )
    opts_madnlpc_default = CCOpt.MadNLPCOptions()

    default_madnlp_c = (
        "madNLP-C default update",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlpc_default,
        ((madnlpc_default_solver_options...,)),
    )

    loqo_sigma_madnlp_c = (
        "madNLP-C loqo sigma update",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlpc_loqo,
        ((madnlpc_default_solver_options...,)),
    )

    loqo_mu_madnlp_c = (
        "madNLP-C loqo mu update",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlpc_default,
        ((madnlpc_loqo_solver_options...,)),
    )

    full_loqo_madnlp_c = (
        "madNLP-C full loqo update",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlpc_loqo,
        ((madnlpc_loqo_solver_options...,)),
    )

    solnames, names, stats = run_macmpec(
        full_loqo_madnlp_c,
        loqo_mu_madnlp_c,
        loqo_sigma_madnlp_c,
        default_madnlp_c,
        range=range,
    )

    return solnames, names, stats
end

function test_loqo_sigma_params(; range=:)
    madnlpc_default_solver_options = Dict(
        :bound_relax_factor=>0.0,
        :print_level=>MadNLP.ERROR,
        :max_iter=>1000,
        :linear_solver=>Ma27Solver,
    )
    opts_madnlpc_loqo_8 =
        CCOpt.MadNLPCOptions(relaxation_update=CCOpt.LOQORelaxationUpdate())
    opts_madnlpc_loqo_9 =
        CCOpt.MadNLPCOptions(relaxation_update=CCOpt.LOQORelaxationUpdate(), sigma_min=1e-9)
    opts_madnlpc_default = CCOpt.MadNLPCOptions()

    default_madnlp_c = (
        "madNLP-C default update",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlpc_default,
        ((madnlpc_default_solver_options...,)),
    )

    loqo8_madnlp_c = (
        "madNLP-C loqo update min 1e-8",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlpc_loqo_8,
        ((madnlpc_default_solver_options...,)),
    )

    loqo9_madnlp_c = (
        "madNLP-C loqo update min 1e-9",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlpc_loqo_9,
        ((madnlpc_default_solver_options...,)),
    )

    solnames, names, stats = run_macmpec(loqo8_madnlp_c, loqo9_madnlp_c, range=range)

    return solnames, names, stats
end

function test_mpecopt(; range=:)
    opts_ipopt = CCOpt.HomotopySolverOptions()
    opts_ipopt.print_level = MadNLP.ERROR
    opts_ipopt.nlp_solver_options[:print_level] = 0
    opts_ipopt.nlp_solver_options[:max_iter] = 3000
    opts_ipopt.nlp_solver_options[:linear_solver] = "ma27"

    opts_madnlp_c = CCOpt.MadNLPCOptions(
        print_level=MadNLP.ERROR,
        use_specialized_barrier_update=false,
        relaxation_update=CCOpt.LOQORelaxationUpdate(mu_factor=1e-2),
    )
    madnlpc_solver_options = Dict(
        :bound_relax_factor=>0.0,
        :print_level=>MadNLP.ERROR,
        :max_iter=>3000,
        :linear_solver=>Ma27Solver,
        :barrier=>MadNLP.LOQOUpdate(gamma=0.05),
    )

    opts_mpecopt = CCOpt.MadNLPCOptions(
        kkt_regularization=:vicente_wright,
        phase_I_oracle=:naive,
        print_level=MadNLP.ERROR,
        use_mpecopt=true,
        eps_proj=1e-3,
    )

    default_ipopt = (
        "Ipopt Homotopy",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt,
        (NLPModelsIpopt.IpoptSolver,),
    )

    default_madnlp_c = (
        "CCOpt madNLP-C",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c,
        ((madnlpc_solver_options...,)),
    )
    mpecopt = (
        "madNLP-C-mpecopt",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_mpecopt,
        ((madnlpc_solver_options...,)),
    )

    solnames, names, stats = run_macmpec(
        mpecopt,
        default_madnlp_c,
        default_ipopt,
        #default_ipopt,
        #default_madnlp,
        range=range,
    )
    return solnames, names, stats
end

function test_fb(; range=:)
    opts_ipopt = CCOpt.HomotopySolverOptions()
    opts_ipopt.print_level = MadNLP.ERROR
    opts_ipopt.nlp_solver_options[:print_level] = 0
    opts_ipopt.nlp_solver_options[:max_iter] = 3000
    opts_ipopt.nlp_solver_options[:linear_solver] = "ma27"

    opts_scholtes = CCOpt.MadNLPCOptions(print_level=MadNLP.ERROR)
    madnlpc_solver_options = Dict(
        :bound_relax_factor=>0.0,
        :print_level=>MadNLP.ERROR,
        :max_iter=>3000,
        :linear_solver=>Ma27Solver,
    )

    opts_fb = CCOpt.MadNLPCOptions(
        print_level=MadNLP.ERROR,
        relaxation=CCOpt.FischerBurmeisterRelaxation,
    )
    opts_cck = CCOpt.MadNLPCOptions(
        print_level=MadNLP.ERROR,
        relaxation=CCOpt.ChenChenKanzowRelaxation,
    )
    opts_nr = CCOpt.MadNLPCOptions(
        print_level=MadNLP.ERROR,
        relaxation=CCOpt.NaturalResidualRelaxation,
    )

    default_ipopt = (
        "Ipopt Homotopy",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt,
        (NLPModelsIpopt.IpoptSolver,),
    )

    scholtes = (
        "madNLP-C scholtes",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_scholtes,
        ((madnlpc_solver_options...,)),
    )
    fb = (
        "madNLP-C fb",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_fb,
        ((madnlpc_solver_options...,)),
    )

    cck = (
        "madNLP-C cck",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_cck,
        ((madnlpc_solver_options...,)),
    )

    nr = (
        "madNLP-C nr",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_nr,
        ((madnlpc_solver_options...,)),
    )

    solnames, names, stats = run_macmpec(
        nr,
        cck,
        fb,
        scholtes,
        default_ipopt,
        #default_ipopt,
        #default_madnlp,
        range=range,
    )
    return solnames, names, stats
end

function test_sigma_mu_ratio(; range=:)
    opts_ipopt = CCOpt.HomotopySolverOptions()
    opts_ipopt.print_level = MadNLP.ERROR
    opts_ipopt.nlp_solver_options[:print_level] = 0
    opts_ipopt.nlp_solver_options[:max_iter] = 3000
    opts_ipopt.nlp_solver_options[:linear_solver] = "ma27"

    opts_1 = CCOpt.MadNLPCOptions(print_level=MadNLP.ERROR)

    madnlpc_solver_options = Dict(
        :bound_relax_factor=>0.0,
        :print_level=>MadNLP.ERROR,
        :max_iter=>3000,
        :linear_solver=>Ma27Solver,
        #:barrier=>MadNLP.MonotoneUpdate(mu_min=1e-16)
    )

    opts_10 = CCOpt.MadNLPCOptions(
        print_level=MadNLP.ERROR,
        relaxation_update=CCOpt.ProportionalRelaxationUpdate(sigma_mu_ratio=10.0),
    )
    opts_01 = CCOpt.MadNLPCOptions(
        print_level=MadNLP.ERROR,
        relaxation_update=CCOpt.ProportionalRelaxationUpdate(sigma_mu_ratio=0.1),
    )
    opts_sqr = CCOpt.MadNLPCOptions(
        print_level=MadNLP.ERROR,
        relaxation_update=CCOpt.ProportionalRelaxationUpdate(sigma_mu_exp=2.0),
    )

    opts_sqrt = CCOpt.MadNLPCOptions(
        print_level=MadNLP.ERROR,
        relaxation_update=CCOpt.ProportionalRelaxationUpdate(sigma_mu_exp=0.5),
    )

    default_ipopt = (
        "Ipopt Homotopy",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt,
        (NLPModelsIpopt.IpoptSolver,),
    )

    scholtes_1 = (
        "madNLP-C 1",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_1,
        ((madnlpc_solver_options...,)),
    )
    scholtes_10 = (
        "madNLP-C 10",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_10,
        ((madnlpc_solver_options...,)),
    )

    scholtes_01 = (
        "madNLP-C 0.1",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_01,
        ((madnlpc_solver_options...,)),
    )

    scholtes_sqr = (
        "madNLP-C squared",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_sqr,
        ((madnlpc_solver_options...,)),
    )
    scholtes_sqrt = (
        "madNLP-C sqrt",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_sqrt,
        ((madnlpc_solver_options...,)),
    )

    solnames, names, stats = run_macmpec(
        scholtes_sqr,
        #scholtes_sqrt,
        scholtes_1,
        scholtes_10,
        scholtes_01,
        default_ipopt,
        #default_ipopt,
        #default_madnlp,
        range=range,
    )
    return solnames, names, stats
end

function test_mu_init(; range=:)
    opts_ipopt = CCOpt.HomotopySolverOptions()
    opts_ipopt.print_level = MadNLP.ERROR
    opts_ipopt.nlp_solver_options[:print_level] = 0
    opts_ipopt.nlp_solver_options[:max_iter] = 3000
    opts_ipopt.nlp_solver_options[:linear_solver] = "ma27"

    madmpec_opts = CCOpt.MadNLPCOptions(print_level=MadNLP.ERROR)

    nlp_opts_1 = Dict(
        :bound_relax_factor=>0.0,
        :print_level=>MadNLP.ERROR,
        :max_iter=>3000,
        :linear_solver=>Ma27Solver,
        :barrier=>MadNLP.MonotoneUpdate(mu_init=1.0),
    )
    nlp_opts_01 = Dict(
        :bound_relax_factor=>0.0,
        :print_level=>MadNLP.ERROR,
        :max_iter=>3000,
        :linear_solver=>Ma27Solver,
        :barrier=>MadNLP.MonotoneUpdate(mu_init=0.1),
    )
    nlp_opts_10 = Dict(
        :bound_relax_factor=>0.0,
        :print_level=>MadNLP.ERROR,
        :max_iter=>3000,
        :linear_solver=>Ma27Solver,
        :barrier=>MadNLP.MonotoneUpdate(mu_init=10.0),
    )

    default_ipopt = (
        "Ipopt Homotopy",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt,
        (NLPModelsIpopt.IpoptSolver,),
    )

    mu_1 = (
        "madNLP-C mu0 = 1",
        solve_benchmark_problem,
        save_madnlp_c_df,
        madmpec_opts,
        ((nlp_opts_1...,)),
    )
    mu_10 = (
        "madNLP-C mu0 = 10",
        solve_benchmark_problem,
        save_madnlp_c_df,
        madmpec_opts,
        ((nlp_opts_10...,)),
    )

    mu_01 = (
        "madNLP-C mu0 = 0.1",
        solve_benchmark_problem,
        save_madnlp_c_df,
        madmpec_opts,
        ((nlp_opts_01...,)),
    )

    solnames, names, stats = run_macmpec(
        #scholtes_sqrt,
        mu_1,
        mu_10,
        mu_01,
        default_ipopt,
        #default_ipopt,
        #default_madnlp,
        range=range,
    )
    return solnames, names, stats
end

function test_slack_reset(; range=:)
    opts_ipopt = CCOpt.HomotopySolverOptions()
    opts_ipopt.print_level = MadNLP.ERROR
    opts_ipopt.nlp_solver_options[:print_level] = 0
    opts_ipopt.nlp_solver_options[:max_iter] = 3000
    opts_ipopt.nlp_solver_options[:linear_solver] = "ma27"

    opts_reset = CCOpt.MadNLPCOptions(print_level=MadNLP.ERROR, reset_slacks_on_update=true)
    opts_default = CCOpt.MadNLPCOptions(print_level=MadNLP.ERROR)

    madnlpc_solver_options = Dict(
        :bound_relax_factor=>0.0,
        :print_level=>MadNLP.ERROR,
        :max_iter=>3000,
        :linear_solver=>Ma27Solver,
    )

    default_ipopt = (
        "Ipopt Homotopy",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt,
        (NLPModelsIpopt.IpoptSolver,),
    )

    reset = (
        "madNLP-C slack reset",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_reset,
        ((madnlpc_solver_options...,)),
    )

    default_madnlpc = (
        "madNLP-C",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_default,
        ((madnlpc_solver_options...,)),
    )

    solnames, names, stats = run_macmpec(
        #reset,
        default_madnlpc,
        default_ipopt,
        range=range,
    )
    return solnames, names, stats
end

function test_madnlp_c_reg(; range=:)
    madnlpc_solver_options = Dict(
        :bound_relax_factor=>0.0,
        :print_level=>MadNLP.ERROR,
        :max_iter=>3000,
        :linear_solver=>Ma27Solver,
    )
    opts_madnlp_c_no_reg =
        CCOpt.MadNLPCOptions(kkt_regularization=:none, print_level=MadNLP.ERROR)
    opts_madnlp_c_reg =
        CCOpt.MadNLPCOptions(kkt_regularization=:vicente_wright, print_level=MadNLP.ERROR)
    opts_madnlp_c_reg_larger_eta = CCOpt.MadNLPCOptions(
        kkt_regularization=:vicente_wright,
        eta_factor=10.0,
        print_level=MadNLP.ERROR,
    )

    no_reg_madnlp_c = (
        "ma27 madNLP-C no vw",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_no_reg,
        ((madnlpc_solver_options...,)),
    )
    reg_madnlp_c = (
        "ma27 madNLP-C vw eta_fac=0.1",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_reg,
        ((madnlpc_solver_options...,)),
    )
    reg_madnlp_c_larger_eta = (
        "ma27 madNLP-C vw eta_fac=10.0",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_reg_larger_eta,
        ((madnlpc_solver_options...,)),
    )

    solnames, names, stats =
        run_macmpec(reg_madnlp_c, no_reg_madnlp_c, reg_madnlp_c_larger_eta; range=range)

    return solnames, names, stats
end

function test_two_sided(; range=:)
    madnlpc_solver_options = Dict(
        :bound_relax_factor=>0.0,
        :bound_push=>1e-1,
        :print_level=>MadNLP.ERROR,
        :max_iter=>3000,
        :linear_solver=>Ma27Solver,
        :rethrow_error=>false,
    )
    opts_madnlp_c_default = CCOpt.MadNLPCOptions(center_complementarities=true)
    opts_madnlp_c_two_sided = CCOpt.MadNLPCOptions(
        relaxation_update=CCOpt.TwoSidedScholtesUpdate(),
        center_complementarities=true,
        centering_factor=0.9,
    )
    opts_madnlp_c_lb = CCOpt.MadNLPCOptions(
        kkt_regularization=:none,
        relaxation_update=CCOpt.RelaxLBUpdate(),
        center_complementarities=true,
    )

    default_madnlp_c = (
        "ma27 madNLP-C",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_default,
        ((madnlpc_solver_options...,)),
    )
    two_sided_madnlp_c = (
        "ma27 madNLP-C two_sided",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_two_sided,
        ((madnlpc_solver_options...,)),
    )

    lb_madnlp_c = (
        "ma27 madNLP-C lb kappa=1.0",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_lb,
        ((madnlpc_solver_options...,)),
    )

    solnames, names, stats = run_macmpec(
        lb_madnlp_c,
        #two_sided_madnlp_c,
        default_madnlp_c;
        range=range,
    )

    return solnames, names, stats
end

function test_bound_push(; range=:)
    madnlpc_solver_options_bp1 = Dict(
        :bound_relax_factor=>0.0,
        :bound_push=>1e-1,
        :print_level=>MadNLP.ERROR,
        :max_iter=>3000,
        :linear_solver=>Ma27Solver,
        :rethrow_error=>false,
    )
    madnlpc_solver_options_bp2 = Dict(
        :bound_relax_factor=>0.0,
        :bound_push=>1e0,
        :print_level=>MadNLP.ERROR,
        :max_iter=>3000,
        :linear_solver=>Ma27Solver,
        :rethrow_error=>false,
    )
    madnlpc_solver_options_bp3 = Dict(
        :bound_relax_factor=>0.0,
        :bound_push=>1e1,
        :print_level=>MadNLP.ERROR,
        :max_iter=>3000,
        :linear_solver=>Ma27Solver,
        :rethrow_error=>false,
    )
    madnlpc_solver_options = Dict(
        :bound_relax_factor=>0.0,
        :print_level=>MadNLP.ERROR,
        :max_iter=>3000,
        :linear_solver=>Ma27Solver,
        :rethrow_error=>false,
    )
    opts_madnlp_c_default = CCOpt.MadNLPCOptions()
    opts_madnlp_c_center = CCOpt.MadNLPCOptions(center_complementarities=true)

    default_madnlp_c = (
        "ma27 madNLP-C",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_default,
        ((madnlpc_solver_options...,)),
    )
    bp1_madnlp_c = (
        "ma27 madNLP-C bp=1e-1",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_default,
        ((madnlpc_solver_options_bp1...,)),
    )
    bp2_madnlp_c = (
        "ma27 madNLP-C bp=1e0",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_default,
        ((madnlpc_solver_options_bp2...,)),
    )
    bp3_madnlp_c = (
        "ma27 madNLP-C bp=1e1",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_default,
        ((madnlpc_solver_options_bp3...,)),
    )

    center_madnlp_c = (
        "ma27 madNLP-C center bp=1e-1",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_center,
        ((madnlpc_solver_options_bp1...,)),
    )

    solnames, names, stats = run_macmpec(
        #bp1_madnlp_c,
        #bp2_madnlp_c,
        #bp3_madnlp_c,
        center_madnlp_c,
        #two_sided_madnlp_c,
        default_madnlp_c;
        range=range,
    )

    return solnames, names, stats
end

function test_eigenvalue_decomp(; range=:)
    opts_exact_penalty = CCOpt.ExactPenaltyOptions(; print_level=MadNLP.ERROR)
    opts_exact_penalty_eig_8 = CCOpt.ExactPenaltyOptions(;
        print_level=MadNLP.ERROR,
        kkt_regularization=:eigenvalue_decomposition,
        min_eig_value=1e-8,
    )
    opts_exact_penalty_eig_6 = CCOpt.ExactPenaltyOptions(;
        print_level=MadNLP.ERROR,
        kkt_regularization=:eigenvalue_decomposition,
        min_eig_value=1e-6,
    )
    opts_exact_penalty_eig_4 = CCOpt.ExactPenaltyOptions(;
        print_level=MadNLP.ERROR,
        kkt_regularization=:eigenvalue_decomposition,
        min_eig_value=1e-4,
    )
    opts_exact_penalty_eig_2 = CCOpt.ExactPenaltyOptions(;
        print_level=MadNLP.ERROR,
        kkt_regularization=:eigenvalue_decomposition,
        min_eig_value=1e-2,
    )
    opts_exact_penalty_eig_1 = CCOpt.ExactPenaltyOptions(;
        print_level=MadNLP.ERROR,
        kkt_regularization=:eigenvalue_decomposition,
        min_eig_value=1e-1,
    )
    opts_exact_penalty_critical_rho = CCOpt.ExactPenaltyOptions(;
        print_level=MadNLP.ERROR,
        kkt_regularization=:critical_rho,
    )
    # opts_exact_penalty_dynamic =
    #     CCOpt.ExactPenaltyOptions(; print_level=MadNLP.ERROR, dynamic_tau_update=true)
    exact_penalty_solver_options = Dict(
        :bound_relax_factor=>0.0,
        :print_level=>MadNLP.ERROR,
        :max_iter=>1000,
        :linear_solver=>Ma27Solver,
    )

    default_exact_penalty = (
        "ma27 exact penalty",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_exact_penalty,
        ((exact_penalty_solver_options...,)),
    )
    eig_exact_penalty_8 = (
        "ma27 exact penalty lam_min = 1e-8",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_exact_penalty_eig_8,
        ((exact_penalty_solver_options...,)),
    )
    eig_exact_penalty_6 = (
        "ma27 exact penalty lam_min = 1e-6",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_exact_penalty_eig_6,
        ((exact_penalty_solver_options...,)),
    )
    eig_exact_penalty_4 = (
        "ma27 exact penalty lam_min = 1e-4",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_exact_penalty_eig_4,
        ((exact_penalty_solver_options...,)),
    )
    eig_exact_penalty_2 = (
        "ma27 exact penalty lam_min = 1e-2",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_exact_penalty_eig_2,
        ((exact_penalty_solver_options...,)),
    )
    eig_exact_penalty_1 = (
        "ma27 exact penalty lam_min = 1e-1",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_exact_penalty_eig_1,
        ((exact_penalty_solver_options...,)),
    )

    critical_rho_exact_penalty = (
        "ma27 exact penalty critical rho",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_exact_penalty_critical_rho,
        ((exact_penalty_solver_options...,)),
    )

    solnames, names, stats = run_macmpec(
        eig_exact_penalty_8,
        eig_exact_penalty_6,
        critical_rho_exact_penalty,
        eig_exact_penalty_4,
        eig_exact_penalty_2,
        eig_exact_penalty_1,
        default_exact_penalty,
        #default_madnlp,
        range=range,
    )

    return solnames, names, stats
end

function test_dynamic_reg(; range=:)
    opts_exact_penalty =
        CCOpt.ExactPenaltyOptions(; print_level=MadNLP.ERROR, dynamic_tau_update=true)
    opts_exact_penalty_eig_8 = CCOpt.ExactPenaltyOptions(;
        print_level=MadNLP.ERROR,
        kkt_regularization=:eigenvalue_decomposition,
        min_eig_value=1e-8,
    )
    opts_exact_penalty_eig_6 = CCOpt.ExactPenaltyOptions(;
        print_level=MadNLP.ERROR,
        kkt_regularization=:eigenvalue_decomposition,
        min_eig_value=1e-6,
    )
    opts_exact_penalty_eig_4 = CCOpt.ExactPenaltyOptions(;
        print_level=MadNLP.ERROR,
        kkt_regularization=:eigenvalue_decomposition,
        min_eig_value=1e-4,
    )
    opts_exact_penalty_eig_2 = CCOpt.ExactPenaltyOptions(;
        print_level=MadNLP.ERROR,
        kkt_regularization=:eigenvalue_decomposition,
        min_eig_value=1e-2,
    )
    opts_exact_penalty_eig_1 = CCOpt.ExactPenaltyOptions(;
        print_level=MadNLP.ERROR,
        kkt_regularization=:eigenvalue_decomposition,
        min_eig_value=1e-1,
    )
    opts_exact_penalty_critical_rho = CCOpt.ExactPenaltyOptions(;
        print_level=MadNLP.ERROR,
        kkt_regularization=:critical_rho,
        dynamic_tau_update=true,
    )
    # opts_exact_penalty_dynamic =
    #     CCOpt.ExactPenaltyOptions(; print_level=MadNLP.ERROR, dynamic_tau_update=true)
    exact_penalty_solver_options = Dict(
        :bound_relax_factor=>0.0,
        :print_level=>MadNLP.ERROR,
        :max_iter=>1000,
        :linear_solver=>Ma27Solver,
    )

    default_exact_penalty = (
        "ma27 exact penalty",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_exact_penalty,
        ((exact_penalty_solver_options...,)),
    )
    eig_exact_penalty_8 = (
        "ma27 exact penalty lam_min = 1e-8",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_exact_penalty_eig_8,
        ((exact_penalty_solver_options...,)),
    )
    eig_exact_penalty_6 = (
        "ma27 exact penalty lam_min = 1e-6",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_exact_penalty_eig_6,
        ((exact_penalty_solver_options...,)),
    )
    eig_exact_penalty_4 = (
        "ma27 exact penalty lam_min = 1e-4",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_exact_penalty_eig_4,
        ((exact_penalty_solver_options...,)),
    )
    eig_exact_penalty_2 = (
        "ma27 exact penalty lam_min = 1e-2",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_exact_penalty_eig_2,
        ((exact_penalty_solver_options...,)),
    )
    eig_exact_penalty_1 = (
        "ma27 exact penalty lam_min = 1e-1",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_exact_penalty_eig_1,
        ((exact_penalty_solver_options...,)),
    )

    critical_rho_exact_penalty = (
        "ma27 exact penalty critical rho",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_exact_penalty_critical_rho,
        ((exact_penalty_solver_options...,)),
    )

    solnames, names, stats = run_macmpec(
        #eig_exact_penalty_8,
        #eig_exact_penalty_6,
        critical_rho_exact_penalty,
        #eig_exact_penalty_4,
        #eig_exact_penalty_2,
        #eig_exact_penalty_1,
        default_exact_penalty,
        #default_madnlp,
        range=range,
    )

    return solnames, names, stats
end

function test_ll(; range=:)
    opts_exact_penalty =
        CCOpt.ExactPenaltyOptions(; print_level=MadNLP.ERROR, dynamic_rho_update=true)
    opts_exact_penalty_ll = CCOpt.ExactPenaltyOptions(;
        print_level=MadNLP.ERROR,
        penalty=CCOpt.LasryLionsPenalty,
        dynamic_rho_update=false,
    )
    # opts_exact_penalty_dynamic =
    #     CCOpt.ExactPenaltyOptions(; print_level=MadNLP.ERROR, dynamic_tau_update=true)
    exact_penalty_solver_options = Dict(
        :bound_relax_factor=>0.0,
        :print_level=>MadNLP.ERROR,
        :max_iter=>1000,
        :linear_solver=>Ma27Solver,
        :rethrow_error=>false,
    )

    exact_penalty_solver_options_noscale = Dict(
        :bound_relax_factor=>0.0,
        :print_level=>MadNLP.ERROR,
        :max_iter=>1000,
        :linear_solver=>Ma27Solver,
        :rethrow_error=>false,
        :nlp_scaling=>false,
    )

    default_exact_penalty = (
        "ell1",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_exact_penalty,
        ((exact_penalty_solver_options...,)),
    )
    exact_penalty_ll = (
        "lasry-lions",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_exact_penalty_ll,
        ((exact_penalty_solver_options...,)),
    )
    exact_penalty_ll_noscale = (
        "lasry-lions no scaling",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_exact_penalty_ll,
        ((exact_penalty_solver_options_noscale...,)),
    )

    solnames, names, stats = run_macmpec(
        exact_penalty_ll,
        exact_penalty_ll_noscale,
        default_exact_penalty,
        #default_madnlp,
        range=range,
    )

    return solnames, names, stats
end

function test_lb_kkt_bound(; range=:)
    madnlpc_solver_options = Dict(
        :bound_relax_factor=>0.0,
        :bound_push=>1e-1,
        :print_level=>MadNLP.ERROR,
        :max_iter=>3000,
        :linear_solver=>Ma27Solver,
        :rethrow_error=>false,
    )
    opts_madnlp_c_default = CCOpt.MadNLPCOptions(center_complementarities=true)
    opts_madnlp_c_lb3 = CCOpt.MadNLPCOptions(
        kkt_regularization=:none,
        relaxation_update=CCOpt.RelaxLBUpdate(relax_threshold=1e-3),
        center_complementarities=true,
    )
    opts_madnlp_c_lb4 = CCOpt.MadNLPCOptions(
        kkt_regularization=:none,
        relaxation_update=CCOpt.RelaxLBUpdate(relax_threshold=1e-4),
        center_complementarities=true,
    )

    opts_madnlp_c_lb5 = CCOpt.MadNLPCOptions(
        kkt_regularization=:none,
        relaxation_update=CCOpt.RelaxLBUpdate(relax_threshold=1e-5),
        center_complementarities=true,
    )
    opts_madnlp_c_lb6 = CCOpt.MadNLPCOptions(
        kkt_regularization=:none,
        relaxation_update=CCOpt.RelaxLBUpdate(relax_threshold=1e-6),
        center_complementarities=true,
    )
    opts_madnlp_c_lb7 = CCOpt.MadNLPCOptions(
        kkt_regularization=:none,
        relaxation_update=CCOpt.RelaxLBUpdate(relax_threshold=1e-7),
        center_complementarities=true,
    )

    opts_madnlp = CCOpt.HomotopySolverOptions(max_inner_iter=3000)
    opts_madnlp.nlp_solver_options = Dict(
        :bound_relax_factor=>0.0,
        :print_level=>MadNLP.ERROR,
        :max_iter=>3000,
        :barrier=>MadNLP.QualityFunctionUpdate(),
        :linear_solver=>Ma27Solver,
    )

    default_madnlp_c = (
        "ma27 madNLP-C",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_default,
        ((madnlpc_solver_options...,)),
    )
    lb3_madnlp_c = (
        "ma27 madNLP-C lb thresh=1e-3",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_lb3,
        ((madnlpc_solver_options...,)),
    )
    lb4_madnlp_c = (
        "ma27 madNLP-C lb thresh=1e-4",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_lb4,
        ((madnlpc_solver_options...,)),
    )
    lb5_madnlp_c = (
        "ma27 madNLP-C lb thresh=1e-5",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_lb5,
        ((madnlpc_solver_options...,)),
    )
    lb6_madnlp_c = (
        "ma27 madNLP-C lb thresh=1e-6",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_lb6,
        ((madnlpc_solver_options...,)),
    )
    lb7_madnlp_c = (
        "ma27 madNLP-C lb thresh=1e-7",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_lb7,
        ((madnlpc_solver_options...,)),
    )
    default_madnlp = (
        "ma27 madNLP",
        solve_benchmark_problem,
        save_madnlp_df,
        opts_madnlp,
        (MadNLP.MadNLPSolver,),
    )

    solnames, names, stats = run_macmpec(
        #lb3_madnlp_c,
        #lb4_madnlp_c,
        #lb5_madnlp_c,
        #lb6_madnlp_c,
        lb7_madnlp_c,
        #two_sided_madnlp_c,
        default_madnlp_c,
        default_madnlp;
        range=range,
    )

    return solnames, names, stats
end

function test_rolloff(; range=:)
    madnlpc_solver_options = Dict(
        :bound_relax_factor=>0.0,
        :bound_push=>1e-1,
        :print_level=>MadNLP.ERROR,
        :max_iter=>3000,
        :linear_solver=>Ma27Solver,
        :rethrow_error=>false,
    )
    opts_madnlp = CCOpt.HomotopySolverOptions(max_inner_iter=3000)
    opts_madnlp.nlp_solver_options = Dict(
        :bound_relax_factor=>0.0,
        :print_level=>MadNLP.ERROR,
        :max_iter=>3000,
        :barrier=>MadNLP.QualityFunctionUpdate(),
        :linear_solver=>Ma27Solver,
    )
    opts_madnlp_c_default = CCOpt.MadNLPCOptions(center_complementarities=false)
    opts_madnlp_c_rolloff = CCOpt.MadNLPCOptions(
        relaxation_update=CCOpt.RolloffRelaxationUpdate(
            rolloff_slope=2.5,
            rolloff_point=1e-12,
            rolloff_max=1.0,
        ),
        center_complementarities=false,
        sigma_min=1e-8,
    )
    opts_madnlp_c_rolloff2 = CCOpt.MadNLPCOptions(
        relaxation_update=CCOpt.RolloffRelaxationUpdate(
            rolloff_slope=2.0,
            rolloff_point=1e-8,
            rolloff_max=1.0,
        ),
        center_complementarities=false,
        sigma_min=1e-8,
    )
    opts_madnlp_c_rolloff3 = CCOpt.MadNLPCOptions(
        relaxation_update=CCOpt.RolloffRelaxationUpdate(
            rolloff_slope=1.5,
            rolloff_point=1e-4,
            rolloff_max=1.0,
        ),
        center_complementarities=false,
        sigma_min=1e-8,
    )

    opts_madnlp_c_rolloff4 = CCOpt.MadNLPCOptions(
        relaxation_update=CCOpt.RolloffRelaxationUpdate(
            rolloff_slope=2.0,
            rolloff_point=1e-9,
            rolloff_max=0.1,
        ),
        center_complementarities=false,
        sigma_min=1e-8,
    )
    opts_ipopt = CCOpt.HomotopySolverOptions(max_inner_iter=3000)
    opts_ipopt.print_level = MadNLP.ERROR
    opts_ipopt.nlp_solver_options[:print_level] = 0
    opts_ipopt.nlp_solver_options[:max_iter] = 3000
    opts_ipopt.nlp_solver_options[:linear_solver] = "ma27"

    default_madnlp_c = (
        "ma27 madNLP-C",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_default,
        ((madnlpc_solver_options...,)),
    )
    rolloff_madnlp_c = (
        "ma27 madNLP-C rolloff",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_rolloff,
        ((madnlpc_solver_options...,)),
    )
    rolloff2_madnlp_c = (
        "ma27 madNLP-C rolloff 2",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_rolloff2,
        ((madnlpc_solver_options...,)),
    )
    rolloff3_madnlp_c = (
        "ma27 madNLP-C rolloff 3",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_rolloff3,
        ((madnlpc_solver_options...,)),
    )
    rolloff4_madnlp_c = (
        "ma27 madNLP-C rolloff 3",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_rolloff4,
        ((madnlpc_solver_options...,)),
    )

    default_madnlp = (
        "ma27 madNLP",
        solve_benchmark_problem,
        save_madnlp_df,
        opts_madnlp,
        (MadNLP.MadNLPSolver,),
    )

    default_ipopt = (
        "ma27 Ipopt",
        solve_benchmark_problem,
        save_ipopt_df,
        opts_ipopt,
        (NLPModelsIpopt.IpoptSolver,),
    )

    solnames, names, stats = run_macmpec(
        #rolloff_madnlp_c,
        #rolloff2_madnlp_c,
        #rolloff3_madnlp_c,
        #rolloff4_madnlp_c,
        #two_sided_madnlp_c,
        default_madnlp_c,
        #default_madnlp,
        default_ipopt;
        range=range,
    )

    return solnames, names, stats
end

function test_q_regularization(; range=:)
    madnlpc_solver_options = Dict(
        :bound_relax_factor=>0.0,
        :bound_push=>1e-1,
        :print_level=>MadNLP.ERROR,
        :max_iter=>3000,
        :linear_solver=>Ma27Solver,
        :rethrow_error=>false,
    )
    opts_madnlp_c_default = CCOpt.MadNLPCOptions()
    opts_madnlp_c_eig = CCOpt.MadNLPCOptions(q_regularization=:eigenvalue_decomposition)
    opts_madnlp_c_crit = CCOpt.MadNLPCOptions(q_regularization=:critical_rho)

    default_madnlp_c = (
        "ma27 madNLP-C",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_default,
        ((madnlpc_solver_options...,)),
    )
    eig_madnlp_c = (
        "ma27 madNLP-C eig",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_eig,
        ((madnlpc_solver_options...,)),
    )
    crit_madnlp_c = (
        "ma27 madNLP-C crit",
        solve_benchmark_problem,
        save_madnlp_c_df,
        opts_madnlp_c_crit,
        ((madnlpc_solver_options...,)),
    )

    solnames, names, stats = run_macmpec(
        eig_madnlp_c,
        crit_madnlp_c,
        #two_sided_madnlp_c,
        default_madnlp_c;
        range=range,
    )

    return solnames, names, stats
end
