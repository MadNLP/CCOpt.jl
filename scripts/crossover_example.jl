using MPCCBenchmark, JuMP, ExaModels
using MadNLP, MadNLPHSL
using CCOpt

function solve_model(model; tol=1e-8)
    model = MPCCBenchmark.reformulate_to_vertical!(JuMP.backend(model))
    ind_cc1, ind_cc2 = MPCCBenchmark.reformulate_to_standard_form!(model)
    ind_x1 = getfield.(ind_cc1, :value)
    ind_x2 = getfield.(ind_cc2, :value)

    nlp = ExaModel(model)
    mpcc = CCOpt.MPCCModelVarVar(nlp, ind_x1, ind_x2)

    madnlpc_opts = CCOpt.RelaxationOptions(
        ;
        print_level=MadNLP.INFO,
        relaxation=CCOpt.ScholtesRelaxation,
		    relaxation_update=CCOpt.RolloffRelaxationUpdate(sigma_min=tol),
        use_magic_step=false,
        use_specialized_barrier_update=false,
    )
    solver = CCOpt.RelaxationSolver(
        mpcc;
        solver_opts=madnlpc_opts,
        print_level=MadNLP.INFO,
        bound_relax_factor=0.0,
        max_iter=3000,
        tol=tol,
        linear_solver=Ma27Solver,
    )
    stats = CCOpt.solve_homotopy!(solver)
    return (
        stats.solution,
        mpcc
    )
end


function crossover_example(matpower_dir; tol=1e-8, proj_tr=1e3*tol)
    model = MPCCBenchmark.load_model(("case_ACTIVSg500", 10), MPCCBenchmark.SCOPFBenchmark(matpower_dir))
    x_relax, mpcc = solve_model(model; tol=tol)

    crossover_options = CCOpt.CrossoverOptions(
        ;
        lpcc_solver_opts=CCOpt.RelaxationOptions(print_level=MadNLP.ERROR),
        lpcc_solver_kwargs=(print_level=MadNLP.ERROR,
                            tol=1e-8,
                            linear_solver=Ma27Solver,
                            bound_relax_factor=0,
                            bound_push=1e-4),
        bnlp_solver_kwargs=(print_level=MadNLP.ERROR, linear_solver=Ma27Solver),
        proj_tr=proj_tr,
        stationarity_tol=1e-7,
        s_stat_tol=1e-7
    )

    crossover_stats = CCOpt.crossover(mpcc, x_relax, crossover_options);
end
