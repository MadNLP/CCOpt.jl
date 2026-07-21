@testset "RelaxationSolver Test" begin
    @testset "Default Options" begin
        mpcc = SimpleMPCCModel(Float64)
        ccopt_opts = CCOpt.RelaxationOptions(; print_level=MadNLP.ERROR)
        solver =
            CCOpt.RelaxationSolver(mpcc; solver_opts=ccopt_opts, print_level=MadNLP.ERROR)

        copyto!(get_x0(mpcc), [2; 1])

        stats = CCOpt.solve_homotopy!(solver)

        @test stats.status ∈ [MadNLP.SOLVE_SUCCEEDED, MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL]
        @test stats.objective ≈ 1 atol=1e-5
        @test stats.solution ≈ [1, 0] atol=1e-5
    end

    @testset "Quality Function" begin
        mpcc = SimpleMPCCModel(Float64)
        ccopt_opts = CCOpt.RelaxationOptions(; print_level=MadNLP.ERROR)
        solver =
            CCOpt.RelaxationSolver(mpcc;
                                   solver_opts=ccopt_opts,
                                   print_level=MadNLP.ERROR,
                                   barrier=MadNLP.QualityFunctionUpdate()
                                   )

        copyto!(get_x0(mpcc), [2; 1])

        stats = CCOpt.solve_homotopy!(solver)

        @test stats.status ∈ [MadNLP.SOLVE_SUCCEEDED, MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL]
        @test stats.objective ≈ 1 atol=1e-5
        @test stats.solution ≈ [1, 0] atol=1e-5
    end

    @testset "Quality Function specialization" begin
        mpcc = SimpleMPCCModel(Float64)
        ccopt_opts = CCOpt.RelaxationOptions(
            ;
            print_level=MadNLP.ERROR,
            use_specialized_barrier_update=true,
        )
        solver =
            CCOpt.RelaxationSolver(mpcc;
                                   solver_opts=ccopt_opts,
                                   print_level=MadNLP.ERROR,
                                   barrier=MadNLP.QualityFunctionUpdate()
                                   )

        copyto!(get_x0(mpcc), [2; 1])

        stats = CCOpt.solve_homotopy!(solver)

        @test stats.status ∈ [MadNLP.SOLVE_SUCCEEDED, MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL]
        @test stats.objective ≈ 1 atol=1e-5
        @test stats.solution ≈ [1, 0] atol=1e-5
    end

    @testset "Rolloff relaxation" begin
        mpcc = SimpleMPCCModel(Float64)
        ccopt_opts = CCOpt.RelaxationOptions(
            ;
            print_level=MadNLP.ERROR,
            relaxation_update = CCOpt.RolloffRelaxationUpdate(),
        )
        solver =
            CCOpt.RelaxationSolver(mpcc;
                                   solver_opts=ccopt_opts,
                                   print_level=MadNLP.ERROR,
                                   )

        copyto!(get_x0(mpcc), [2; 1])

        stats = CCOpt.solve_homotopy!(solver)

        @test stats.status ∈ [MadNLP.SOLVE_SUCCEEDED, MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL]
        @test stats.objective ≈ 1 atol=1e-5
        @test stats.solution ≈ [1, 0] atol=1e-5
    end

    @testset "Endgame relaxation algorithm" begin
        mpcc = SimpleMPCCModel(Float64)
        ccopt_opts = CCOpt.RelaxationOptions(
            ;
            print_level=MadNLP.ERROR,
            endgame_strategy=CCOpt.RelaxLBEndgameStrategy(),
        )
        solver =
            CCOpt.RelaxationSolver(mpcc;
                                   solver_opts=ccopt_opts,
                                   print_level=MadNLP.ERROR,
                                   )

        copyto!(get_x0(mpcc), [2; 1])

        stats = CCOpt.solve_homotopy!(solver)

        @test stats.status ∈ [MadNLP.SOLVE_SUCCEEDED, MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL]
        @test stats.objective ≈ 1 atol=1e-5
        @test stats.solution ≈ [1, 0] atol=1e-5
    end
end
