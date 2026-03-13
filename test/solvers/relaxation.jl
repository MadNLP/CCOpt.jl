@testset "MadNLP-C Test" begin
    @testset "Default Options" begin
        mpcc = SimpleMPCCModel(Float64)
        madnlpc_opts = CCOpt.RelaxationOptions(; print_level=MadNLP.ERROR)
        solver =
            CCOpt.RelaxationSolver(mpcc; solver_opts=madnlpc_opts, print_level=MadNLP.ERROR)

        copyto!(get_x0(mpcc), [2; 1])

        stats = CCOpt.solve_homotopy!(solver)

        @test stats.status ∈ [MadNLP.SOLVE_SUCCEEDED, MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL]
        @test stats.objective ≈ 1 atol=1e-5
        @test stats.solution ≈ [1, 0] atol=1e-5
    end

    @testset "With Magic Steps" begin
        mpcc = SimpleMPCCModel(Float64)
        madnlpc_opts =
            CCOpt.RelaxationOptions(; print_level=MadNLP.ERROR, use_magic_step=true)
        solver =
            CCOpt.RelaxationSolver(mpcc; solver_opts=madnlpc_opts, print_level=MadNLP.ERROR)

        copyto!(get_x0(mpcc), [2; 1])

        stats = CCOpt.solve_homotopy!(solver)

        @test stats.status ∈ [MadNLP.SOLVE_SUCCEEDED, MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL]
        @test stats.objective ≈ 1 atol=1e-5
        @test stats.solution ≈ [1, 0] atol=1e-5
    end

    @testset "Magic Steps Check" begin
        kappa = 1.0
        sigma1 = 0.1
        x1 = 0.5
        y1 = 2.0
        xp, yp = CCOpt.project_scholtes_explicit(x1, y1, 0.0, 0.0, kappa, sigma1)
        @test xp ≈ 0.050285 atol=1e-5
        @test yp ≈ 1.988628 atol=1e-5
    end
end
