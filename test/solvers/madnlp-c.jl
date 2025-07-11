@testset "MadNLP-C Test" begin
    @testset "Default Options" begin
        mpcc = SimpleMPCCModel(Float64)
        madnlpc_opts = MadMPEC.MadNLPCOptions(; print_level=MadNLP.ERROR)
        solver =
            MadMPEC.MadNLPCSolver(mpcc; madnlpc_opts=madnlpc_opts, print_level=MadNLP.ERROR)

        copyto!(mpcc.meta.x0, [2; 1])

        stats = MadMPEC.solve_homotopy!(solver)

        @test stats.status == MadNLP.SOLVE_SUCCEEDED
        @test stats.objective ≈ 1 atol=1e-5
        @test stats.solution ≈ [1, 0] atol=1e-5
    end

    @testset "With Magic Steps" begin
        mpcc = SimpleMPCCModel(Float64)
        madnlpc_opts =
            MadMPEC.MadNLPCOptions(; print_level=MadNLP.ERROR, use_magic_step=true)
        solver =
            MadMPEC.MadNLPCSolver(mpcc; madnlpc_opts=madnlpc_opts, print_level=MadNLP.ERROR)

        copyto!(mpcc.meta.x0, [2; 1])

        stats = MadMPEC.solve_homotopy!(solver)

        @test stats.status == MadNLP.SOLVE_SUCCEEDED
        @test stats.objective ≈ 1 atol=1e-5
        @test stats.solution ≈ [1, 0] atol=1e-5
    end

    @testset "Magic Steps Check" begin
        kappa = 1.0
        sigma1 = 0.1
        x1 = 0.5
        y1 = 2.0
        xp, yp = MadMPEC.project_scholtes_explicit(x1, y1, 0.0, 0.0, kappa, sigma1)
        @test xp ≈ 0.050285 atol=1e-5
        @test yp ≈ 1.988628 atol=1e-5
    end
end
