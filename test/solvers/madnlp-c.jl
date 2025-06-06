@testset "MadNLP-C Test" begin
    @testset "Default Options" begin
        mpcc = SimpleMPCCModel(Float64)
        solver = MadMPEC.MadNLPCSolver(mpcc; print_level=MadNLP.INFO)

        copyto!(mpcc.meta.x0, [2; 1])

        stats = MadMPEC.solve_homotopy!(solver)

        @test stats.status == MadNLP.SOLVE_SUCCEEDED
        @test stats.objective ≈ 1 atol=1e-5
        @test stats.solution ≈ [1, 0] atol=1e-5
    end

    @testset "With Magic Steps" begin
        mpcc = SimpleMPCCModel(Float64)
        madnlpc_opts = MadMPEC.MadNLPCOptions(; use_magic_step=true)
        solver =
            MadMPEC.MadNLPCSolver(mpcc; madnlpc_opts=madnlpc_opts, print_level=MadNLP.INFO)

        copyto!(mpcc.meta.x0, [2; 1])

        stats = MadMPEC.solve_homotopy!(solver)

        @test stats.status == MadNLP.SOLVE_SUCCEEDED
        @test stats.objective ≈ 1 atol=1e-5
        @test stats.solution ≈ [1, 0] atol=1e-5
    end
end
