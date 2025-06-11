@testset "ExactPenalty Test" begin
    @testset "Default Options" begin
        mpcc = SimpleMPCCModel(Float64)
        madnlpell1_opts = MadMPEC.ExactPenaltyOptions(; print_level=MadNLP.ERROR)
        solver = MadMPEC.ExactPenaltySolver(
            mpcc;
            madnlpell1_opts=madnlpell1_opts,
            print_level=MadNLP.ERROR,
        )

        copyto!(mpcc.meta.x0, [2; 1])

        stats = MadMPEC.solve_homotopy!(solver)

        @test stats.status == MadNLP.SOLVE_SUCCEEDED
        @test stats.objective ≈ 1 atol=1e-5
        @test stats.solution ≈ [1, 0] atol=1e-5
    end

    @testset "With dynamic update" begin
        mpcc = SimpleMPCCModel(Float64)
        madnlpell1_opts = MadMPEC.ExactPenaltyOptions(;
            print_level=MadNLP.ERROR,
            dynamic_sigma_update=true,
        )
        solver = MadMPEC.ExactPenaltySolver(
            mpcc;
            madnlpell1_opts=madnlpell1_opts,
            print_level=MadNLP.ERROR,
        )

        copyto!(mpcc.meta.x0, [2; 1])

        stats = MadMPEC.solve_homotopy!(solver)

        @test stats.status == MadNLP.SOLVE_SUCCEEDED
        @test stats.objective ≈ 1 atol=1e-5
        @test stats.solution ≈ [1, 0] atol=1e-5
    end
end
