@testset "ExactPenalty Test" begin
    @testset "Default Options" begin
        mpcc = SimpleMPCCModel2(Float64)
        madnlpell1_opts = MadMPEC.ExactPenaltyOptions{Float64}(; print_level=MadNLP.ERROR)
        solver = MadMPEC.ExactPenaltySolver(
            mpcc;
            solver_opts=madnlpell1_opts,
            print_level=MadNLP.ERROR,
        )

        stats = MadMPEC.solve_homotopy!(solver)

        @test stats.status == MadNLP.SOLVE_SUCCEEDED
        @test stats.objective ≈ 0.01 atol=1e-5
        @test stats.solution ≈ [0, 1] atol=1e-5
    end

    @testset "With dynamic update" begin
        mpcc = SimpleMPCCModel2(Float64)
        madnlpell1_opts = MadMPEC.ExactPenaltyOptions{Float64}(;
            print_level=MadNLP.ERROR,
            dynamic_rho_update=true,
        )
        solver = MadMPEC.ExactPenaltySolver(
            mpcc;
            solver_opts=madnlpell1_opts,
            print_level=MadNLP.ERROR,
        )

        stats = MadMPEC.solve_homotopy!(solver)

        @test stats.status == MadNLP.SOLVE_SUCCEEDED
        @test stats.objective ≈ 0.01 atol=1e-5
        @test stats.solution ≈ [0, 1] atol=1e-5
    end
end
