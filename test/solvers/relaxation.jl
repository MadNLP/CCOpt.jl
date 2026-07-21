@testset "RelaxationSolver Test" begin
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
end
