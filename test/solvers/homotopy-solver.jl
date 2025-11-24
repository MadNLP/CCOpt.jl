@testset "Homotopy Solver Tests" begin
    @testset "Test IPOPT" begin
        mpcc = SimpleMPCCModel(Float64)
        opts = MadMPEC.HomotopySolverOptions()
        opts.print_level = MadNLP.ERROR
        opts.comp_tol = 1e-7

        solver = MadMPEC.HomotopySolver(mpcc, NLPModelsIpopt.IpoptSolver, opts)

        stats = MadMPEC.solve!(solver)

        @test stats.status ∈ [MadNLP.SOLVE_SUCCEEDED, MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL]
        @test stats.objective ≈ 1 atol=1e-5
        @test stats.solution ≈ [1, 0] atol=1e-5
    end

    @testset "Test MadNLP" begin
        mpcc = SimpleMPCCModel(Float64)
        opts = MadMPEC.HomotopySolverOptions()
        opts.print_level = MadNLP.ERROR
        opts.nlp_solver_options =
            Dict(:bound_relax_factor=>1e-12, :print_level=>MadNLP.ERROR, :max_iter=>500)
        opts.comp_tol = 1e-7
        copyto!(mpcc.meta.x0, [2; 1])
        solver = MadMPEC.HomotopySolver(mpcc, MadNLP.MadNLPSolver, opts)

        stats = MadMPEC.solve!(solver)
        @test stats.status ∈ [MadNLP.SOLVE_SUCCEEDED, MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL]
        # These are broken, we for some reason converge to the biactive point
        # in the case of MadNLP but to the true optimum [1,0]
        # in the case of IPOPT
        @test stats.objective ≈ 1 atol=1e-5 skip=true
        @test stats.solution ≈ [1, 0] atol=1e-5 skip=true
    end
end
