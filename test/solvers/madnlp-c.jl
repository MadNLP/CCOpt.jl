@testset "MadNLP-C Test" begin
    mpcc = SimpleMPCCModel(Float64)
    scholtes = MadMPEC.ScholtesRelaxation(mpcc)
    solver = MadNLP.MadNLPSolver(scholtes)

    copyto!(scholtes.meta.x0, [2; 1])

    stats = MadMPEC.solve_homotopy!(solver)

    @test stats.status == MadNLP.SOLVE_SUCCEEDED
    @test stats.objective ≈ 1 atol=1e-5
    @test stats.solution ≈ [1, 0] atol=1e-5
end
