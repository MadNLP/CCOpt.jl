@testset "Scholtes Relaxation Tests" begin
    @testset "API" for T in [Float64, Float32]
        f(x) = (x[1] - 1)^2 + (x[2] - 1)^2
        ∇f(x) = T[2 * (x[1] - 1); 2 * (x[2] - 1)]
        H(x) = T[2.0 0; 0 2.0]
        c(x; σ=0) = T[x[1]*x[2]-σ]
        J(x) = hcat(x[2], x[1])
        H(x, y) = H(x) + y[1] * T[0 1; 1 0]

        snlp = SimpleMPCCModel(T)
        nlp = CCOpt.ScholtesRelaxation(snlp)
        n = get_nvar(nlp)
        m = get_ncon(nlp)
        @test n == 2
        @test m == 1
        @test get_lcon(nlp) ≈ [typemin(T)]
        @test get_ucon(nlp) ≈ [0.0]

        x = randn(T, n)
        y = randn(T, m)
        v = randn(T, n)
        w = randn(T, m)
        Jv = zeros(T, m)
        Jtw = zeros(T, n)
        Hv = zeros(T, n)
        Hvals = zeros(T, get_nnzh(nlp))

        @test obj(nlp, x) ≈ f(x)
        @test grad(nlp, x) ≈ ∇f(x)
        @test hess(nlp, x) ≈ H(x)
        @test hprod(nlp, x, v) ≈ H(x) * v
        @test cons(nlp, x) ≈ c(x)
        @test jac(nlp, x) ≈ J(x)
        @test jprod(nlp, x, v) ≈ J(x) * v
        @test jtprod(nlp, x, w) ≈ J(x)' * w
        @test hess(nlp, x, y) ≈ H(x, y)
        @test hprod(nlp, x, y, v) ≈ H(x, y) * v
    end
end
