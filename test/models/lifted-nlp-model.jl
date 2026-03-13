@testset "Lifted NLP Model Tests" begin
    @testset "API" for T in [Float64, Float32], M in [NLPModelMeta, SimpleNLPMeta]
        f(x) = (x[1] - 2)^2 + (x[2] - 1)^2
        ∇f(x) = T[2 * (x[1] - 2); 2 * (x[2] - 1); 0]
        H(x) = T[2.0 0 0; 0 2.0 0; 0 0 0]
        c(x) = T[x[1] - 2x[2] + 1; -x[1]^2 / 4 - x[2]^2 + 1 - x[3]]
        J(x) = T[1.0 -2.0 0; -0.5x[1] -2.0x[2] -1]
        H(x, y) = H(x) + y[2] * T[-0.5 0 0; 0 -2.0 0; 0 0 0]

        snlp = SimpleNLPModel(T, M)
        nlp = CCOpt.LiftedNLPModel(snlp, [2])
        n = get_nvar(nlp)
        m = get_ncon(nlp)
        @test n == 3
        @test m == 2

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
