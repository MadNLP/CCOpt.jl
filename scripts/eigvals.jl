
using NLPModels
using SparseArrays
using LinearAlgebra

function analyze_kkt(solver)
    Ii = solver.ind_ineq
    Ie = setdiff(1:solver.m, Ii)
    m = solver.m
    nx = NLPModels.get_nvar(solver.nlp)
    ns = length(solver.ind_ineq)

    Ji = solver.kkt.jac_com[Ii, 1:nx]
    Je = solver.kkt.jac_com[Ie, 1:nx]
    W = solver.kkt.hess_com[1:nx, 1:nx]
    Σ = solver.kkt.hess_com[(nx+1):(nx+ns), (nx+1):(nx+ns)]

    δ = solver.kkt.reg[1]
    Σx = spdiagm(solver.kkt.pr_diag[1:nx] .- δ)
    Σs = spdiagm(solver.kkt.pr_diag[(nx+1):(nx+ns)] .- δ)
    K = Symmetric(W, :L) + Σx + Ji' * Σs * Ji

    Jd = Array(Je)

    s = svd(Jd).S
    si = svd(Array(Ji)).S

    Z = nullspace(Jd)
    H = Z' * K * Z
    spec = real.(eigvals(H))
    println("δ: $(δ)")
    println("Je: $(minimum(s)) $(maximum(s))")
    println("Ji: $(minimum(si)) $(maximum(si))")
    println("H: $(minimum(spec)) $(maximum(spec))")
    specK = real.(eigvals(Array(K)))
    println("K: $(minimum(specK)) $(maximum(specK))")
    specW = real.(eigvals(Array(Symmetric(W, :L))))
    println("W: $(minimum(specW)) $(maximum(specW))")
    C = Ji' * Σs * Ji
    specC = real.(eigvals(Array(C)))
    println("C: $(minimum(specC)) $(maximum(specC))")
    println("Σx: $(minimum(Σx)) $(maximum(Σx))")
    return println("Σs: $(minimum(Σs)) $(maximum(Σs))")
end
