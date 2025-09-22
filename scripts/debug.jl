function get_lams(solver)
    # Determine if KKT system is well conditioned
    W = solver.kkt.aug_com
    K = Array(Symmetric(W, :L))
    s = eigvals(K)

    s_pos = s[s .> 0]
    s_neg = s[s .< 0]
    println("λ+min: ", s_pos[1])
    println("λ+max: ", s_pos[end])
    println("λ-min: ", s_neg[end])
    return println("λ-max: ", s_neg[1])
end
