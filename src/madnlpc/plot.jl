using Plots

wait_for_key(prompt) = (print(stdout, prompt); read(stdin, 1); nothing)

# NOTE: This is only for plotting the basic example:
let x1 = [],
    x2 = [],
    𝜎 = 1,
    primal_plot = plot(; xlims=(0, 3), ylims=(0, 3), aspect_ratio=:equal)

    global function reset_plot()
        x1 = []
        x2 = []
        𝜎 = 1
        primal_plot = plot(; xlims=(1e-12, 10), ylims=(1e-12, 10), aspect_ratio=:equal)
        Plots.xaxis!(primal_plot, scale=:log10)
        return Plots.yaxis!(primal_plot, scale=:log10)
    end
    global function plot_complementarities(solver::MadNLPCSolver)
        push!(x1, MadNLP.variable(solver.ipm.x)[1])
        push!(x2, MadNLP.variable(solver.ipm.x)[2])
        𝜎 = solver.scholtes.𝜎[]
        plot(primal_plot, x1, x2; markershape=:circle, show=true)
        plot(primal_plot, (x) -> 𝜎/x, 1e-12:0.01:10;)
        return wait_for_key("press any key to continue...")
    end
end
