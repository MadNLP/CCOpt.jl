using Plots, LaTeXStrings

# NOTE: This is only for plotting the basic example:
let x1 = [],
    x2 = [],
    z1 = [],
    z2 = [],
    s = Vector{Float64}(),
    zs = Vector{Float64}(),
    σ = 1,
    primal_plot = plot(; xlims=(0, 3), ylims=(0, 3), aspect_ratio=:equal),
    colors = [],
    anim_plots = []

    global function reset_plot()
        x1 = []
        x2 = []
        z1 = []
        z2 = []
        s = []
        zs = []
        colors = []
        σ = 1
        anim_plots = []
        return nothing
    end

    global function plot_complementarities(solver::MadNLPCSolver; magic_step=false)
        if magic_step
            push!(colors, 1.0)
        else
            push!(colors, 0.0)
        end
        # NOTE shifted by bound relaxation
        push!(x1, MadNLP.variable(solver.ipm.x)[1])
        push!(x2, MadNLP.variable(solver.ipm.x)[2])
        push!(z1, MadNLP.variable(solver.ipm.zl)[1])
        push!(z2, MadNLP.variable(solver.ipm.zl)[2])
        push!(s, -MadNLP.slack(solver.ipm.x)[end])
        push!(zs, MadNLP.slack(solver.ipm.zu)[end])
        σ = solver.scholtes.𝜎[]
        println(colors)
        pxx = plot(
            x1,
            x2;
            legend=true,
            markershape=:circle,
            markersize=2,
            xlims=(1e-11, 100),
            ylims=(1e-11, 100),
            line_z=(length(colors) > 1 ? colors[2:end] : [1]),
            linecolor=palette([:blue, :green], 2),
            show=false,
            aspect_ratio=:equal,
            xaxis=:log10,
            yaxis=:log10,
            colorbar=false,
            xlabel=L"$x_1$",
            ylabel=L"$x_2$",
        )
        plot!(pxx, (x) -> σ/x, 1e-12:0.01:10; legend=false)

        pz1 = plot(
            x1,
            z1;
            legend=true,
            markershape=:circle,
            markersize=2,
            xlims=(1e-11, 100),
            ylims=(1e-11, 100),
            line_z=(length(colors) > 1 ? colors[2:end] : [1]),
            linecolor=palette([:blue, :green], 2),
            show=false,
            aspect_ratio=:equal,
            xaxis=:log10,
            yaxis=:log10,
            colorbar=false,
            xlabel=L"$x_1$",
            ylabel=L"$z_1$",
        )
        plot!(pz1, (x) -> σ/x, 1e-12:0.01:10; legend=false)

        pz2 = plot(
            x2,
            z2;
            legend=true,
            markershape=:circle,
            markersize=2,
            xlims=(1e-11, 100),
            ylims=(1e-11, 100),
            line_z=(length(colors) > 1 ? colors[2:end] : [1]),
            linecolor=palette([:blue, :green], 2),
            show=false,
            aspect_ratio=:equal,
            xaxis=:log10,
            yaxis=:log10,
            colorbar=false,
            xlabel=L"$x_2$",
            ylabel=L"$z_2$",
        )
        plot!(pz2, (x) -> σ/x, 1e-12:0.01:10; legend=false)

        pzs = plot(
            clamp.(s, 5e-10, Inf),
            clamp.(zs, 5e-10, Inf);
            legend=true,
            markershape=:circle,
            markersize=2,
            xlims=(1e-11, 100),
            ylims=(1e-11, 100),
            line_z=(length(colors) > 1 ? colors[2:end] : [1]),
            linecolor=palette([:blue, :green], 2),
            show=false,
            aspect_ratio=:equal,
            xaxis=:log10,
            yaxis=:log10,
            colorbar=false,
            xlabel=L"$s$",
            ylabel=L"$z_s$",
        )
        plot!(pzs, (x) -> σ/x, 1e-12:0.01:10; legend=false)

        plt = plot(pxx, pz1, pz2, pzs; layout=(2, 2), size=(900, 900))
        push!(anim_plots, plt)
        display(plt)

        #Plots.xaxis!(pxx, scale=:log10)
        #Plots.yaxis!(pxx, scale=:log10)
        # plot!(primal_plot,
        #       x1, z1;
        #       layout=l,
        #       subplot=2,
        #       markershape=:circle,
        #       xlims=(1e-12, 100), ylims=(1e-12, 100),
        #       linecolor=colors[2:end])
        #display()
        #plot!(primal_plot, (x) -> σ/x,  1e-12:0.01:10; layout=l, show=true, legend=false)
        return wait_for_key("press any key to continue...")
    end

    global function finalize_animation()
        return anim = animate(anim_plots, "convergence.gif", fps=1)
    end
end
