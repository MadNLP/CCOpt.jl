using MadMPEC, Plots, LaTeXStrings
include("../readlog.jl")

function plot_solver_traj(
    name::AbstractString,
    prob::MadMPEC.MPCCModel,
    iters_fname::AbstractString;
    range=:,
    save_plt=false,
    save_ext=".png",
)
    iters = readlog(iters_fname)
    k_newton = [i.k for i in iters[range] if !i.magic]
    k_magic = [i.k for i in iters[range] if i.magic]
    k_mu = [iters[i].k for i in 1:(length(iters)-1) if iters[i+1].mu < iters[i].mu]
    apr = [clamp(i.alpha_pr, 1e-12, Inf) for i in iters[range] if !i.magic]
    adu = [clamp(i.alpha_du, 1e-12, Inf) for i in iters[range] if !i.magic]
    inf_pr = [clamp(i.inf_pr, 1e-12, Inf) for i in iters[range] if !i.magic]
    inf_du = [clamp(i.inf_du, 1e-12, Inf) for i in iters[range] if !i.magic]
    obj = [i.obj for i in iters[range] if !i.magic]
    mu = [i.mu for i in iters[range] if !i.magic]
    sp_min = [minimum(i.KKT_s[i.KKT_s .> 0]) for i in iters[range] if !i.magic]
    sp_max = [maximum(i.KKT_s[i.KKT_s .> 0]) for i in iters[range] if !i.magic]
    sn_min = [minimum(i.KKT_s[i.KKT_s .< 0]) for i in iters[range] if !i.magic]
    sn_max = [maximum(i.KKT_s[i.KKT_s .< 0]) for i in iters[range] if !i.magic]

    println(sp_min)
    println(sp_max)
    println(sn_min)
    println(sn_max)

    apr_plt = plot(
        k_newton,
        apr,
        size=(1900, 400),
        ylim=(1e-5, 1.1),
        yaxis=:log10,
        ylabel=L"$\alpha_{pr}$",
        legend=false,
        tickfontsize=15,
        bottommargin=20Plots.px,
        leftmargin=50Plots.px,
        labelfontsize=15,
        linetype=:steppre,
    )
    vline!(apr_plt, k_magic)
    vline!(apr_plt, k_mu)
    adu_plt = plot(
        k_newton,
        adu,
        size=(1900, 400),
        ylim=(1e-5, 1.1),
        yaxis=:log10,
        ylabel=L"$\alpha_{du}$",
        legend=false,
        tickfontsize=15,
        bottommargin=20Plots.px,
        leftmargin=50Plots.px,
        labelfontsize=15,
        linetype=:steppre,
    )
    vline!(adu_plt, k_magic)
    vline!(adu_plt, k_mu)
    a_plt =
        plot(apr_plt, adu_plt, layout=(2, 1), size=(1900, 1000), plot_title=name*" alpha")
    display(a_plt)
    if save_plt
        savefig(name*iters_fname*"_alpha"*save_ext)
    end

    inf_pr_plt = plot(
        k_newton,
        inf_pr,
        size=(1900, 400),
        yaxis=:log10,
        ylabel="inf_pr",
        legend=false,
        tickfontsize=15,
        bottommargin=20Plots.px,
        leftmargin=20Plots.px,
        labelfontsize=15,
        linetype=:steppre,
    )
    vline!(inf_pr_plt, k_magic)
    vline!(inf_pr_plt, k_mu)
    inf_du_plt = plot(
        k_newton,
        inf_du,
        size=(1900, 400),
        yaxis=:log10,
        ylabel="inf_du",
        legend=false,
        tickfontsize=15,
        bottommargin=20Plots.px,
        leftmargin=50Plots.px,
        labelfontsize=15,
        linetype=:steppre,
    )
    vline!(inf_du_plt, k_magic)
    vline!(inf_du_plt, k_mu)
    inf_plt = plot(
        inf_pr_plt,
        inf_du_plt,
        layout=(2, 1),
        size=(1900, 1000),
        plot_title=name*" infeasibility",
    )
    display(inf_plt)
    if save_plt
        savefig(name*iters_fname*"_inf"*save_ext)
    end

    obj_plt = plot(
        k_newton,
        obj,
        size=(1900, 400),
        ylabel="obj",
        legend=false,
        tickfontsize=15,
        bottommargin=20Plots.px,
        leftmargin=50Plots.px,
        labelfontsize=15,
        linetype=:steppre,
        plot_title=name*" objective",
    )
    display(obj_plt)
    if save_plt
        savefig(name*iters_fname*"_obj"*save_ext)
    end

    mu_plt = plot(
        k_newton,
        mu,
        size=(1900, 400),
        yaxis=:log10,
        ylabel="mu",
        legend=false,
        tickfontsize=15,
        bottommargin=20Plots.px,
        leftmargin=50Plots.px,
        labelfontsize=15,
        linetype=:steppre,
        plot_title=name*" mu",
    )
    display(mu_plt)
    if save_plt
        savefig(name*iters_fname*"_mu"*save_ext)
    end

    s_plt = plot(
        k_newton,
        [sp_min, sp_max, .-sn_min, .-sn_max],
        size=(1900, 700),
        yaxis=:log10,
        legend=true,
        labels=[L"\lambda^+_{min}" L"\lambda^+_{max}" L"\lambda^-_{min}" L"\lambda^-_{max}"],
        legendfontsize=15,
        tickfontsize=15,
        bottommargin=20Plots.px,
        leftmargin=50Plots.px,
        labelfontsize=15,
        linetype=:steppre,
    )
    display(s_plt)
    if save_plt
        savefig(name*iters_fname*"_lam"*save_ext)
    end
    return nothing
end
