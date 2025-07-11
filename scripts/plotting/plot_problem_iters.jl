using MadMPEC, Plots, LaTeXStrings, LinearAlgebra
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

function calculate_mpcc_multiplier_estimate(iters_fname::AbstractString; range=:)
    iters = readlog(iters_fname)
    z1 = [i.z1 for i in iters[range] if !i.magic]
    x1 = [i.x1 for i in iters[range] if !i.magic]
    z2 = [i.z2 for i in iters[range] if !i.magic]
    x2 = [i.x2 for i in iters[range] if !i.magic]
    zs = [i.zs for i in iters[range] if !i.magic]

    nu = map(
        (x1, x2, z1, z2, zs) -> -z1 .+ zs .* (x2 ./ (sqrt.(x1 .^ 2 .+ x2 .^ 2))),
        x1,
        x2,
        z1,
        z2,
        zs,
    )
    xi = map(
        (x1, x2, z1, z2, zs) -> -z2 .+ zs .* (x1 ./ (sqrt.(x1 .^ 2 .+ x2 .^ 2))),
        x1,
        x2,
        z1,
        z2,
        zs,
    )
    return iters, nu, xi
end

function plot_mpcc_multiplier_estimate_error(
    name::AbstractString,
    prob::MadMPEC.MPCCModel,
    iters_fname::AbstractString;
    range=:,
)
    iters, nu, xi = calculate_mpcc_multiplier_estimate(iters_fname; range=range)
    k_newton = [i.k for i in iters[range] if !i.magic]
    k_magic = [i.k for i in iters[range] if i.magic]
    k_mu = [iters[i].k for i in 1:(length(iters)-1) if iters[i+1].mu < iters[i].mu]

    nu_star, xi_star = nu[end], xi[end]

    nu_error = map((nu) -> norm(nu-nu_star), nu)
    xi_error = map((xi) -> norm(xi-xi_star), xi)
    println(nu)
    println(xi)

    nu_err_plt = plot(
        k_newton[1:(end-1)],
        nu_error[1:(end-1)],
        size=(1900, 400),
        yaxis=:log10,
        ylabel=L"$||\nu - \nu^*||_2$",
        legend=false,
        tickfontsize=15,
        bottommargin=20Plots.px,
        leftmargin=50Plots.px,
        labelfontsize=15,
        linetype=:steppre,
    )
    vline!(nu_err_plt, k_magic)
    vline!(nu_err_plt, k_mu)
    xi_err_plt = plot(
        k_newton[1:(end-1)],
        xi_error[1:(end-1)],
        size=(1900, 400),
        yaxis=:log10,
        ylabel=L"$||\xi - \xi^*||_2$",
        legend=false,
        tickfontsize=15,
        bottommargin=20Plots.px,
        leftmargin=50Plots.px,
        labelfontsize=15,
        linetype=:steppre,
    )
    vline!(xi_err_plt, k_magic)
    vline!(xi_err_plt, k_mu)
    err_plt = plot(
        nu_err_plt,
        xi_err_plt,
        layout=(2, 1),
        size=(1900, 1000),
        plot_title=name*" mpcc mult err",
    )
    return display(err_plt)
end

function diagonal_kkt_entries(
    name::AbstractString,
    prob::MadMPEC.MPCCModel,
    iters_fname::AbstractString;
    range=:,
    bound_relax=1e-12,
    save_plt=false,
    save_ext=".png",
)
    iters = readlog(iters_fname)

    k = [i.k for i in iters[range]]
    k_newton = [i.k for i in iters[range] if !i.magic]
    k_magic = [i.k for i in iters[range] if i.magic]
    k_mu = [iters[i].k for i in 1:(length(iters)-1) if iters[i+1].mu < iters[i].mu]

    mu = [i.mu for i in iters[range]]

    lbx1 = prob.meta.lvar[prob.meta.ind_cc1] .- bound_relax
    lbx2 = prob.meta.lvar[prob.meta.ind_cc2] .- bound_relax

    z1 = [i.z1 for i in iters[range]]
    x1 = [i.x1 .- lbx1 for i in iters[range]]
    z2 = [i.z2 for i in iters[range]]
    x2 = [i.x2 .- lbx2 for i in iters[range]]
    s = [-i.s .+ bound_relax for i in iters[range]]
    zs = [i.zs for i in iters[range]]

    X1Z1 = [(1 ./ x1) .* z1 for (x1, z1) in zip(x1, z1)]
    X2Z2 = [(1 ./ x2) .* z2 for (x2, z2) in zip(x2, z2)]
    X2SZS = [x2 .^ 2 .* (1 ./ s) .* zs for (x2, s, zs) in zip(x2, s, zs)]
    X1SZS = [x1 .^ 2 .* (1 ./ s) .* zs for (x1, s, zs) in zip(x1, s, zs)]

    𝛴11 = [x1z1 .+ x2szs for (x1z1, x2szs) in zip(X1Z1, X2SZS)]
    𝛴22 = [x2z2 .+ x1szs for (x2z2, x1szs) in zip(X2Z2, X1SZS)]

    x1z1_plt = plot(
        k,
        [minimum(a) for a in X1Z1],
        yaxis=:log10,
        linetype=:steppre,
        legend=true,
        label=L"$\min(\frac{z_1}{x_1})$",
    )
    plot!(
        x1z1_plt,
        k,
        [maximum(a) for a in X1Z1],
        linetype=:steppre,
        label=L"$\min(\frac{z_1}{x_1})$",
    )
    plot!(x1z1_plt, k, 1 ./ mu, linetype=:steppre, legend=true, label=L"$\frac{1}{\mu}$")
    plot!(x1z1_plt, k, mu, linetype=:steppre, legend=true, label=L"$\mu$")
    vline!(x1z1_plt, k_magic, linestyle=:dash, label="")
    vline!(x1z1_plt, k_mu, linestyle=:dash, label="")
    if save_plt
        savefig(name*iters_fname*"_x1z1"*save_ext)
    end

    x2z2_plt = plot(
        k,
        [minimum(a) for a in X2Z2],
        yaxis=:log10,
        linetype=:steppre,
        legend=true,
        label=L"$\min(\frac{z_2}{x_2})$",
    )
    plot!(
        x2z2_plt,
        k,
        [maximum(a) for a in X2Z2],
        linetype=:steppre,
        legend=true,
        label=L"$\max(\frac{z_2}{x_2})$",
    )
    plot!(x2z2_plt, k, 1 ./ mu, linetype=:steppre, legend=true, label=L"$\frac{1}{\mu}$")
    plot!(x2z2_plt, k, mu, linetype=:steppre, legend=true, label=L"$\mu$")
    vline!(x2z2_plt, k_magic, linestyle=:dash, label="")
    vline!(x2z2_plt, k_mu, linestyle=:dash, label="")
    if save_plt
        savefig(name*iters_fname*"_x2z2"*save_ext)
    end

    x2szs_plt = plot(
        k,
        [minimum(a) for a in X2SZS],
        yaxis=:log10,
        linetype=:steppre,
        legend=true,
        label=L"$\min(x_2^2\frac{z_s}{s})$",
    )
    plot!(
        x2szs_plt,
        k,
        [maximum(a) for a in X2SZS],
        linetype=:steppre,
        legend=true,
        label=L"$\max(x_2^2\frac{z_s}{s})$",
    )
    plot!(x2szs_plt, k, 1 ./ mu, linetype=:steppre, legend=true, label=L"$\frac{1}{\mu}$")
    plot!(x2szs_plt, k, mu, linetype=:steppre, legend=true, label=L"$\mu$")
    vline!(x2szs_plt, k_magic, linestyle=:dash, label="")
    vline!(x2szs_plt, k_mu, linestyle=:dash, label="")
    if save_plt
        savefig(name*iters_fname*"_x2szs"*save_ext)
    end

    x1szs_plt = plot(
        k,
        [minimum(a) for a in X1SZS],
        yaxis=:log10,
        linetype=:steppre,
        legend=true,
        label=L"$\min(x_1^2\frac{z_s}{s})$",
    )
    plot!(
        x1szs_plt,
        k,
        [maximum(a) for a in X1SZS],
        linetype=:steppre,
        legend=true,
        label=L"$\max(x_1^2\frac{z_s}{s})$",
    )
    plot!(x1szs_plt, k, 1 ./ mu, linetype=:steppre, legend=true, label=L"$\frac{1}{\mu}$")
    plot!(x1szs_plt, k, mu, linetype=:steppre, legend=true, label=L"$\mu$")
    vline!(x1szs_plt, k_magic, linestyle=:dash, label="")
    vline!(x1szs_plt, k_mu, linestyle=:dash, label="")
    if save_plt
        savefig(name*iters_fname*"_x1szs"*save_ext)
    end

    𝛴11_plt = plot(
        k,
        [minimum(a) for a in 𝛴11],
        yaxis=:log10,
        linetype=:steppre,
        legend=true,
        label=L"$\min(\Sigma_{11})$",
    )
    plot!(
        𝛴11_plt,
        k,
        [maximum(a) for a in 𝛴11],
        linetype=:steppre,
        legend=true,
        label=L"$\max(\Sigma_{11})$",
    )
    plot!(𝛴11_plt, k, 1 ./ mu, linetype=:steppre, legend=true, label=L"$\frac{1}{\mu}$")
    plot!(𝛴11_plt, k, mu, linetype=:steppre, legend=true, label=L"$\mu$")
    vline!(𝛴11_plt, k_magic, linestyle=:dash, label="")
    vline!(𝛴11_plt, k_mu, linestyle=:dash, label="")
    if save_plt
        savefig(name*iters_fname*"_Sigma11"*save_ext)
    end

    𝛴22_plt = plot(
        k,
        [minimum(a) for a in 𝛴22],
        yaxis=:log10,
        linetype=:steppre,
        legend=true,
        label=L"$\min(\Sigma_{22})$",
    )
    plot!(
        𝛴22_plt,
        k,
        [maximum(a) for a in 𝛴22],
        linetype=:steppre,
        legend=true,
        label=L"$\max(\Sigma_{22})$",
    )
    plot!(𝛴22_plt, k, 1 ./ mu, linetype=:steppre, legend=true, label=L"$\frac{1}{\mu}$")
    plot!(𝛴22_plt, k, mu, linetype=:steppre, legend=true, label=L"$\mu$")
    vline!(𝛴22_plt, k_magic, linestyle=:dash, label="")
    vline!(𝛴22_plt, k_mu, linestyle=:dash, label="")
    if save_plt
        savefig(name*iters_fname*"_Sigma22"*save_ext)
    end

    return display(𝛴11_plt)
end
