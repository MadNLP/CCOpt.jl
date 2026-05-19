using CCOpt, Plots, LaTeXStrings, LinearAlgebra, PythonPlot
import Plots: plot
include("../readlog.jl")

function plot_solver_traj_for_paper(
    name::AbstractString,
    prob::MPCCModel,
    iters_fname::AbstractString;
    range=:,
    save_ext=".pdf",
    plot_k_mu=true,
    save_plt=false,
    xlim=nothing,
)
    default()
    default(
        titlefont=(20, "serif"),
        legendfont=(12, "serif"),
        guidefont=(12, "serif"),
        tickfontsize=12,
        linewidth=3,
    )
    pythonplot()
    iters = readlog(iters_fname)
    k_newton = [i.k for i in iters[range] if !i.magic];
    push!(k_newton, k_newton[end]+1)
    k_magic = [i.k for i in iters[range] if i.magic]
    k_mu = [iters[i].k for i in 1:(length(iters)-1) if iters[i+1].mu < iters[i].mu]
    apr = [clamp(i.alpha_pr, 1e-12, Inf) for i in iters[range] if !i.magic];
    push!(apr, apr[end])
    adu = [clamp(i.alpha_du, 1e-12, Inf) for i in iters[range] if !i.magic];
    push!(adu, adu[end])
    inf_pr = [clamp(i.inf_pr, 1e-12, Inf) for i in iters[range] if !i.magic];
    push!(inf_pr, inf_pr[end])
    inf_du = [clamp(i.inf_du, 1e-12, Inf) for i in iters[range] if !i.magic];
    push!(inf_du, inf_du[end])
    inf_compl = [clamp(i.inf_compl, 1e-12, Inf) for i in iters[range] if !i.magic];
    push!(inf_compl, inf_compl[end])
    inf_rnlp = [clamp(i.inf_rnlp, 1e-12, Inf) for i in iters[range] if !i.magic];
    push!(inf_rnlp, inf_rnlp[end])
    inf_pr_cc = [clamp(i.inf_pr_cc, 1e-12, Inf) for i in iters[range] if !i.magic];
    push!(inf_pr_cc, inf_pr_cc[end])
    obj = [i.obj for i in iters[range] if !i.magic];
    push!(obj, obj[end])
    mu = [i.mu for i in iters[range] if !i.magic];
    push!(mu, mu[end])
    sp_min = [minimum(i.KKT_s[i.KKT_s .> 0]) for i in iters[range] if !i.magic];
    push!(sp_min, sp_min[end])
    sp_max = [maximum(i.KKT_s[i.KKT_s .> 0]) for i in iters[range] if !i.magic];
    push!(sp_max, sp_max[end])
    sn_min = [minimum(i.KKT_s[i.KKT_s .< 0]) for i in iters[range] if !i.magic];
    push!(sn_min, sn_min[end])
    sn_max = [maximum(i.KKT_s[i.KKT_s .< 0]) for i in iters[range] if !i.magic];
    push!(sn_max, sn_max[end])
    n_fact = [i.n_fact for i in iters[range] if !i.magic];
    n_fact_diff = diff(n_fact)
    push!(n_fact, n_fact[end])

    a_plt = plot(
        k_newton[2:end],
        [apr[2:end], adu[2:end]],
        size=(1000, 400),
        ylim=(1e-5, 9),
        xlim=(1, maximum(k_newton)),
        yaxis=:log10,
        ylabel=L"\alpha",
        xlabel="Iteration",
        legend=:bottomright,
        label=[L"\alpha_{\mathrm{pr}}" L"\alpha_{\mathrm{du}}"],
        linetype=:steppost,
    )
    plot_k_mu && vline!(a_plt, k_mu; style=:dot, label="")

    display(a_plt)
    if save_plt
        PythonPlot.savefig(name*"_"*iters_fname*"_alpha"*save_ext)
    end

    s_plt = plot(
        k_newton[2:end],
        [sp_min[2:end], sp_max[2:end], .-sn_min[2:end], .-sn_max[2:end]],
        size=(1000, 400),
        xlim=(1, maximum(k_newton)),
        yaxis=:log10,
        yticks=[1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8, 1e10],
        legend=:topleft,
        legend_column=-1,
        labels=[L"\lambda^+_{min}" L"\lambda^+_{max}" L"\lambda^-_{min}" L"\lambda^-_{max}"],
        ylabel="KKT matrix eigenvalues",
        xlabel="Iterations",
        linetype=:steppost,
        reuse=false,
    )
    display(s_plt)
    if save_plt
        PythonPlot.savefig(name*"_"*iters_fname*"_lam"*save_ext)
    end

    k_plt = plot(
        k_newton[2:end],
        max.(sp_max[2:end], sn_max[2:end]) ./ (min.(sp_min[2:end], .-sn_min[2:end])),
        size=(1000, 400),
        xlim=(1, maximum(k_newton)),
        yaxis=:log10,
        legend=false,
        ylabel=L"\kappa(K)",
        xlabel="Iterations",
        linetype=:steppost,
        reuse=false,
    )
    display(k_plt)
    if save_plt
        PythonPlot.savefig(name*"_"*iters_fname*"_cond"*save_ext)
    end

    inf_plt = plot(
        k_newton[2:end],
        [inf_pr[2:end], inf_du[2:end], inf_compl[2:end], inf_pr_cc[2:end]],
        size=(600, 400),
        xlim=xlim,
        yaxis=:log10,
        ylabel="Primal-Dual Infeasibility",
        xlabel="Iterations",
        labels=[L"||c(x,s)||" L"||\mathcal{L}(x,s,y,z)||" L"||Xz-\mu||" L"||X_1 x_2||"],
        color=[:blue :red :orange :purple],
        legend=:topright,
        linetype=:steppost,
        reuse=false,
    )

    plot_k_mu && vline!(inf_plt, k_mu; style=:dot, color=:green, label="")
    display(inf_plt)
    if save_plt
        PythonPlot.savefig(name*"_"*iters_fname*"_inf"*save_ext)
    end

    obj_plt = plot(
        k_newton[2:end],
        obj[2:end],
        xlim=xlim,
        size=(600, 400),
        ylabel=L"f(x)",
        xlabel="Iteration",
        legend=false,
        linetype=:steppost,
        reuse=false,
    )
    display(obj_plt)
    if save_plt
        PythonPlot.savefig(name*"_"*iters_fname*"_obj"*save_ext)
    end

    fact_plt = plot(
        k_newton[2:end],
        n_fact[2:end],
        size=(1000, 400),
        ylabel="# of KKT factorizations",
        xlabel="Iteration",
        legend=false,
        linetype=:steppost,
        reuse=false,
    )
    display(fact_plt)

    if save_plt
        PythonPlot.savefig(name*"_"*iters_fname*"_n_fact"*save_ext)
    end
end

function plot_solver_traj(
    name::AbstractString,
    prob::MPCCModel,
    iters_fname::AbstractString;
    range=:,
    save_plt=false,
    save_ext=".png",
    plt_mu_updates=true,
)
    pythonplot()
    iters = readlog(iters_fname)
    k_newton = [i.k for i in iters[range] if !i.magic]
    k_magic = [i.k for i in iters[range] if i.magic]
    k_mu =
        plt_mu_updates ?
        [iters[i].k for i in 1:(length(iters)-1) if iters[i+1].mu < iters[i].mu] : []
    apr = [clamp(i.alpha_pr, 1e-12, Inf) for i in iters[range] if !i.magic]
    adu = [clamp(i.alpha_du, 1e-12, Inf) for i in iters[range] if !i.magic]
    inf_pr = [clamp(i.inf_pr, 1e-12, Inf) for i in iters[range] if !i.magic]
    inf_du = [clamp(i.inf_du, 1e-12, Inf) for i in iters[range] if !i.magic]
    inf_cc = [clamp(dot(i.x1, i.x2), 1e-12, Inf) for i in iters[range] if !i.magic]
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
    inf_cc_plt = plot(
        k_newton,
        inf_cc,
        size=(1900, 400),
        yaxis=:log10,
        ylabel="inf_cc",
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
        inf_cc_plt,
        layout=(3, 1),
        size=(1900, 1000),
        plot_title=name*" infeasibility",
        reuse=false,
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
        reuse=false,
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
        ylim=(1e-12, 10),
        ylabel="mu",
        legend=false,
        tickfontsize=15,
        bottommargin=20Plots.px,
        leftmargin=50Plots.px,
        labelfontsize=15,
        linetype=:steppre,
        plot_title=name*" mu",
        reuse=false,
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
        reuse=false,
    )
    display(s_plt)
    if save_plt
        savefig(name*iters_fname*"_lam"*save_ext)
    end
    return nothing
end

function calculate_mpcc_multiplier_estimate(iters_fname::AbstractString; range=:, filt=true)
    iters = readlog(iters_fname)
    if filt
        nu1 = [i.nu1_filt for i in iters[range] if !i.magic]
        nu2 = [i.nu2_filt for i in iters[range] if !i.magic]
    else
        nu1 = [i.nu1 for i in iters[range] if !i.magic]
        nu2 = [i.nu2 for i in iters[range] if !i.magic]
    end

    return iters, hcat(nu1...)', hcat(nu2...)'
end

function plot_mpcc_multiplier_estimate_error(
    name::AbstractString,
    prob::MPCCModel,
    iters_fname::AbstractString;
    range=:,
)
    iters, nu, nu2 = calculate_mpcc_multiplier_estimate(iters_fname; range=range)
    k_newton = [i.k for i in iters[range] if !i.magic]
    k_magic = [i.k for i in iters[range] if i.magic]
    k_mu = [iters[i].k for i in 1:(length(iters)-1) if iters[i+1].mu < iters[i].mu]

    nu_star, nu2_star = nu[end], nu2[end]

    nu_error = map((nu) -> norm(nu-nu_star), nu)
    nu2_error = map((nu2) -> norm(nu2-nu2_star), nu2)
    println(nu)
    println(nu2)

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

symlog(x, n=-12) = sign(x)*(log(10, 1+abs(x)/(10.0^n)))

function symlogformatter(x, n=-12)
    if sign(x) == 0
        "\$0\$"
    else
        s = sign(x)==1 ? "+" : "-"
        nexp = sign(x)*(abs(x) + n)
        if sign(x) == -1
            nexp = -nexp
        end
        "\$$(s)10^{$(nexp)}\$"
    end
end

function plot_mpcc_multipliers(
    name::AbstractString,
    prob::MPCCModel,
    iters_fname::AbstractString;
    range=:,
    tau=0.5,
    filt=true,
)
    iters, nu1, nu2 =
        calculate_mpcc_multiplier_estimate(iters_fname; range=range, filt=filt)
    mu = [i.mu for i in iters[range] if !i.magic]
    inf_pr = [i.inf_pr for i in iters[range] if !i.magic]
    inf_du = [i.inf_du for i in iters[range] if !i.magic]
    δ1 = hcat([i.delta1 for i in iters[range] if !i.magic]...)'
    δ2 = hcat([i.delta2 for i in iters[range] if !i.magic]...)'

    fmt_tup =
        (markershape=:circle, markersize=1, yformatter=symlogformatter, tickfontsize=4)
    nuticks = -10:2:10
    nu1plt = plot(symlog.(nu1); yticks=nuticks, legend=:none, fmt_tup...)
    plot!(nu1plt, symlog.((mu) .^ tau); linestyle=:dash)
    plot!(nu1plt, -symlog.((mu) .^ tau); linestyle=:dash)
    nu2plt = plot(symlog.(nu2); yticks=nuticks, legend=:none, fmt_tup...)
    plot!(nu2plt, symlog.((mu) .^ tau); linestyle=:dash)
    plot!(nu2plt, -symlog.((mu) .^ tau); linestyle=:dash)
    delta1plt = plot(symlog.(δ1); legend=:none, fmt_tup...)
    delta2plt = plot(symlog.(δ2); legend=:none, fmt_tup...)

    muplt =
        plot(symlog.(mu); label=L"\mu", legend=:bottomleft, legendfontsize=4, fmt_tup...)
    plot!(muplt, symlog.(inf_pr); label=L"||c(x)||")
    plot!(muplt, symlog.(inf_du); label=L"||\nabla\mathcal{L}||")
    mainplt = plot(nu1plt, nu2plt, delta1plt, delta2plt, layout=(2, 2))
    return display(plot(mainplt, muplt, layout=grid(2, 1, heights=[0.8, 0.2])))
end
function diagonal_kkt_entries(
    name::AbstractString,
    prob::MPCCModel,
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
