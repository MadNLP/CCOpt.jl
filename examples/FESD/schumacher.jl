
using MadMPEC
using JuMP
using MadNLP
using MadNLPHSL
using NLPModelsJuMP
using NLPModelsIpopt
using Plots

include("utils.jl")

function track(x)
    sig = 0.1
    step1 = 0.5 * (1.0 + tanh((x-pi)/sig))
    step2 = 0.5 * (1.0 + tanh((x-2*pi)/sig))
    return sin(x) * (1 - step1) + (π - x) * step1 * (1 - step2) + (-pi - sin(x)) * step2
end

"""

    An optimal control problem from:
    Optimal control of systems with discontinuous differential equations
    April 2012, Numerische Mathematik 114(4):653-695
    DOI: 10.1007/s00211-009-0262-2
    David E. Stewart Mihai Anitescu

"""
function schumacher_fesd_model(N, nfe, rk::RKScheme; big_M=1e5, step_eq=:lcc)
    nh = N * nfe      # total number of finite elements
    nf = 2            # total number of nonsmooth modes
    nc = length(rk.c) # total number of intermediate integration points

    T_numerics = 5.0
    T_final_max = 5 * π
    T_final_min = 2.0
    T_guess = 3.0
    μN = 4.0
    alpha = 10.0

    hmin, hmax = 0.1/nh, 2.0*T_numerics/nh
    sot_min, sot_max = 0.0, 25.0

    track_width = 0.5

    target = [3*π; -π]
    x_tracks = range(0.0, target[1], nh)
    y_tracks = range(0.0, target[2], nh)

    # TODO
    qx0 = range(0, target[1], nh+1)
    qy0 = range(0, target[2], nh+1)

    model = Model()
    # State
    @variable(model, qx[i=1:(nh+1), j=1:(nc+1)], start=qx0[i])
    @variable(model, qy[i=1:(nh+1), j=1:(nc+1)], start=qy0[i])
    @variable(model, vx[i=1:(nh+1), j=1:(nc+1)], start=target[1]/nh)
    @variable(model, vy[i=1:(nh+1), j=1:(nc+1)], start=target[2]/nh)
    @variable(model, phi[1:(nh+1), j=1:(nc+1)]) # Orientation
    @variable(model, -2.0 <= a[1:N] <= 2.0)
    @variable(model, -2.0 <= s[1:N] <= 2.0)
    # Step in numerical time
    @variable(model, clock[i=1:(nh+1)], start=T_numerics/nh*(i-1))         # physical time
    @variable(model, hmin <= h[1:nfe, 1:N] <= hmax, start=T_numerics/nh)
    @variable(model, sot_min <= sot <= sot_max, start=T_guess)
    # Stewart's multipliers
    @variable(model, 0.0 <= θ[1:nh, 1:nc, 1:nf])
    @variable(model, 0.0 <= λ[1:nh, 0:nc, 1:nf])
    @variable(model, μ[1:nh, 0:nc])
    # Switch detection
    @variable(model, 0.0 <= λs[1:nh, 1:nf])
    @variable(model, 0.0 <= θs[1:nh, 1:nf])
    @variable(model, 0.0 <= πλ[1:(nh-1), 1:nf])
    @variable(model, 0.0 <= πθ[1:(nh-1), 1:nf])
    @variable(model, 0.0 <= τ1[1:(nh-1), 1:nf])
    @variable(model, 0.0 <= τ2[1:(nh-1), 1:nf])
    @variable(model, 0.0 <= v[1:(nh-1), 1:nf])

    @expression(
        model,
        terminal_cost,
        alpha * ((qx[nh+1, 1] - target[1])^2 + (qy[nh+1, 1] - target[2])^2) + clock[nh+1]
    )

    # Initial position
    @constraints(model, begin
        qx0, qx[1, 1] == 0.0
        qy0, qy[1, 1] == 0.0
        vx0, vx[1, 1] == 0.0
        vy0, vy[1, 1] == 0.0
        phi0, phi[1, 1] == 0.0
        t0, clock[1] == 0.0
    end)
    # Stewart position function
    @expression(
        model,
        g[i=1:nh, j=1:(nc+1)],
        -sin(phi[i, j]) * vx[i, j] + cos(phi[i, j]) * vy[i, j]
    )
    # Initial position for multipliers
    # TODO
    # @constraint(model, μ[1, 0] == 0.0) # N.B: assume initial position is negative
    @constraint(model, λ[1, 0, 1] == g[1, 1] - μ[1, 0])
    @constraint(model, λ[1, 0, 2] == -g[1, 1] - μ[1, 0])
    # Dynamics
    @expressions(
        model,
        begin
            dqx[i=1:nh, j=1:nc], sot * vx[i, j]
            dqy[i=1:nh, j=1:nc], sot * vy[i, j]
            dvx[i=1:nh, j=1:nc],
            sot * (
                a[div(i-1, nfe)+1] * cos(phi[i, j]) -
                μN * (θ[i, j, 1] - θ[i, j, 2]) * sin(phi[i, j])
            )
            dvy[i=1:nh, j=1:nc],
            sot * (
                a[div(i-1, nfe)+1] * sin(phi[i, j]) +
                μN * (θ[i, j, 1] - θ[i, j, 2]) * cos(phi[i, j])
            )
            dph[i=1:nh, j=1:nc],
            sot *
            (s[div(i-1, nfe)+1] * (cos(phi[i, j]) * vx[i, j] + sin(phi[i, j]) * vy[i, j]))
        end
    )
    # Collocations
    @constraints(
        model,
        begin
            con_dqx[i=1:nh, j=2:(nc+1)],
            qx[i, j] == qx[i, 1] + h[i] * sum(rk.a[j-1, k] * dqx[i, k] for k in 1:nc)
            con_dqy[i=1:nh, j=2:(nc+1)],
            qy[i, j] == qy[i, 1] + h[i] * sum(rk.a[j-1, k] * dqy[i, k] for k in 1:nc)
            con_dvx[i=1:nh, j=2:(nc+1)],
            vx[i, j] == vx[i, 1] + h[i] * sum(rk.a[j-1, k] * dvx[i, k] for k in 1:nc)
            con_dvy[i=1:nh, j=2:(nc+1)],
            vy[i, j] == vy[i, 1] + h[i] * sum(rk.a[j-1, k] * dvy[i, k] for k in 1:nc)
            con_dph[i=1:nh, j=2:(nc+1)],
            phi[i, j] == phi[i, 1] + h[i] * sum(rk.a[j-1, k] * dph[i, k] for k in 1:nc)
        end
    )
    # Continuity
    # w.r.t primal variables
    @constraints(
        model,
        begin
            [i=1:nh],
            qx[i+1, 1] == qx[i, 1] + h[i] * sum(rk.b[j] * dqx[i, j] for j in 1:nc)
            [i=1:nh],
            qy[i+1, 1] == qy[i, 1] + h[i] * sum(rk.b[j] * dqy[i, j] for j in 1:nc)
            [i=1:nh],
            vx[i+1, 1] == vx[i, 1] + h[i] * sum(rk.b[j] * dvx[i, j] for j in 1:nc)
            [i=1:nh],
            vy[i+1, 1] == vy[i, 1] + h[i] * sum(rk.b[j] * dvy[i, j] for j in 1:nc)
            [i=1:nh],
            phi[i+1, 1] == phi[i, 1] + h[i] * sum(rk.b[j] * dph[i, j] for j in 1:nc)
            [i=1:nh], clock[i+1] == clock[i] + h[i] * sot
        end
    )
    # w.r.t dual variables
    @constraints(model, begin
        [i=1:(nh-1), j=1:nf], λ[i, nc, j] == λ[i+1, 0, j]
        # [i=1:nh-1], μ[i, nc] == μ[i+1, 0]
    end)
    # State constraints
    @constraints(
        model,
        begin
            sc[i=1:nh, j=1:(nc+1)],
            -track_width/2.0 <= qy[i, j] - track(qx[i, j]) <= track_width/2.0
        end
    )
    # Stewart model
    @constraints(model, begin
        [i=1:nh, j=1:nc], sum(θ[i, j, k] for k in 1:nf) == 1.0
        [i=1:nh, j=0:nc], g[i, j+1] - λ[i, j, 1] - μ[i, j] == 0
        [i=1:nh, j=0:nc], -g[i, j+1] - λ[i, j, 2] - μ[i, j] == 0
    end)
    # Switch detection
    @constraints(
        model,
        begin
            [t=1:nh, j=1:nf], sum(θ[t, i, j] for i in 1:nc) == θs[t, j]
            [t=1:nh, j=1:nf], sum(λ[t, i, j] for i in 0:nc) == λs[t, j]
            [t=1:nh, i=1:nc, j=1:nf], [θ[t, i, j], λs[t, j]] ∈ MOI.Complements(2)
        end
    )

    if step_eq == :lcc
        # Logical constraints
        @constraints(
            model,
            begin
                # OR
                [t=1:(nh-1), j=1:nf], πλ[t, j] >= λs[t, j]
                [t=1:(nh-1), j=1:nf], πλ[t, j] >= λs[t+1, j]
                [t=1:(nh-1), j=1:nf], πλ[t, j] <= λs[t, j] + λs[t+1, j]
                # OR
                [t=1:(nh-1), j=1:nf], πθ[t, j] >= θs[t, j]
                [t=1:(nh-1), j=1:nf], πθ[t, j] >= θs[t+1, j]
                [t=1:(nh-1), j=1:nf], πθ[t, j] <= θs[t, j] + θs[t+1, j]
                # AND
                [t=1:(nh-1), j=1:nf], v[t, j] <= πλ[t, j]
                [t=1:(nh-1), j=1:nf], v[t, j] <= πθ[t, j]
                [t=1:(nh-1), j=1:nf], v[t, j] >= πθ[t, j] - τ1[t, j]
                [t=1:(nh-1), j=1:nf], πθ[t, j] - πλ[t, j] == τ1[t, j] - τ2[t, j]
                [t=1:(nh-1), j=1:nf], [τ1[t, j], τ2[t, j]] ∈ MOI.Complements(2)
            end
        )
        # Step equilibration
        @constraints(
            model,
            begin
                [t=1:(nh-1)], -big_M * sum(v[t, j] for j in 1:nf) <= (h[t+1] - h[t])
                [t=1:(nh-1)], (h[t+1] - h[t]) <= big_M * sum(v[t, j] for j in 1:nf)
            end
        )
        @expression(model, step_eq_cost, 0)
    elseif step_eq == :heuristic_mean
        @expression(model, step_eq_cost, sum(1*(h .- (T_numerics/nh)) .^ 2))
    end
    @constraint(model, [j=1:N], sum(h[i, j] for i in 1:nfe) == T_numerics / N)

    @objective(model, Min, terminal_cost + step_eq_cost)
    return model
end

N = 100
nfe = 3
model = schumacher_fesd_model(N, nfe, CrankNicolson(); big_M=1e5, step_eq=:heuristic_mean)

# Parse the complementarity constraints and reformulate them in vertical form
ind_cc1, ind_cc2 = parse_ccons!(model)
# Get evaluator as NLPModels
nlp = MathOptNLPModel(model)

# Build a MPCC problem
mpcc = MadMPEC.MPCCModelVarVar(nlp, ind_cc1, ind_cc2)
solver_type = :madnlpc
# Solve problem with MadMPEC
if solver_type == :homotopy
    opts = MadMPEC.HomotopySolverOptions()
    opts.print_level = MadNLP.INFO
    opts.σ₀ = 1e-4
    opts.comp_tol = 1e-6

    solver = MadMPEC.HomotopySolver(mpcc, NLPModelsIpopt.IpoptSolver, opts)
    stats = MadMPEC.solve!(solver)
elseif solver_type == :madnlpc
    # MadNLPC
    madnlpc_opts = MadMPEC.MadNLPCOptions(;
        print_level=MadNLP.INFO,
        use_mpecopt=true,
        phase_I_oracle=:lpcc,
        eps_proj=1e-3,
    )
    solver = MadMPEC.MadNLPCSolver(
        mpcc;
        solver_opts=madnlpc_opts,
        print_level=MadNLP.INFO,
        barrier=MadNLP.MonotoneUpdate(mu_init=1e-4),
        tol=1e-6,
    )
    #solver = MadMPEC.MadNLPCSolver(mpcc; solver_opts=madnlpc_opts, print_level=MadNLP.INFO, barrier=MadNLP.LOQOUpdate(gamma=0.05), tol=1e-6)
    stats = MadMPEC.solve_homotopy!(solver)
end
# scholtes = MadMPEC.ScholtesRelaxation(mpcc)
# solver = MadNLP.MadNLPSolver(scholtes; print_level=MadNLP.DEBUG, linear_solver=Ma57Solver, max_iter=1000)
# stats = MadMPEC.solve_homotopy!(solver)

plot()
nh = N * nfe
qx = stats.solution[1:((nh+1)*3)][1:(nh+1)]
qy = stats.solution[((nh+1)*3+1):(6*(nh+1))][1:(nh+1)]
plot!(qx, qy, label="Trajectory")
xlabel!("qx")
ylabel!("qy")
xx = 0:0.1:3π
plot!(xx, track.(xx) .+ 0.25, label="Track lb")
plot!(xx, track.(xx) .- 0.25, label="Track ub")
