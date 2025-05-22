
using JuMP
using NLPModelsJuMP

include("utils.jl")

"""
    stewart_anitescu_fesd_collocation

Implementation of the finite element method with switch detection (FESD).

Follow closely the linearized switch detection described in [1].

## Reference

[1] Anton Pozharskiy, Armin Nurkanović, Moritz Diehl.
"Finite Elements with Switch Detection for Numerical Optimal Control of Projected Dynamical Systems", 2024.

"""
function stewart_anitescu_linearized_fesd(nh, rk::RKScheme; relax=1e-6, big_M=1000.0)
    if rk.c[end] != 1.0
        error("FESD collocation has not yet been implemented for RKScheme with c[end] < 1.0")
    end
    nf = 2
    nc = length(rk.c)
    tf = 2.0

    # N.B.: I observe that Ipopt is sensitive to the lower and upper bounds
    # we set on the time step `h`.
    hmin, hmax = 0.001 * tf / nh, 1000 * tf / nh

    model = Model()
    # States
    @variable(model, x[1:nh+1, 1:nc+1])
    @variable(model, y[1:nh+1, 1:nc+1])
    # Stewart's multipliers
    @variable(model, 0.0 <= θ[1:nh, 1:nc, 1:nf])
    @variable(model, 0.0 <= λ[1:nh, 0:nc, 1:nf])
    @variable(model, μ[1:nh, 0:nc])
    # Switch detection
    @variable(model, 0.0 <= λs[1:nh, 1:nf])
    @variable(model, 0.0 <= θs[1:nh, 1:nf])
    @variable(model, 0.0 <= πλ[1:nh-1, 1:nf])
    @variable(model, 0.0 <= πθ[1:nh-1, 1:nf])
    @variable(model, 0.0 <= τ1[1:nh-1, 1:nf])
    @variable(model, 0.0 <= τ2[1:nh-1, 1:nf])
    @variable(model, 0.0 <= v[1:nh-1, 1:nf])
    # Integration step
    @variable(model, hmin <= h[1:nh] <= hmax, start=tf/nh)
    @constraint(model, sum(h) == tf)

    # Initial position
    @constraint(model, y[1, 1] == 0.0)
    # Initial position for multipliers
    @constraint(model, μ[1, 0] == x[1, 1]) # N.B: assume initial position is negative
    @constraint(model, λ[1, 0, 1] ==  x[1, 1] - μ[1, 0])
    @constraint(model, λ[1, 0, 2] == -x[1, 1] - μ[1, 0])
    # Dynamics
    @expressions(model, begin
        dx[t=1:nh, i=1:nc], 3.0 * θ[t, i, 1] + 1.0 * θ[t, i, 2]
        dy[t=1:nh, i=1:nc], x[t, i+1]^2
    end)
    # Collocations
    @constraints(model, begin
        [t=1:nh, i=2:nc+1], x[t, i] == x[t, 1] + h[t] * sum(rk.a[i-1, j] * dx[t, j] for j in 1:nc)
        [t=1:nh, i=2:nc+1], y[t, i] == y[t, 1] + h[t] * sum(rk.a[i-1, j] * dy[t, j] for j in 1:nc)
    end)
    # Continuity
    # w.r.t primal variables
    @constraints(model, begin
        [t=1:nh], x[t+1, 1] == x[t, 1] + h[t] * sum(rk.b[j] * dx[t, j] for j in 1:nc)
        [t=1:nh], y[t+1, 1] == y[t, 1] + h[t] * sum(rk.b[j] * dy[t, j] for j in 1:nc)
    end)
    # w.r.t dual variables
    @constraints(model, begin
        [t=1:nh-1, j=1:nf], λ[t, nc, j] == λ[t+1, 0, j]
        [t=1:nh-1], μ[t, nc] == μ[t+1, 0]
    end)
    # Stewart model
    @constraints(model, begin
        [t=1:nh, i=1:nc], sum(θ[t, i, j] for j in 1:nf) == 1.0
        # g_1(x) = x
        [t=1:nh, i=0:nc], x[t, i+1] - λ[t, i, 1] - μ[t, i] == 0
        # g_2(x) = -x
        [t=1:nh, i=0:nc], -x[t, i+1] - λ[t, i, 2] - μ[t, i] == 0
    end)
    # Switch detection
    @constraints(model, begin
        [t=1:nh, j=1:nf], sum(θ[t, i, j] for i in 1:nc) == θs[t, j]
        [t=1:nh, j=1:nf], sum(λ[t, i, j] for i in 0:nc) == λs[t, j]
        # [t=1:nh, i=1:nc, j=1:nf], θ[t, i, j] * λs[t, j] <= relax
        [t=1:nh, i=1:nc, j=1:nf], [θ[t, i, j], λs[t, j]] ∈ MOI.Complements(2)
    end)
    # Logical constraints
    @constraints(model, begin
        # OR
        [t=1:nh-1, j=1:nf], πλ[t, j] >= λs[t, j]
        [t=1:nh-1, j=1:nf], πλ[t, j] >= λs[t+1, j]
        [t=1:nh-1, j=1:nf], πλ[t, j] <= λs[t, j] + λs[t+1, j]
        # OR
        [t=1:nh-1, j=1:nf], πθ[t, j] >= θs[t, j]
        [t=1:nh-1, j=1:nf], πθ[t, j] >= θs[t+1, j]
        [t=1:nh-1, j=1:nf], πθ[t, j] <= θs[t, j] + θs[t+1, j]
        # AND
        [t=1:nh-1, j=1:nf], v[t, j] <= πλ[t, j]
        [t=1:nh-1, j=1:nf], v[t, j] <= πθ[t, j]
        [t=1:nh-1, j=1:nf], v[t, j] >= πθ[t, j] - τ1[t, j]
        [t=1:nh-1, j=1:nf], πθ[t, j] - πλ[t, j] == τ1[t, j] - τ2[t, j]
        # [t=1:nh-1, j=1:nf], τ1[t, j] * τ2[t, j] <= relax
        [t=1:nh-1, j=1:nf], [τ1[t, j], τ2[t, j]] ∈ MOI.Complements(2)
    end)
    # Step equilibration
    @constraints(model, begin
        [t=1:nh-1], -big_M * sum(v[t, j] for j=1:nf) <= (h[t+1] - h[t])
        [t=1:nh-1], (h[t+1] - h[t]) <= big_M * sum(v[t, j] for j=1:nf)
    end)
    # Objective
    @objective(model, Min, y[nh+1, 1] + (x[nh+1, 1] - 5/3)^2)

    return model
end

# Formulate FESD using JuMP
model = stewart_anitescu_linearized_fesd(50, CrankNicolson(); big_M=1e5)
# Parse the complementarity constraints and reformulate them in vertical form
ind_cc1, ind_cc2 = parse_ccons!(model)
# Get evaluator as NLPModels
nlp = MathOptNLPModel(model)

