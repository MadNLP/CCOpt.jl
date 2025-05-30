using JuMP, NLPModelsJuMP, ADNLPModels, MadMPEC
using LinearAlgebra
using Distributions, StatsBase, Random
import SparseArrays: sprand

function sprand(dist::Sampleable, m::Integer, n::Integer, density::AbstractFloat)
    return sprand(m, n, density, n -> rand(dist, n))
end

function sprand(
    rng::AbstractRNG,
    dist::Sampleable,
    m::Integer,
    n::Integer,
    density::AbstractFloat,
)
    return sprand(rng, m, n, density, (rng, n) -> rand(rng, dist, n))
end

const nl_funs = [
    "Fletcher",
    "Himmelblau",
    "McCormick",
    "Powell",
    "Trigonometric",
    "Rosenbrock",
    "Raydan1",
    "Raydan2",
    "Diagonal3",
    "Diagonal4",
    "Diagonal5",
    "Extended_Triagonal",
    "Three_Exponential_Terms",
    "Generalized_PSC1",
    "Extended_PSC1",
    "Fletcvb3",
    "Bdqrtic",
    "Tridia",
    "EG2",
    "Edensch",
    "Indef",
    "Cube",
    "Bdexp",
    "Genhumps",
    "Arwhead",
    "Quartic",
    "Cosine",
    "Sine",
]

function generate_mpcc_ad(
    n0::Int,
    n1::Int,
    n_ineq::Int,
    nl_obj::AbstractString;
    s_dupe=0.5,
    s_cc=0.5,
    s_ineq=0.5,
    s=0.5,
    random_matrix_density=false,
    rng=Random.default_rng(),
)
    n_obj = n0 + n1
    n_xy = n0 + n1
    n_cc = n1

    # Distributions to pull from
    N_x0 = Normal(0, 1)
    N_y0 = Normal(0, 1)
    U_c = Uniform(0, 1)
    U_d = Uniform(1, 3)
    U_A = Uniform(0, 2)
    U_B = Uniform(0, 2)
    U_d1 = Uniform(0, 2)
    U_d2 = Uniform(0, 2)
    U_N = Uniform(-1, 1)
    U_E = Uniform(3, 4)
    U_q = Uniform(-20, 10)
    U_ubw = Uniform(1e1, 1e3)

    # sparsity of matricies
    if random_matrix_density
        s = rand(rng, Uniform(0.1, 0.3))
    end

    # Generate initial guess
    x0 = rand(rng, N_x0, n0)
    y0 = rand(rng, N_y0, n1)

    # generate ubw
    ubw = rand(rng, U_ubw, n_xy)

    if nl_obj == "Quadratic_psd"
        H = sprand(rng, Float64, n_obj, n_obj, s)
        H = H'*H

        f_nl = (w) -> begin
            return 0.5*w[1:n_obj]'*H*w[1:n_obj];
        end
    elseif nl_obj == "Quadratic_ind"
        H = sprand(rng, Float64, n_obj, n_obj, s)

        f_nl = (w) -> begin
            return 0.5*w[1:n_obj]'*H*w[1:n_obj];
        end
    elseif nl_obj == "Fletcher"
        f_nl = (w) -> begin
            out = 0
            for i in 1:(n_obj-1)
                out += 100.0*(w[i+1]-w[i]+1-w[i]^2)^2
            end
            return out
        end
    elseif nl_obj == "Himmelblau"
        f_nl = (w) -> begin
            out = 0
            for i in 1:Int(floor(n_obj/2))
                out += (w[2*i-1]-w[2*i]-11)^2+(w[2*i-1]+w[2*i]^2-7)^2
            end
            return out
        end
    elseif nl_obj == "McCormick"
        f_nl =
            (w) -> begin
                out = 0
                for i in 1:(n_obj-1)
                    out += (-1.5*w[i]+2.5*w[i+1]+1+(w[i]-w[i+1])^2+sin(w[i]+w[i+1]))
                end
                return out
            end
    elseif nl_obj == "Powell"
        f_nl =
            (w) -> begin
                out = 0
                for i in 1:Int(floor(n_obj/4))
                    out +=
                        (w[4*i-3]+10*w[4*i-2])^2 +
                        5*(w[4*i-1]-w[4*i])^2 +
                        (w[4*i-2]-2*w[4*i-1])^4 +
                        10*(w[4*i-3]-w[4*i])^4
                end
                return out
            end
    elseif nl_obj == "Trigonometric"
        f_nl = (w) -> begin
            out = 0
            for i in 1:n_obj
                out += ((n_obj-sum(cos.(w))) + i*(1-cos(w[i]))-sin(w[i]))^2
            end
            return out
        end
    elseif nl_obj == "Rosenbrock"
        f_nl = (w) -> begin
            out = 0
            for i in 1:(n_obj-1)
                out += 100*(w[i+1]-w[i]^2)^2+(1-w[i])^2
            end
            return out
        end
    elseif nl_obj == "Raydan1"
        f_nl = (w) -> begin
            out = 0
            for i in 1:n_obj
                out += i/10*(exp(w[i])-w[i])
            end
            return out
        end
    elseif nl_obj == "Raydan2"
        f_nl = (w) -> begin
            out = 0
            for i in 1:n_obj
                out += exp(w[i])-w[i]
            end
            return out
        end
    elseif nl_obj == "Diagonal3"
        f_nl = (w) -> begin
            out = 0
            for i in 1:n_obj
                out += exp(w[i])-i*sin(w[i])
            end
            return out
        end
    elseif nl_obj == "Diagonal4"
        f_nl = (w) -> begin
            out = 0
            for i in 1:Int(floor(n_obj/2))
                out += 0.5*(w[2*i-1]^2+100*w[2*i]^2)
            end
            return out
        end
    elseif nl_obj == "Diagonal5"
        f_nl = (w) -> begin
            out = 0
            for i in 1:n_obj
                out += log(exp(w[i])-exp(-w[i]))
            end
            return out
        end
    elseif nl_obj == "Extended_Triagonal"
        f_nl = (w) -> begin
            out = 0
            for i in 1:Int(floor(n_obj/2))
                out += (w[2*i-1]+w[2*i]-3)^2+(w[2*i-1]-w[2*i]-3)^4
            end
            return out
        end
    elseif nl_obj == "Three_Exponential_Terms"
        f_nl =
            (w) -> begin
                out = 0
                for i in 1:Int(floor(n_obj/2))
                    out +=
                        exp(w[2*i-1]+3*w[2*i]-0.1)+exp(w[2*i-1]-3*w[2*i]-0.1)+exp(
                            -w[2*i-1]-0.1,
                        )
                end
                return out
            end
    elseif nl_obj == "Generalized_PSC1"
        f_nl = (w) -> begin
            out = 0
            for i in 1:(n_obj-1)
                out += (w[i]^2+w[i+1]^2+w[i]*w[i+1])^2+sin(w[i])^2+cos(w[i])^2
            end
            return out
        end
    elseif nl_obj == "Extended_PSC1"
        f_nl =
            (w) -> begin
                out = 0
                for i in 1:Int(floor(n_obj/2))
                    out +=
                        (w[2*i-1]^2+w[2*i]^2+w[2*i-1]*w[2*i])^2+sin(w[2*i-1])^2+cos(
                            w[2*i],
                        )^2;
                end
                return out
            end
    elseif nl_obj == "Fletcvb3"
        f_nl = (w) -> begin
            out = 0
            p = 1/1e8
            c = 1
            out += 0.5*p*(w[1]^2+w[end]^2)
            for i in 1:(n_obj-1)
                out += 0.5*p*(w[i]-w[i+1])^2
            end
            for i in 1:n_obj
                h = 1/(i+1)
                out -= (((p*(h^2+2))/h^2)*w[i]+(c*p/h^2)*cos(w[i]))
            end
            return out
        end
    elseif nl_obj == "Bdqrtic"
        f_nl =
            (w) -> begin
                out = 0
                for i in 1:(n_obj-4)
                    out +=
                        (-4*w[i]+3)^2+(w[i]^2+2*w[i+1]^2+3*w[i+2]^2+4*w[i+3]^2+5*w[i+4]^2)
                end
                return out
            end
    elseif nl_obj == "Tridia"
        f_nl = (w) -> begin
            out = 0
            for i in 2:n_obj
                out += i*(2*w[i]-w[i-1])^2
            end
            return out
        end
    elseif nl_obj == "EG2"
        f_nl = (w) -> begin
            out = 0
            for i in 1:(n_obj-1)
                out += sin(w[i]+w[i]^2-1)+0.5*sin(w[i]^2)
            end
            return out
        end
    elseif nl_obj == "Edensch"
        f_nl = (w) -> begin
            out = 0
            for i in 1:(n_obj-1)
                out += (w[i]-2)^4+(w[i]*w[i+1]-2*w[i+1])^2+(w[i+1]+1)^2
            end
            return out
        end
    elseif nl_obj == "Indef"
        f_nl = (w) -> begin
            out = 0
            for i in 2:(n_obj-1)
                out += 0.5*cos(2*w[i]-w[end]-w[1])
            end
            return out
        end
    elseif nl_obj == "Cube"
        f_nl = (w) -> begin
            out = 0
            for i in 2:n_obj
                out += 100*(w[i]-w[i-1]^3)^2
            end
            return out
        end
    elseif nl_obj == "Bdexp"
        f_nl = (w) -> begin
            out = 0
            for i in 1:(n_obj-2)
                out += (w[i]-w[i+1])*exp(-w[i+2]*(w[i]+w[i+1]))
            end
            return out
        end
    elseif nl_obj == "Genhumps"
        f_nl = (w) -> begin
            out = 0
            for i in 1:(n_obj-1)
                out += sin(2*w[i])^2*sin(2*w[i+1])^2+0.05*(w[i]^2+w[i+1]^2)
            end
            return out
        end
    elseif nl_obj == "Arwhead"
        f_nl = (w) -> begin
            out = 0
            for i in 1:(n_obj-1)
                out += (-4*w[i]+3)+(w[i]^2+w[end]^2)^2
            end
            return out
        end
    elseif nl_obj == "Quartic"
        f_nl = (w) -> begin
            out = 0
            for i in 1:n_obj
                out += (w[i]-1)^4;
            end
            return out
        end
    elseif nl_obj == "Cosine"
        f_nl = (w) -> begin
            out = 0
            for i in 1:(n_obj-1)
                out += cos(-0.5*w[i+1]+w[i]^2);
            end
            return out
        end
    elseif nl_obj == "Sine"
        f_nl = (w) -> begin
            out = 0
            for i in 1:(n_obj-1)
                out += sin(-0.5*w[i+1]+w[i]^2);
            end
            return out
        end
    end

    c_obj = rand(rng, U_c, n0)
    d_obj = rand(rng, U_d, n1)

    # Build final objective function
    f(w) = f_nl(w) + [c_obj; d_obj]'*w[1:n_xy]

    r = rand(rng, DiscreteUniform(1, n1))
    # Generate matricies
    A = sprand(rng, U_A, n_ineq, n0, s)
    B = sprand(rng, U_B, n_ineq, n1, s)
    E = sprand(rng, U_E, r, n1-r, s)
    d1 = rand(rng, U_d1, r)
    d2 = rand(rng, U_d2, n1-r)
    D1 = diagm(d1)
    D2 = diagm(d2)
    M = vcat(hcat(D1, E), hcat(-E', D2))
    N = sprand(rng, U_N, n1, n0, s)
    q = rand(rng, U_q, n1);
    𝜖 = rand(rng, Normal(0, 1))
    a = A*x0 + B*y0 .- abs(𝜖)
    # Lower bound ineq
    c_ineq(w) = A*w[1:n0] + B*w[(n0+1):(n0+n1)] - a
    n_nln_ineq = Int(round(s_ineq*n_ineq))
    n_dupe = Int(round(s_dupe*n_ineq))
    ind_dupe = sample(rng, 1:n_ineq, n_dupe; replace=false)
    c_ineq_nln(w) = begin
        out = c_ineq(w)
        out[1:n_nln_ineq] =
            out[1:n_nln_ineq] + out[1:n_nln_ineq] .^ 2 + out[1:n_nln_ineq] .^ 4
        return [out; out[ind_dupe]]
    end

    n_nln_cc = Int(round(s_cc*n_cc))
    c_cc(w) = M*w[1:n0] + N*w[(n0+1):(n0+n1)] + q
    c_cc_nln(w) = begin
        c = c_cc(w)
        c[1:n_nln_cc] = c[1:n_nln_cc] + c[1:n_nln_cc] .^ 2 + c[1:n_nln_cc] .^ 4
        return c
    end
    c(w) = [c_cc_nln(w); c_ineq_nln(w)]

    nlp_vc = ADNLPModels.ADNLPModel(
        f,
        [x0; y0],
        [-Inf*ones(n0); zeros(n1)],
        ubw,
        c,
        zeros(n_cc+n_ineq+n_dupe),
        Inf*ones(n_cc+n_ineq+n_dupe),
    )

    mpcc = MadMPEC.MPCCModelVarCon(nlp_vc, collect((n0+1):(n0+n1)), collect(1:n_cc))
    name = "$(nl_obj)_$(n0)_$(n1)_$(n_ineq)"

    return name, MadMPEC.vertical_form(mpcc)
end

function generate_mpcc_jump(
    n0::Int,
    n1::Int,
    n_ineq::Int,
    nl_obj::AbstractString;
    s_dupe=0.5,
    s_cc=0.5,
    s_ineq=0.5,
    s=0.5,
    𝛼=10,
    random_matrix_density=true,
    rng=Random.default_rng(),
)
    n_obj = n0 + n1
    n_xy = n0 + n1
    n_cc = n1

    # Distributions to pull from
    N_x0 = Normal(0, 1)
    N_y0 = Normal(0, 1)
    U_c = Uniform(0, 1)
    U_d = Uniform(1, 3)
    U_A = Uniform(0, 2)
    U_B = Uniform(0, 2)
    U_d1 = Uniform(0, 2)
    U_d2 = Uniform(0, 2)
    U_N = Uniform(-1, 1)
    U_E = Uniform(3, 4)
    U_q = Uniform(-20, 10)
    U_ubw = Uniform(1e1, 1e3)

    # sparsity of matricies
    if random_matrix_density
        s = rand(rng, Uniform(0.1, 0.3))
    end
    println(s)

    # Generate initial guess
    x0 = rand(rng, N_x0, n0)
    y0 = rand(rng, N_y0, n1)

    # JuMP model
    model = JuMP.Model()
    # generate ubw
    ubw = rand(rng, U_ubw, n_xy)
    @variable(model, -Inf <= x[i=1:n0] <= ubw[i])
    @variable(model, 0 <= y[i=1:n1] <= ubw[n0+i])
    w = vcat(x, y)

    obj = 0
    if nl_obj == "Quadratic_psd"
        H = sprand(rng, Float64, n_obj, n_obj, s)
        H = H'*H
        obj = 0.5*w[1:n_obj]'*H*w[1:n_obj];
    elseif nl_obj == "Quadratic_ind"
        H = sprand(rng, Float64, n_obj, n_obj, s)
        obj = 0.5*w[1:n_obj]'*H*w[1:n_obj]
    elseif nl_obj == "Fletcher"
        for i in 1:(n_obj-1)
            obj += 100.0*(w[i+1]-w[i]+1-w[i]^2)^2
        end
    elseif nl_obj == "Himmelblau"
        for i in 1:Int(floor(n_obj/2))
            obj += (w[2*i-1]-w[2*i]-11)^2+(w[2*i-1]+w[2*i]^2-7)^2
        end
    elseif nl_obj == "McCormick"
        for i in 1:(n_obj-1)
            obj += (-1.5*w[i]+2.5*w[i+1]+1+(w[i]-w[i+1])^2+sin(w[i]+w[i+1]))
        end
    elseif nl_obj == "Powell"
        for i in 1:Int(floor(n_obj/4))
            obj +=
                (w[4*i-3]+10*w[4*i-2])^2 +
                5*(w[4*i-1]-w[4*i])^2 +
                (w[4*i-2]-2*w[4*i-1])^4 +
                10*(w[4*i-3]-w[4*i])^4
        end
    elseif nl_obj == "Trigonometric"
        for i in 1:n_obj
            obj += ((n_obj-sum(cos.(w))) + i*(1-cos(w[i]))-sin(w[i]))^2
        end
    elseif nl_obj == "Rosenbrock"
        for i in 1:(n_obj-1)
            obj += 100*(w[i+1]-w[i]^2)^2+(1-w[i])^2
        end
    elseif nl_obj == "Raydan1"
        for i in 1:n_obj
            obj += i/10*(exp(w[i])-w[i])
        end
    elseif nl_obj == "Raydan2"
        for i in 1:n_obj
            obj += exp(w[i])-w[i]
        end
    elseif nl_obj == "Diagonal3"
        for i in 1:n_obj
            obj += exp(w[i])-i*sin(w[i])
        end
    elseif nl_obj == "Diagonal4"
        for i in 1:Int(floor(n_obj/2))
            obj += 0.5*(w[2*i-1]^2+100*w[2*i]^2)
        end
    elseif nl_obj == "Diagonal5"
        for i in 1:n_obj
            obj += log(exp(w[i])-exp(-w[i]))
        end
    elseif nl_obj == "Extended_Triagonal"
        for i in 1:Int(floor(n_obj/2))
            obj += (w[2*i-1]+w[2*i]-3)^2+(w[2*i-1]-w[2*i]-3)^4
        end
    elseif nl_obj == "Three_Exponential_Terms"
        for i in 1:Int(floor(n_obj/2))
            obj += exp(w[2*i-1]+3*w[2*i]-0.1)+exp(w[2*i-1]-3*w[2*i]-0.1)+exp(-w[2*i-1]-0.1)
        end
    elseif nl_obj == "Generalized_PSC1"
        for i in 1:(n_obj-1)
            obj += (w[i]^2+w[i+1]^2+w[i]*w[i+1])^2+sin(w[i])^2+cos(w[i])^2
        end
    elseif nl_obj == "Extended_PSC1"
        for i in 1:Int(floor(n_obj/2))
            obj += (w[2*i-1]^2+w[2*i]^2+w[2*i-1]*w[2*i])^2+sin(w[2*i-1])^2+cos(w[2*i])^2;
        end
    elseif nl_obj == "Fletcvb3"
        p = 1/1e8
        c = 1
        obj += 0.5*p*(w[1]^2+w[end]^2)
        for i in 1:(n_obj-1)
            obj += 0.5*p*(w[i]-w[i+1])^2
        end
        for i in 1:n_obj
            h = 1/(i+1)
            obj -= (((p*(h^2+2))/h^2)*w[i]+(c*p/h^2)*cos(w[i]))
        end
    elseif nl_obj == "Bdqrtic"
        for i in 1:(n_obj-4)
            obj += (-4*w[i]+3)^2+(w[i]^2+2*w[i+1]^2+3*w[i+2]^2+4*w[i+3]^2+5*w[i+4]^2)
        end
    elseif nl_obj == "Tridia"
        for i in 2:n_obj
            obj += i*(2*w[i]-w[i-1])^2
        end
    elseif nl_obj == "EG2"
        for i in 1:(n_obj-1)
            obj += sin(w[i]+w[i]^2-1)+0.5*sin(w[i]^2)
        end
    elseif nl_obj == "Edensch"
        for i in 1:(n_obj-1)
            obj += (w[i]-2)^4+(w[i]*w[i+1]-2*w[i+1])^2+(w[i+1]+1)^2
        end
    elseif nl_obj == "Indef"
        for i in 2:(n_obj-1)
            obj += 0.5*cos(2*w[i]-w[end]-w[1])
        end
    elseif nl_obj == "Cube"
        for i in 2:n_obj
            obj += 100*(w[i]-w[i-1]^3)^2
        end
    elseif nl_obj == "Bdexp"
        for i in 1:(n_obj-2)
            obj += (w[i]-w[i+1])*exp(-w[i+2]*(w[i]+w[i+1]))
        end
    elseif nl_obj == "Genhumps"
        for i in 1:(n_obj-1)
            obj += sin(2*w[i])^2*sin(2*w[i+1])^2+0.05*(w[i]^2+w[i+1]^2)
        end
    elseif nl_obj == "Arwhead"
        for i in 1:(n_obj-1)
            obj += (-4*w[i]+3)+(w[i]^2+w[n_obj]^2)^2
        end
    elseif nl_obj == "Quartic"
        for i in 1:n_obj
            obj += (w[i]-1)^4;
        end
    elseif nl_obj == "Cosine"
        for i in 1:(n_obj-1)
            obj += cos(-0.5*w[i+1]+w[i]^2);
        end
    elseif nl_obj == "Sine"
        for i in 1:(n_obj-1)
            obj += sin(-0.5*w[i+1]+w[i]^2);
        end
    end

    c_obj = rand(rng, U_c, n0)
    d_obj = rand(rng, U_d, n1)

    # Build final objective function
    obj += 𝛼*[c_obj; d_obj]'*w[1:n_xy]
    @objective(model, Min, obj)

    r = rand(rng, DiscreteUniform(1, n1))
    # Generate matricies
    A = sprand(rng, U_A, n_ineq, n0, s)
    B = sprand(rng, U_B, n_ineq, n1, s)
    E = sprand(rng, U_E, r, n1-r, s)
    d1 = rand(rng, U_d1, r)
    d2 = rand(rng, U_d2, n1-r)
    D1 = diagm(d1)
    D2 = diagm(d2)
    M = vcat(hcat(D1, E), hcat(-E', D2))
    N = sprand(rng, U_N, n1, n0, s)
    q = rand(rng, U_q, n1);
    𝜖 = rand(rng, Normal(0, 1))
    a = A*x0 + B*y0 .- abs(𝜖)
    # Lower bound ineq
    n_nln_ineq = Int(round(s_ineq*n_ineq))
    n_dupe = Int(round(s_dupe*n_ineq))
    ind_dupe = sample(rng, 1:n_ineq, n_dupe; replace=false)

    c_ineq::Vector{NonlinearExpr} = A*x + B*y - a
    c_ineq[1:n_nln_ineq] =
        c_ineq[1:n_nln_ineq] + c_ineq[1:n_nln_ineq] .^ 2 + c_ineq[1:n_nln_ineq] .^ 4
    c_ineq = vcat(c_ineq, c_ineq[ind_dupe])

    n_nln_cc = Int(round(s_cc*n_cc))
    c_cc::Vector{NonlinearExpr} = M*x + N*y + q
    c_cc[1:n_nln_cc] = c_cc[1:n_nln_cc] + c_cc[1:n_nln_cc] .^ 2 + c_cc[1:n_nln_cc] .^ 4

    c = vcat(c_cc, c_ineq)
    @constraint(model, c .>= 0)

    # nlp_vc = ADNLPModels.ADNLPModel(
    #     f,
    #     [x0; y0],
    #     [-Inf*ones(n0); zeros(n1)],
    #     ubw,
    #     c,
    #     zeros(n_cc+n_ineq+n_dupe),
    #     Inf*ones(n_cc+n_ineq+n_dupe),
    # )
    nlp_vc = MathOptNLPModel(model)
    mpcc = MadMPEC.MPCCModelVarCon(nlp_vc, collect((n0+1):(n0+n1)), collect(1:n_cc))

    name = "$(nl_obj)_$(n0)_$(n1)_$(n_ineq)"

    return name, MadMPEC.vertical_form(mpcc)
end

function generate_benchmark_ad(n_probs)
    mpccs = Vector{MadMPEC.MPCCModel}()
    names = Vector{String}()
    #rng = Xoshiro(1)
    # Use twister? I guess it doesn't matter
    rng = MersenneTwister(3)
    ns = sample(rng, 10:300, n_probs; replace=false)
    ns_ineq = Vector{Int}()
    for n in ns
        push!(ns_ineq, Int(round(rand(rng, Uniform(0.1*n, 2*n)))))
    end
    for nl_fun in nl_funs
        for (n, n_ineq) in zip(ns, ns_ineq)
            println("n = $(n), n_ineq = $(n_ineq), nl_fun = $(nl_fun)")
            name, mpcc = generate_mpcc_ad(n, n, n_ineq, nl_fun)
            push!(mpccs, mpcc)
            push!(names, name)
        end
    end
    return names, mpccs
end

function generate_benchmark_jump(n_probs)
    mpccs = Vector{MadMPEC.MPCCModel}()
    names = Vector{String}()
    #rng = Xoshiro(1)
    # Use twister? I guess it doesn't matter
    rng = MersenneTwister(3)
    ns = sample(rng, 10:300, n_probs; replace=false)
    ns_ineq = Vector{Int}()
    for n in ns
        push!(ns_ineq, Int(round(rand(rng, Uniform(0.1*n, 2*n)))))
    end
    for nl_fun in nl_funs
        for (n, n_ineq) in zip(ns, ns_ineq)
            println("n = $(n), n_ineq = $(n_ineq), nl_fun = $(nl_fun)")
            name, mpcc = generate_mpcc_jump(n, n, n_ineq, nl_fun)
            push!(mpccs, mpcc)
            push!(names, name)
            GC.gc(true) # Try Hard GC
        end
    end
    return names, mpccs
end

struct RandomMPCCBenchmark
    ns::Vector{Int}
    ns_ineq::Vector{Int}
    nl_funs::Vector{String}
    rng_seed::Int
    states::Vector{Random.DSFMT.DSFMT_state}
    len::Int

    function RandomMPCCBenchmark(n_probs::Int, nl_funs::Vector{String}, rng_seed::Int)
        rng = MersenneTwister(rng_seed)
        ns = sample(rng, 10:300, n_probs; replace=false)
        ns_ineq = Vector{Int}()
        for n in ns
            push!(ns_ineq, Int(round(rand(rng, Uniform(0.1*n, 2*n)))))
        end

        len = length(nl_funs)*n_probs
        states = Vector{Random.DSFMT.DSFMT_state}(undef, len)
        states[1] = copy(rng.state)
        return new(ns, ns_ineq, nl_funs, rng_seed, states, len)
    end
end

function Base.iterate(bench::RandomMPCCBenchmark)
    rng = MersenneTwister(bench.rng_seed, copy(bench.states[1]))
    #println("$(1), $(rand(rng))")
    mpcc = generate_mpcc_jump(
        bench.ns[1],
        bench.ns[1],
        bench.ns_ineq[1],
        bench.nl_funs[1];
        rng=rng,
    )
    bench.states[2] = copy(rng.state)
    #println("$(1) end, $(rand(rng))")
    return mpcc, 2
end

ind2subv(shape, indices) = Tuple.(CartesianIndices(shape)[indices])

function Base.iterate(bench::RandomMPCCBenchmark, state::Int)
    if state <= bench.len
        rng = MersenneTwister(bench.rng_seed, copy(bench.states[state]))
        #println("$(state), $(rand(rng))")
        ind_n, ind_nl_fun = ind2subv((length(bench.ns), length(bench.nl_funs)), state)
        mpcc = generate_mpcc_jump(
            bench.ns[ind_n],
            bench.ns[ind_n],
            bench.ns_ineq[ind_n],
            bench.nl_funs[ind_nl_fun];
            rng=rng,
        )
        if state + 1 <= bench.len
            bench.states[state+1] = copy(rng.state)
        end
        #println("$(state) end, $(rand(rng))")
        return mpcc, state+1
    else
        return nothing
    end
end

Base.length(bench::RandomMPCCBenchmark) = bench.len
