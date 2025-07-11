function project_scholtes_bisect(
    xk::T,
    yk::T,
    𝜅::T,
    𝜎::T;
    step_tol::T=1e-8,
    abs_tol::T=1e-8,
    rel_tol::T=1e-8,
) where {T <: Real}
    @assert xk*yk > 𝜅*𝜎 # TODO (and what if this isn't true?)

    (l, r) = (0, 10*xk)
    m_last = typemax(T)
    stop = false
    𝜏 = 𝜅*𝜎
    while !stop
        m = (l+r)/2
        fm = m^4 - xk*m^3 + yk*𝜏*m - 𝜏^2
        println(m, " ", fm)
        if fm < 0.0
            l = m
        else
            r = m
        end

        if abs(m_last - m) <= step_tol ||
           abs((m_last - m)/m) <= rel_tol ||
           abs(fm) <= abs_tol
            stop = true
        end
        m_last = m
    end
    return m_last, 𝜏/m_last
end

function project_scholtes_explicit!(
    x_target::AbstractVector{T},
    y_target::AbstractVector{T},
    xk::AbstractVector{T},
    yk::AbstractVector{T},
    xl::AbstractVector{T},
    yl::AbstractVector{T},
    𝜅::T,
    𝜎::T;
    heuristic=:min_f,
) where {T <: Real}
    @assert length(x_target) ==
            length(y_target) ==
            length(xk) ==
            length(yk) ==
            length(xl) ==
            length(yl)
    for ii in 1:length(x_target)
        # Hack for very small cases:
        if xk[ii]-xl[ii] ≤ 1e-6
            y_target[ii] = yk[ii]
            x_target[ii] = (𝜅*𝜎)/(yk[ii]-yl[ii])
            continue
        elseif yk[ii]-yl[ii] ≤ 1e-6
            x_target[ii] = xk[ii]
            y_target[ii] = (𝜅*𝜎)/(xk[ii]-xl[ii])
            continue
        end
        if xk[ii]-xl[ii] < yk[ii]-yl[ii]
            (x_target[ii], y_target[ii]) = project_scholtes_explicit(
                xk[ii],
                yk[ii],
                xl[ii],
                yl[ii],
                𝜅,
                𝜎;
                heuristic=heuristic,
            )
        else
            (y_target[ii], x_target[ii]) = project_scholtes_explicit(
                yk[ii],
                xk[ii],
                yl[ii],
                xl[ii],
                𝜅,
                𝜎;
                heuristic=heuristic,
            )
        end
        
    end
end

function project_scholtes_explicit(
    xk::T,
    yk::T,
    xl::T,
    yl::T,
    𝜅::T,
    𝜎::T;
    heuristic=:min_f,
) where {T <: Real}
    # TODO(@anton) do some simplifications to stop duplicate evals (and check if compiler is smart enough to do it itself):
    #              candidates:
    #              - sqrt(c)
    #              - common subexpressions in g h
    # TODO(@anton) check if complex early?
    # TODO(@anton) type instability in some of the expressions?

    # Assert reasonable inputs:
    @assert xk > xl
    @assert yk > yl
    @assert 1 ≥ 𝜅 > 0
    @assert 𝜎 > 0
    @assert heuristic ∈ [:min_diff, :min_f, :max_cos_grad]

    # Calculate actual relaxation parameter
    𝜏 = 𝜅*𝜎
    # get shifted xk
    xk = xk - xl
    yk = yk - yl

    # Intermediate calculations which come from the closed form solution to the polynomial:
    # x^4 - xk*x^3 + yk*𝜏*x - 𝜏^2 = 0
    #! format: off
    a::Complex{T} = 256*𝜏^3 - 192*𝜏^2*xk*yk + 27*𝜏*xk^4 - 6*𝜏*xk^2*yk^2 + 27*𝜏*yk^4 - 4*xk^3*yk^3
    b::Complex{T} = (3*𝜏*yk)^2 - (3*𝜏*xk)^2 + sqrt(3)*𝜏^(3/2)*sqrt(a) #7
    c::Complex{T} = 4*2^(1/3)*3^(2/3)*b^(2/3) - 96*𝜏^2 + 24*𝜏*xk*yk + 3*2^(2/3)*3^(1/3)*xk^2*b^(1/3) #6
    d::Complex{T} = 2^(5/6)*3^(2/3)*b^(2/3)*sqrt(c) #5
    e::Complex{T} = 3*2^(1/6)*3^(1/3)*xk^2*b^(1/3)*sqrt(c) #4
    f::Complex{T} = (2^(2/3)*3^(1/3)*sqrt(c))/(24*b^(1/6)) #3
    g::Complex{T} = 24*sqrt(2)*𝜏^2*sqrt(c) + 9*sqrt(2)*xk^3*sqrt(b) - 72*sqrt(2)*𝜏*yk*sqrt(b) - d - 6*sqrt(2)*𝜏*xk*yk*sqrt(c) + e
    h::Complex{T} = 24*sqrt(2)*𝜏^2*sqrt(c) - 9*sqrt(2)*xk^3*sqrt(b) + 72*sqrt(2)*𝜏*yk*sqrt(b) - d - 6*sqrt(2)*𝜏*xk*yk*sqrt(c) + e
    i::Complex{T} = (2^(5/12)*3^(1/3)*sqrt(g))/(12*b^(1/6)*c^(1/4)) #2
    j::Complex{T} = (2^(5/12)*3^(1/3)*sqrt(h))/(12*b^(1/6)*c^(1/4)) #1
    #! format: on
    # Get roots TODO(@anton) can we get which ones are real early? does that help?
    roots::Vector{Complex{T}} = [xk/4 - f - j, xk/4 - f + j, xk/4 + f - i, xk/4 + f + i]
    # TODO(@anton) what is an appropriate tolerance for abs(imag(ri))?
    real_roots::Vector{T} =
        [real(ri) for ri in roots if abs(imag(ri)) ≤ 1e-10 && real(ri) > 0] # TODO(@anton) maybe needs approx?
    fnorm(x) = 0.5*(x-xk)^2 + 0.5*((𝜏/x)-yk)^2
    fcos(x) = ((𝜏/x)*(xk-x) + (x)*(yk-𝜏/x))/(norm((xk, yk))*norm((x, 𝜏/x)))
    if isempty(real_roots)
        @warn "something went wrong in scholtes projection, we have no positive real roots"
        println(xk)
        println(yk)
        println(xk*yk)
        println(𝜏)
        println("a=$(a)")
        println("b=$(b)")
        println(
            "$((3*𝜏*yk)^2 - (3*𝜏*xk)^2 + sqrt(3)*𝜏^(3/2)*sqrt(a)) = $((3*𝜏*yk)^2) - $((3*𝜏*xk)^2) + $(sqrt(3)*𝜏^(3/2)*sqrt(a))",
        )
        println("c=$(c)")
        println("d=$(d)")
        println("e=$(e)")
        println("f=$(f)")
        println("g=$(g)")
        println("h=$(h)")
        println("i=$(i)")
        println("j=$(j)")
        println(roots)
    elseif length(real_roots) == 1
        return real_roots[1]+xl, 𝜏/real_roots[1]+yl
    else # We have multiple roots (check distances and then min norm(x-y), this is a heuristic)
        if heuristic == :min_f
            minnorm = minimum(fnorm, real_roots)
            candidates = [r for r in real_roots if fnorm(r) ≈ minnorm]
            if length(candidates) == 1
                return candidates[1]+xl, 𝜏/candidates[1]+yl
            else
                x = argmax(fcos, real_roots)
                return x+xl, 𝜏/x+yl
            end
        elseif heuristic == :min_diff
            x = argmin((x) -> abs(x - 𝜏/x), real_roots)
            return x+xl, 𝜏/x+yl
        elseif heuristic == :max_cos_grad
            x = argmax(fcos, real_roots)
            return x+xl, 𝜏/x+yl
        else
            error("undefined heuristic")
        end
    end
end
