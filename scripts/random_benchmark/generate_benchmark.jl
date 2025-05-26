using ADNLPModels

function generate_mpcc(
    n0::Int,
    n1::Int,
    n_ineq::Int,
    nl_obj::AbstractString,
    s_q::Real,
    s_d::Real,
)
    n_obj = n0 + n1

    if nl_obj == "Quadratic_psd"

    elseif nl_obj == "Quadratic_ind"

    elseif nl_obj == "Fletcher"
        f_nl(x) = begin
            out = 0
            for i in 1:(n_obj-1)
                out += 100.0*(x[i+1]-x[i]+1-x[i]^2)^2
            end
            return out
        end
    elseif nl_obj == "Himmelblau"
    elseif nl_obj == "McCormick"
    elseif nl_obj == "Powell"
    elseif nl_obj == "Trigonometric"
    elseif nl_obj == "Rosenbrock"
    elseif nl_obj == "Raydan1"
    elseif nl_obj == "Raydan2"
    elseif nl_obj == "Diagonal3"
    elseif nl_obj == "Diagonal4"
    elseif nl_obj == "Diagonal5"
    elseif nl_obj == "Extended_Triagonal"
    elseif nl_obj == "Three_exponential_terms"
    elseif nl_obj == "Generalized_PSC1"
    elseif nl_obj == "Extended_PSC1"
    elseif nl_obj == "Fletcvb3"
    elseif nl_obj == "Bdqrtic"
    elseif nl_obj == "Tridia"
    elseif nl_obj == "EG2"
    elseif nl_obj == "Edensch"
    elseif nl_obj == "Indef"
    elseif nl_obj == "Cube"
    elseif nl_obj == "Bdexp"
    elseif nl_obj == "Genhumps"
    elseif nl_obj == "Arwhead"
    elseif nl_obj == "Quartic"
    elseif nl_obj == "Cosine"
    elseif nl_obj == "Sine"
    end

    return nlp_vc = ADNLPModels.ADNLPModel(f_nl, ones(2*n0), zeros(2*n0), Inf*ones(2*n0))
end
