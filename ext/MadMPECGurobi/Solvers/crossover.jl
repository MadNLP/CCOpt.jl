
function MadMPEC.solve_lpcc(
    lpcc::MadMPEC.LPCCModel,
    solver_opts::MadMPEC.GurobiOptions;
    method=-1,
    kwargs...,
)
    bigm = MadMPEC.BigMModel(lpcc, solver_opts.M)
    A = permutedims(jac(bigm, get_x0(bigm))) # @anton This is slow but eh.
    int = Vector{Cchar}(undef, get_nvar(bigm))
    int[1:get_nvar(lpcc)] .= GRB_CONTINUOUS
    int[(get_nvar(lpcc)+1):end] .= GRB_BINARY

    # Blatantly taken from QuadraticModelsGurobi, but the QuadraticModels interface
    # is quite poor imo, so we don't use it for BigM
    env = Gurobi.Env()
    # -1=automatic, 0=primal simplex, 1=dual simplex, 2=barrier,
    # 3=concurrent, 4=deterministic concurrent, 5=deterministic concurrent simplex.
    # default to barrier
    GRBsetintparam(env, "Method", method)
    # use kwargs change to presolve, scaling and crossover mode
    # example: gurobi(QM, presolve=0) (see gurobi doc for other options)
    for (k, v) in kwargs
        if k==:presolve
            GRBsetintparam(env, "Presolve", v) # 0 = no presolve
        elseif k==:scaling
            GRBsetintparam(env, "ScaleFlag", v) # 0 = no scaling
        elseif k==:crossover
            GRBsetintparam(env, "Crossover", v) # 0 = no crossover
        elseif k==:display
            GRBsetintparam(env, "OutputFlag", v) # 0 = no display
        elseif k==:threads
            GRBsetintparam(env, "Threads", v)
        end
    end

    model = Ref{Ptr{Cvoid}}()
    GRBnewmodel(
        env,
        model,
        "",
        get_nvar(bigm),
        grad(bigm, get_x0(bigm)),
        get_lvar(bigm),
        get_uvar(bigm),
        int,
        C_NULL,
    )
    GRBsetdblattr(model.x, "ObjCon", lpcc.nlp.f0)

    GRBaddrangeconstrs(
        model.x,
        get_ncon(bigm),
        get_nnzj(bigm),
        convert(Array{Cint, 1}, A.colptr .- 1),
        convert(Array{Cint, 1}, A.rowval .- 1),
        A.nzval,
        get_lcon(bigm),
        get_ucon(bigm),
        C_NULL,
    )

    GRBoptimize(model.x)

    col_value = Vector{Float64}(undef, get_nvar(bigm))
    GRBgetdblattrarray(model.x, "X", 0, get_nvar(bigm), col_value)
    status_ref = Ref{Cint}()
    GRBgetintattr(model.x, "Status", status_ref)
    status = status_ref.x
    objective_ref = Ref{Float64}()
    GRBgetdblattr(model.x, "ObjVal", objective_ref)
    objective = objective_ref.x

    y = Vector{Bool}(undef, get_ncc(lpcc))
    y .= col_value[(get_nvar(lpcc)+1):end] .>= 0.5
    stats = (objective=objective, solution=col_value[1:get_nvar(lpcc)])
    return status == 2, stats, y
end
