function CCOpt.solve_lpcc(
    lpcc::CCOpt.LPCCModel,
    solver_opts::CCOpt.GurobiLPCCSolverOptions;
    method=-1,
    kwargs...,
)
    bigm = CCOpt.BigMModel(lpcc, solver_opts.M)
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
    GRBsetintparam(env, GRB_INT_PAR_METHOD, method)
    # use kwargs change to presolve, scaling and crossover mode
    # example: gurobi(QM, presolve=0) (see gurobi doc for other options)
    for (k, v) in kwargs
        if k==:presolve
            GRBsetintparam(env, GRB_INT_PAR_PRESOLVE, v) # 0 = no presolve
        elseif k==:scaling
            GRBsetintparam(env, GRB_INT_PAR_SCALEFLAG, v) # 0 = no scaling
        elseif k==:display
            GRBsetintparam(env, GRB_INT_PAR_OUTPUTFLAG, v) # 0 = no display
        elseif k==:threads
            GRBsetintparam(env, GRB_INT_PAR_THREADS, v)
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
    GRBsetdblattr(model.x, GRB_DBL_ATTR_OBJCON, lpcc.nlp.f0)

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
    if !get_minimize(bigm)
        GRBsetintattr(model, GRB_INT_ATTR_MODELSENSE, GRB_MAXIMIZE);
    end
    GRBoptimize(model.x)

    col_value = Vector{Float64}(undef, get_nvar(bigm))
    GRBgetdblattrarray(model.x, GRB_DBL_ATTR_X, 0, get_nvar(bigm), col_value)
    status_ref = Ref{Cint}()
    GRBgetintattr(model.x, GRB_INT_ATTR_STATUS, status_ref)
    status = status_ref.x
    objective_ref = Ref{Float64}()
    GRBgetdblattr(model.x, GRB_DBL_ATTR_OBJVAL, objective_ref)
    objective = objective_ref.x

    y = Vector{Bool}(undef, get_ncc(lpcc))
    y .= col_value[(get_nvar(lpcc)+1):end] .>= 0.5
    stats = (objective=objective, solution=col_value[1:get_nvar(lpcc)])
    return status == 2, stats, y
end
