function MadMPEC.solve_lpcc(
    lpcc::MadMPEC.LPCCModel,
    solver_opts::MadMPEC.HiGHSOptions;
    kwargs...,
)
    bigm = MadMPEC.BigMModel(lpcc, solver_opts.M)
    A = jac(bigm, get_x0(bigm))
    int = Vector{HiGHS.HighsInt}(undef, get_nvar(bigm))
    int[1:get_nvar(lpcc)] .= kHighsVarTypeContinuous
    int[(get_nvar(lpcc)+1):end] .= kHighsVarTypeInteger

    col_value = Vector{Float64}(undef, get_nvar(bigm))
    row_value = Vector{Float64}(undef, get_ncon(bigm))
    model_status = Vector{HiGHS.HighsInt}(undef, 1)
    highs = Highs_create()
    colptr = Vector{HiGHS.HighsInt}(undef, length(A.colptr))
    rowval = Vector{HiGHS.HighsInt}(undef, length(A.rowval))

    colptr .= A.colptr .- HiGHS.HighsInt(1)
    rowval .= A.rowval .- HiGHS.HighsInt(1)
    status = Highs_passMip(
        highs,
        get_nvar(bigm),#num_col
        get_ncon(bigm),#num_row
        get_nnzj(bigm),# num_nz
        kHighsMatrixFormatColwise,# a_format
        kHighsObjSenseMinimize,# sense TODO(also maximize)
        lpcc.nlp.f0,# offset
        grad(bigm, get_x0(bigm)),# col_cost
        get_lvar(bigm),# col_lower
        get_uvar(bigm),# col_upper
        get_lcon(bigm),# row_lower
        get_ucon(bigm),# row_upper
        colptr,# a_start
        rowval,# a_index
        A.nzval,# a_value
        int,# integrality
    )
    status = Highs_setBoolOptionValue(highs, "output_flag", true)
    status = Highs_run(highs)
    status = Highs_getSolution(highs, col_value, C_NULL, row_value, C_NULL)
    y = Vector{Bool}(undef, get_ncc(lpcc))
    y .= col_value[(get_nvar(lpcc)+1):end] .>= 0.5
    stats = (objective=Highs_getObjectiveValue(highs), solution=col_value[1:get_nvar(lpcc)])
    return Highs_getModelStatus(highs) == kHighsModelStatusOptimal, stats, y
end
