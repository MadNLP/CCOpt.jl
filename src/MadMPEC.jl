module MadMPEC
using MPCC: @default_cc_counters, AbstractMPCCModel
using MPCC
using NLPModels

import MPCC:
    obj,
    grad,
    grad!,
    objgrad,
    objgrad!,
    objcons,
    objcons!,
    cons,
    cons!,
    jth_con,
    jth_congrad,
    jth_congrad!,
    jth_sparse_congrad,
    jac_structure!,
    jac_structure,
    jac_coord!,
    jac_coord,
    jac,
    jprod,
    jprod!,
    jtprod,
    jtprod!,
    jac_op,
    jac_op!,
    jth_hprod,
    jth_hprod!,
    ghjvprod,
    ghjvprod!,
    hess_structure!,
    hess_structure,
    hess_coord!,
    hess_coord,
    hess,
    hess_op,
    hess_op!,
    hprod,
    hprod!,
    sum_counters,
    reset!,
    increment!
export VerticalMPCC

include("VerticalMPCC.jl")

end # module MadMPEC
