using AmplNLReader, MadMPEC

function mpcc_from_ampl(ampl::AmplNLReader.AmplModel)
    # First we find the nonzero elements in the cvar vector:
    # cvar = 0 → normal nonlinear constraint
    # cvar ≠ 0 → complementarity with the cvar-th variable
    ind_ccc2 = findall(!iszero, ampl.meta.cvar)
    # Then we store the corresponding variables
    ind_vcc1 = ampl.meta.cvar[ind_ccc2]
    return MadMPEC.MPCCModelVarCon(ampl, ind_vcc1, ind_ccc2)
end
