using AmplNLReader, MadMPEC

function mpcc_from_ampl(ampl::AmplNLReader.AmplModel)
    ind_ccc2 = findall(ampl.meta.cvar ≠ 0)
    ind_vcc1 = ampl.meta.cvar[ind_ccc2]

    return MadMPEC.MPCCModelVarCon(ampl, ind_vcc1, ind_ccc2)
end
