function CasADiMPCCModel(libpath::String, datapath::String)
    nlp = CasADiNLPModel(libpath, datapath)
    # Read data (for now assume that we are reading from json:
    # TODO(@anton) yell at people to update to JSONv1 :)
    #data = JSON.parsefile(datapath, CasADiNLPData; allownan=true)
    data = JSON.parsefile(datapath; allownan=true)
    mpcc = MPCCModelConCon(nlp, Vector{Int}(data["ind_cc1"]), Vector{Int}(data["ind_cc2"]))
    return vertical_form(mpcc)
end
