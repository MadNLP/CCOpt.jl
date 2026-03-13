using CCOpt, CasADiNLPModels, JSON

function CasADiMPCCModel(libpath::String, datapath::String)
    nlp = CasADiNLPModel(libpath, datapath)
    data = JSON.parsefile(datapath; allownan=true)
    mpcc = MPCCModelConCon(nlp, Vector{Int}(data["ind_cc1"]), Vector{Int}(data["ind_cc2"]))
    return vertical_form(mpcc)
end
