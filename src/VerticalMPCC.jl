"""
Vertical form MPCC in the form:
min  f(x,y,z)
s.t. l <= x <= u
    lcv <= c(x, y, z) <= ucv
    0 <= y _|_ z >= 0

Where the nonlinear constraints c(x,y,z) take the form
lc <= c(x)     <= 0
0  <= y - G(x) <= 0
0  <= z - H(x) <= 0
"""
mutable struct VerticalMPCC{T, S} <: AbstractMPCCModel{T,S}

    meta::NLPModelMeta{T,S}
    cc_meta::MPCCModelMeta{T,S}
    cc_counters::MPCCCounters
    mod::AbstractMPCCModel{T,S}
end

function VerticalMPCC(mod::AbstractMPCCModel{T,S}) where {T,S}
    ncc = mod.cc_meta.ncc
    # TODO(@anton) implement single variable check.
    # linG = mod.G.meta.lin
    # linH = mod.H.meta.lin
    # nlG = setdiff(collect(1:ncc), linG);
    # nlH = setdiff(collect(1:ncc), linH);
    # nnlG = length(nlG)
    # nnlH = length(nlH)

    nGlift = ncc
    nHlift = ncc
    # Add Lifting Variables
    new_nvar = mod.meta.nvar + nGlift + nHlift
    new_lvar = vcat(mod.meta.lvar, -Inf*ones(nGlift), -Inf*ones(nHlift)) # NOTE: lower bounds in G,H
    new_uvar = vcat(mod.meta.uvar, Inf*ones(nGlift), Inf*ones(nHlift)) # NOTE: lower bounds in G,H
    new_lcon = vcat(mod.meta.lcon, zeros(nGlift + nHlift))
    new_ucon = vcat(mod.meta.ucon, zeros(nGlift + nHlift))
    new_ncon = mod.meta.ncon + nGlift + nHlift;
    new_y0 = vcat(mod.meta.y0, zeros(nGlift + nHlift))
    # TODO(@anton) index to only add correct variables here
    new_x0 = vcat(mod.meta.x0, MPCC.consG(mod, mod.meta.x0), MPCC.consH(mod, mod.meta.x0))

    # Calculate the nonzero counts
    # TODO(@anton) need to only count some of the nnzjG/H if we don't lift everything but
    # we actually cannot do this, and would require changes in MPCC.jl
    new_nnzj = mod.meta.nnzj + mod.cc_meta.nnzjG + mod.cc_meta.nnzjH + nGlift + nHlift
    # TODO(@anton) we can't know for certain which G, H are linear, so assume _all_ are nonlinear
    new_nln_nnzj = mod.meta.nln_nnzj + mod.cc_meta.nnzjG + mod.cc_meta.nnzjH + nGlift + nHlift
    # TODO(@anton) calculating the nonzeros in the hessian correctly also requires changes to MPCCModelMeta
    #              and the same for meta.lin

    cc_meta = MPCC.MPCCModelMeta(
        new_nvar,
        ncc,
        lccG = mod.cc_meta.lccG,
        lccH = mod.cc_meta.lccH,
        yG = mod.cc_meta.yG,
        yH = mod.cc_meta.yH,
        nnzjG = ncc, # Because 1 nonzero per variable
        nnzjH = ncc, # Because 1 nonzero per variable
    )

    meta = MPCC.NLPModelMeta(
        new_nvar;
        x0 = new_x0,
        lvar = new_lvar,
        uvar = new_uvar,
        ncon = new_ncon,
        y0 = new_y0,
        lcon = new_lcon,
        ucon = new_ucon,
        nnzj = new_nln_nnzj,
        lin_nnzj = mod.meta.lin_nnzj,
        nln_nnzj = new_nln_nnzj,
        nnzh = new_nvar * (new_nvar + 1) / 2, # mp.meta.nnzh + G.meta.nnzh + H.meta.nnzh
        lin = mod.meta.lin,
        minimize = mod.meta.minimize,
        islp = mod.meta.islp,
        name = "Vertical form of $(mod.meta.name)",
    )

    return VerticalMPCC(meta, cc_meta, MPCCCounters(), mod)

end
