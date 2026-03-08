const LPCCModel{T,VT} = MPCCModel{T,VT,LinearModel{T,VT}} where {T,VT}

######################### linearize! #########################
function LPCCModel(mpcc::MPCCModel{T,VT}, x0::VT; tr::T=T(Inf)) where {T,VT}
    if !is_vertical(mpcc)
        # TODO(@anton) Perhaps we should do this automatically in the future or we can support non-vertical form scholtes
        #              though this makes the callbacks a bit more complicated
        error(
            "LPCCModel currently expects a vertical form MPCC, use vertical_form(mpcc) to convert it.",
        )
    end

    tr_vec = similar(x0)
    tr_vec .= tr
    tr_vec[get_ind_cc1(mpcc)] .= Inf
    tr_vec[get_ind_cc2(mpcc)] .= Inf
    lp = LinearModel(mpcc.nlp, x0; tr=tr_vec)

    return MPCCModelVarVar(lp, get_ind_cc1(mpcc), get_ind_cc2(mpcc))
end

######################### linearize! #########################
# update an lpcc with data in place
# TODO(@anton) cleanup allocations here.
function linearize!(
    lpcc::AbstractMPCCModel{T, VT},
    mpcc::AbstractMPCCModel{T, VT},
    x::VT;
    tr=0.0,
) where {T, VT}
    # Reimplement linearize for quadratic model here because
    # we need tighter control of the nlp meta
    QuadraticModels.linearize!(
        lpcc.nlp,
        mpcc.nlp,
        x,
        lpcc.A,
        lpcc.klasttouch,
        lpcc.csrrowptr,
        lpcc.csrcolval,
        lpcc.csrnzval;
        tr=tr,
    )
    lp = lpcc.nlp

    fixed_map = lpcc.fixed_map
    fixed_map .= 0
    n_fixed = 0
    if tr > 0.0
        for ii in 1:mpcc.meta.ncc
            icc1 = mpcc.meta.ind_cc1[ii]
            icc2 = mpcc.meta.ind_cc2[ii]
            # Assumes at least one branch is in the trust region
            if x[icc1] - tr > mpcc.nlp.meta.lvar[icc1]
                n_fixed += 1
                fixed_map[ii] = 2
                # update bounds in the lp
                lp.meta.uvar[icc2] = lp.meta.lvar[icc2]
                lp.meta.x0[icc2] = lp.meta.lvar[icc2]
            elseif x[icc2] - tr > mpcc.nlp.meta.lvar[icc2]
                n_fixed += 1
                fixed_map[ii] = 1
                # update bounds in the lp by fixing the variable
                lp.meta.uvar[icc1] = lp.meta.lvar[icc1]
                lp.meta.x0[icc1] = lp.meta.lvar[icc1]
            end
        end
    end
    # Prune the new ind_cc
    ind_cc1 = mpcc.meta.ind_cc1[fixed_map .== 0]
    ind_cc2 = mpcc.meta.ind_cc2[fixed_map .== 0]
    cc_types = mpcc.meta.cc_types[fixed_map .== 0]
    ind_x = setdiff(1:mpcc.meta.nvar, union(ind_cc1, ind_cc2))

    lpcc.meta.ncc = mpcc.meta.ncc - n_fixed
    lpcc.meta.ind_cc1 = ind_cc1
    lpcc.meta.ind_cc2 = ind_cc2
    lpcc.meta.cc_types = cc_types
    lpcc.meta.ind_x = ind_x
    return lpcc
end

function tr!(
    lpcc::AbstractMPCCModel{T, VT},
    mpcc::AbstractMPCCModel{T, VT},
    x::VT;
    tr=0.0,
) where {T, VT}
    # Reimplement linearize for quadratic model here because
    # we need tighter control of the nlp meta
    lp = lpcc.nlp
    lp.meta.lvar .= max.(mpcc.meta.lvar .- x, -tr)
    lp.meta.uvar .= max.(mpcc.meta.uvar .- x, tr)

    fixed_map = lpcc.fixed_map
    fixed_map .= 0
    n_fixed = 0
    if tr > 0.0
        for ii in 1:mpcc.meta.ncc
            icc1 = mpcc.meta.ind_cc1[ii]
            icc2 = mpcc.meta.ind_cc2[ii]
            # Assumes at least one branch is in the trust region
            if x[icc1] - tr > mpcc.nlp.meta.lvar[icc1]
                n_fixed += 1
                fixed_map[ii] = 2
                # update bounds in the lp
                lp.meta.uvar[icc2] = lp.meta.lvar[icc2]
                lp.meta.x0[icc2] = lp.meta.lvar[icc2]
            elseif x[icc2] - tr > mpcc.nlp.meta.lvar[icc2]
                n_fixed += 1
                fixed_map[ii] = 1
                # update bounds in the lp by fixing the variable
                lp.meta.uvar[icc1] = lp.meta.lvar[icc1]
                lp.meta.x0[icc1] = lp.meta.lvar[icc1]
            end
        end
    end
    # Prune the new ind_cc
    ind_cc1 = mpcc.meta.ind_cc1[fixed_map .== 0]
    ind_cc2 = mpcc.meta.ind_cc2[fixed_map .== 0]
    cc_types = mpcc.meta.cc_types[fixed_map .== 0]
    ind_x = setdiff(1:mpcc.meta.nvar, union(ind_cc1, ind_cc2))

    lpcc.meta.ncc = mpcc.meta.ncc - n_fixed
    lpcc.meta.ind_cc1 = ind_cc1
    lpcc.meta.ind_cc2 = ind_cc2
    lpcc.meta.cc_types = cc_types
    lpcc.meta.ind_x = ind_x

    return lpcc
end
