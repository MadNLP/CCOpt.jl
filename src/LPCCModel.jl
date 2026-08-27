const LPCCModel = MPCCModel{T,VT,LinearModel{T,VT,MAT}} where {T,VT,MAT}

######################### linearize! #########################
function LPCCModel(mpcc::MPCCModel{T,VT}, x0::VT; tr::T = T(Inf)) where {T,VT}
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
    lp = LinearModel(mpcc.nlp, x0; tr = tr_vec)

    return MPCCModel(lp, get_ind_cc1(mpcc), get_ind_cc2(mpcc))
end
