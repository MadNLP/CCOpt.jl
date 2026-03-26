# TODO(@anton) fix this to be nonquadratic I guess
# FIXME(@anton) This is broken for fixed complementarities. For now, ignore that.
function _adjust_cc_inds!(cb, ind_cc1, ind_cc2)
    fixed = cb.ind_fixed
    for ii in 1:length(ind_cc1)
        n_less1 = count(<(ind_cc1[ii]), fixed)
        n_less2 = count(<(ind_cc2[ii]), fixed)
        ind_cc1[ii] -= n_less1
        ind_cc2[ii] -= n_less2
    end
end
