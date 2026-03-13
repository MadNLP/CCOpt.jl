function MadNLP.print_iter(solver::ExactPenaltySolver; is_resto=false)
    ipm = solver.ipm
    obj_scale = ipm.cb.obj_scale[]
    mod(ipm.cnt.k, 10)==0 && MadNLP.@info(
        ipm.logger,
        @sprintf(
            "iter    objective    inf_pr   inf_du inf_compl lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls lg(τ)   inf_cc"
        )
    )
    if is_resto
        RR = ipm.RR::RobustRestorer
        inf_du = RR.inf_du_R
        inf_pr = RR.inf_pr_R
        inf_compl = RR.inf_compl_R
        mu = log10(RR.mu_R)
    else
        inf_du = ipm.inf_du
        inf_pr = ipm.inf_pr
        inf_compl = ipm.inf_compl
        mu = log10(ipm.mu)
    end
    MadNLP.@info(
        ipm.logger,
        @sprintf(
            "%4i%s% 10.7e %6.2e %6.2e %7.2e %5.1f %6.2e %s %6.2e %6.2e%s  %i %5.1f  %6.2e",
            ipm.cnt.k,
            is_resto ? "r" : " ",
            ipm.obj_val/obj_scale,
            inf_pr,
            inf_du,
            inf_compl,
            mu,
            ipm.cnt.k == 0 ? 0.0 : norm(MadNLP.primal(ipm.d), Inf),
            ipm.del_w == 0 ? "   - " : @sprintf("%5.1f", log(10, ipm.del_w)),
            ipm.alpha_z,
            ipm.alpha,
            ipm.ftype,
            ipm.cnt.l,
            get_log_penalty(solver.pnlp),
            solver.inf_pr_cc
        )
    )
    return
end

function get_inf_pr_cc(solver::ExactPenaltySolver{T}) where {T}
    return @views(
        mapreduce(
            (a, la, b, lb) -> max(min(a-la, b-lb), la-a, lb-b),
            max,
            MadNLP.variable(solver.ipm.x)[get_ind_cc1(solver.mpcc)],
            get_lvar(solver.mpcc)[get_ind_cc1(solver.mpcc)],
            MadNLP.variable(solver.ipm.x)[get_ind_cc2(solver.mpcc)],
            get_lvar(solver.mpcc)[get_ind_cc2(solver.mpcc)];
            init=zero(T),
        )
    )
end

function get_inf_pr_cc_prod(solver::ExactPenaltySolver{T}) where {T}
    return @views(
        mapreduce(
            (a, la, b, lb) -> max((a-la)*(b-lb), la-a, lb-b),
            max,
            MadNLP.variable(solver.ipm.x)[get_ind_cc1(solver.mpcc)],
            get_lvar(solver.mpcc)[get_ind_cc1(solver.mpcc)],
            MadNLP.variable(solver.ipm.x)[get_ind_cc2(solver.mpcc)],
            get_lvar(solver.mpcc)[get_ind_cc2(solver.mpcc)];
            init=zero(T),
        )
    )
end

function get_inf_pr_cc_sum(solver::ExactPenaltySolver{T}) where {T}
    return @views(
        mapreduce(
            (a, la, b, lb) -> max((a-la)*(b-lb), la-a, lb-b),
            +,
            MadNLP.variable(solver.ipm.x)[get_ind_cc1(solver.mpcc)],
            get_lvar(solver.mpcc)[get_ind_cc1(solver.mpcc)],
            MadNLP.variable(solver.ipm.x)[get_ind_cc2(solver.mpcc)],
            get_lvar(solver.mpcc)[get_ind_cc2(solver.mpcc)];
            init=zero(T),
        )
    )
end
