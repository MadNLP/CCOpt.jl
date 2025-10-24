function build_bnlp_solver!(solver::Mpecopt)
    return solver.cnt.bnlp_init_time += @elapsed begin
        mpcc = solver.mpcc
        # Create BNLP
        bnlp = BranchNLP(mpcc, convert(Vector{Bool}, b))
        @views begin
            bnlp.meta.x0 .= MadNLP.variable(solver.ipm.x) # Warmstart the BranchNLP
            bnlp.meta.x0[mpcc.meta.ind_cc1[.!b]] .=
                mpcc.meta.lvar[mpcc.meta.ind_cc1[.!b]]
            bnlp.meta.x0[mpcc.meta.ind_cc2[b]] .= mpcc.meta.lvar[mpcc.meta.ind_cc2[b]]
        end
        # Solve the BNLP
        solver.bnlp_ipm = MadNLP.MadNLPSolver(bnlp; solver.opts.bnlp_opts...) # TODO(@anton) again options for BNLP should live somewhere
        #### WARNING: THIS IS A HACK
        # Because there is no way to pass a counters object we have to make sure
        # that all the pointers get updated
        # TODO(@anton) this needs to be done in a smarter way
        solver.bnlp_ipm.cnt = solver.ipm.cnt
        # HACK ENDS HERE
    end
end
