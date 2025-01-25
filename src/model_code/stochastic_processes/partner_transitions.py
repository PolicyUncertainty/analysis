def partner_transition(period, education, sex, partner_state, options):
    trans_mat = options["partner_trans_mat"]
    trans_vector = trans_mat[sex, education, period, partner_state]
    return trans_vector
