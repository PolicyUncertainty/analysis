def partner_transition(period, education, partner_state, options):
    male_trans_mat = options["partner_trans_mat"]
    trans_vector = male_trans_mat[education, period, partner_state]
    return trans_vector
