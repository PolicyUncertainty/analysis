def partner_transition(period, education, sex, partner_state, model_specs):
    trans_mat = model_specs["partner_trans_mat"]
    trans_vector = trans_mat[sex, education, period, partner_state]
    return trans_vector
