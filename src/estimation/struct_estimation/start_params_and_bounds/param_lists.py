wealth_params = [
    "mu_low",
    "mu_high",
    "bequest_scale",
    "kappa_high_men",
    "kappa_high_women",
    "kappa_low_men",
    "kappa_low_women",
]

# Men - Health specific disutility
men_disutil_params = [
    "disutil_ft_work_good_men",
    "disutil_ft_work_bad_men",
    # "disutil_ft_work_disabled_men",  # NEW
    "disutil_unemployed_good_men",
    "disutil_unemployed_bad_men",
    # "disutil_unemployed_disabled_men",  # NEW
    "disutil_partner_retired_men",
    # "disutil_unemployed_above_58_men",
]

# men_SRA_firing = [
#     "SRA_firing_logit_intercept_men_low",
#     "SRA_firing_logit_intercept_men_high",
# ]

# men_disutil_firing_edu = men_disutil_params_edu + men_SRA_firing

men_taste = [
    "taste_shock_scale_men",
]

men_job_offer_params = [
    "job_finding_logit_const_men",
    "job_finding_logit_high_educ_men",
    # "job_finding_logit_good_health_men",
    "job_finding_logit_above_50_men",
    "job_finding_logit_above_55_men",
    "job_finding_logit_above_60_men",
    # "job_finding_logit_age_above_55_men",
    # "job_finding_logit_age_men",
]

men_disability_params = [
    "disability_logit_const_men",
    # "disability_logit_age_men",
    # "disability_logit_age_above_50_men",
    "disability_logit_above_50_men",
    "disability_logit_above_55_men",
    "disability_logit_above_60_men",
    # "disability_logit_high_educ_men",
    # "disability_logit_low_educ_men",
]

# Women - Health specific disutility
women_disutil_params = [
    "disutil_ft_work_good_women",
    "disutil_ft_work_bad_women",
    # "disutil_ft_work_disabled_women",  # NEW
    "disutil_pt_work_good_women",
    "disutil_pt_work_bad_women",
    # "disutil_pt_work_disabled_women",  # NEW
    "disutil_unemployed_good_women",
    "disutil_unemployed_bad_women",
    # "disutil_unemployed_disabled_women",  # NEW
    "disutil_partner_retired_women",
    "disutil_children_ft_work_high",
    "disutil_children_ft_work_low",
    # "disutil_unemployed_above_58_good_women",
    # "disutil_unemployed_above_58_bad_women",
    "disutil_children_pt_work_high",
    "disutil_children_pt_work_low",
]

# women_SRA_firing = [
#     "SRA_firing_logit_intercept_women_low",
#     "SRA_firing_logit_intercept_women_high",
# ]

# women_disutil_firing = women_disutil_params + women_SRA_firing


women_taste = [
    "taste_shock_scale_women",
]

women_job_offer_params = [
    "job_finding_logit_const_women",
    "job_finding_logit_high_educ_women",
    # "job_finding_logit_good_health_women",
    "job_finding_logit_above_50_women",
    "job_finding_logit_above_55_women",
    "job_finding_logit_above_60_women",
    # "job_finding_logit_age_above_55_women",
    # "job_finding_logit_age_women",
]

women_disability_params = [
    "disability_logit_const_women",
    # "disability_logit_age_women",
    # "disability_logit_age_above_50_women",
    "disability_logit_above_50_women",
    "disability_logit_above_55_women",
    "disability_logit_above_60_women",
    # "disability_logit_high_educ_women",
    # "disability_logit_low_educ_women",
]
