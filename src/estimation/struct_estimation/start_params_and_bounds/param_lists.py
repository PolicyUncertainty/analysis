wealth_params = ["mu_low", "mu_high", "bequest_scale", "kappa"]

# Men - Low Education
men_disutil_params_low = [
    "disutil_ft_work_low_good_men",
    "disutil_ft_work_low_bad_men",
    "disutil_unemployed_low_good_men",
    "disutil_unemployed_low_bad_men",
    "disutil_partner_retired_low_men",
]

# Men - High Education
men_disutil_params_high = [
    "disutil_ft_work_high_good_men",
    "disutil_ft_work_high_bad_men",
    "disutil_unemployed_high_good_men",
    "disutil_unemployed_high_bad_men",
    "disutil_partner_retired_high_men",
]

men_SRA_firing = [
    "SRA_firing_logit_intercept_men_low",
    "SRA_firing_logit_intercept_men_high",
]

# Men - Combine low/high params
low_men_disutil_firing = men_disutil_params_low + ["SRA_firing_logit_intercept_men_low"]
high_men_disutil_firing = men_disutil_params_high + [
    "SRA_firing_logit_intercept_men_high"
]


men_taste = [
    "taste_shock_scale_men",
]

men_job_finding_params = [
    "job_finding_logit_const_men",
    "job_finding_logit_high_educ_men",
    "job_finding_logit_good_health_men",
    "job_finding_logit_age_men",
    "job_finding_logit_age_above_55_men",
]

men_disability_params = [
    "disability_logit_const_men",
    "disability_logit_high_educ_men",
    "disability_logit_age_men",
    "disability_logit_age_above_55_men",
]

# Women - Low Education
women_disutil_params_low = [
    "disutil_ft_work_low_good_women",
    "disutil_ft_work_low_bad_women",
    "disutil_pt_work_low_good_women",
    "disutil_pt_work_low_bad_women",
    "disutil_unemployed_low_good_women",
    "disutil_unemployed_low_bad_women",
    "disutil_partner_retired_low_women",
    "disutil_children_ft_work_low",
]

# Women - High Education
women_disutil_params_high = [
    "disutil_ft_work_high_good_women",
    "disutil_ft_work_high_bad_women",
    "disutil_pt_work_high_good_women",
    "disutil_pt_work_high_bad_women",
    "disutil_unemployed_high_good_women",
    "disutil_unemployed_high_bad_women",
    "disutil_partner_retired_high_women",
    "disutil_children_ft_work_high",
]

low_women_disutil_firing = women_disutil_params_low + [
    "SRA_firing_logit_intercept_women_low"
]
high_women_disutil_firing = women_disutil_params_high + [
    "SRA_firing_logit_intercept_women_high"
]

women_SRA_firing = [
    "SRA_firing_logit_intercept_women_low",
    "SRA_firing_logit_intercept_women_high",
]

women_taste = [
    "taste_shock_scale_women",
]

women_job_offer_params = [
    "job_finding_logit_const_women",
    "job_finding_logit_high_educ_women",
    "job_finding_logit_good_health_women",
    "job_finding_logit_age_women",
    "job_finding_logit_age_above_55_women",
]

women_disability_params = [
    "disability_logit_const_women",
    "disability_logit_age_women",
    "disability_logit_age_above_55_women",
    "disability_logit_high_educ_women",
]
