wealth_params = ["mu_low", "mu_high", "bequest_scale", "kappa"]

# Men - Health specific disutility
men_disutil_params = [
    "disutil_ft_work_good_men",
    "disutil_ft_work_bad_men",
    "disutil_unemployed_good_men",
    "disutil_unemployed_bad_men",
    "disutil_partner_retired_men",
]

men_SRA_firing = [
    "SRA_firing_logit_intercept_men_low",
    "SRA_firing_logit_intercept_men_high",
]
men_disutil_firing = men_disutil_params + men_SRA_firing

men_taste = [
    "taste_shock_scale_men",
]

men_job_offer_old_age_params = [
    "job_finding_logit_const_men",
    "job_finding_logit_high_educ_men",
    "job_finding_logit_good_health_men",
    "job_finding_logit_age_above_55_men",
]

men_job_offer_params = men_job_offer_old_age_params + ["job_finding_logit_age_men"]

men_disability_params = [
    "disability_logit_const_men",
    "disability_logit_high_educ_men",
    "disability_logit_age_men",
    "disability_logit_age_above_55_men",
]

# Women - Health specific disutility
women_disutil_params = [
    "disutil_ft_work_good_women",
    "disutil_ft_work_bad_women",
    "disutil_pt_work_good_women",
    "disutil_pt_work_bad_women",
    "disutil_unemployed_good_women",
    "disutil_unemployed_bad_women",
    "disutil_partner_retired_women",
    # "disutil_children_ft_work_high",
    # "disutil_children_ft_work_low",
]

women_SRA_firing = [
    "SRA_firing_logit_intercept_women_low",
    "SRA_firing_logit_intercept_women_high",
]

women_disutil_firing = women_disutil_params + women_SRA_firing


women_taste = [
    "taste_shock_scale_women",
]

women_job_offer_old_age_params = [
    "job_finding_logit_const_women",
    "job_finding_logit_high_educ_women",
    "job_finding_logit_good_health_women",
    "job_finding_logit_age_above_55_women",
]
women_job_offer_params = women_job_offer_old_age_params + [
    "job_finding_logit_age_women"
]

women_disability_params = [
    "disability_logit_const_women",
    "disability_logit_age_women",
    "disability_logit_age_above_55_women",
    "disability_logit_high_educ_women",
]
