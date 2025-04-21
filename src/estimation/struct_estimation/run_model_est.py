# Set paths of project
import pickle as pkl

from set_paths import create_path_dict

paths_dict = create_path_dict(define_user=False)

from estimation.struct_estimation.scripts.estimate_setup import estimate_model

params_to_estimate_names = [
    "mu_men",
    "mu_women",
    # Men Full-time - 4 parameters
    "disutil_ft_work_high_good_men",
    "disutil_ft_work_high_bad_men",
    "disutil_ft_work_low_good_men",
    "disutil_ft_work_low_bad_men",
    # # Women Full-time - 4 parameters
    "disutil_ft_work_high_good_women",
    "disutil_ft_work_high_bad_women",
    "disutil_ft_work_low_good_women",
    "disutil_ft_work_low_bad_women",
    # # Women Part-time - 4 parameters
    "disutil_pt_work_high_good_women",
    "disutil_pt_work_high_bad_women",
    "disutil_pt_work_low_good_women",
    "disutil_pt_work_low_bad_women",
    # Men unemployment - 2 parameters
    "disutil_unemployed_high_men",
    "disutil_unemployed_low_men",
    # Women Unemployment - 2 parameters
    "disutil_unemployed_high_women",
    "disutil_unemployed_low_women",
    # Children - 2 parameters
    "disutil_children_ft_work_low",
    "disutil_children_ft_work_high",
    "lambda",
    # Men job finding - 3 parameters
    "job_finding_logit_const_men",
    "job_finding_logit_period_men",
    "job_finding_logit_high_educ_men",
    # Women job finding - 3 parameters
    "job_finding_logit_const_women",
    "job_finding_logit_period_women",
    "job_finding_logit_high_educ_women",
    # Disability probability
    "disability_logit_const",
    "disability_logit_period",
    "disability_logit_high_educ",
]
model_name = "disability"
LOAD_LAST_ESTIMATE = True
LOAD_SOL_MODEL = True
SAVE_RESULTS = True
USE_WEIGHTS = False

if LOAD_LAST_ESTIMATE:
    # last_estimate = pkl.load(
    #     open(paths_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
    # )
    last_estimate = {
        "lambda": 0.938775655255427,
        "disability_logit_const": -5.07217057388155,
        "disability_logit_period": 0.0606801631919178,
        "disability_logit_high_educ": -0.505096284451959,
        "disutil_unemployed_high_men": 1.7547023562030946,
        "disutil_unemployed_low_men": 1.285171306604484,
        "disutil_ft_work_high_good_men": 0.8209885491499525,
        "disutil_ft_work_high_bad_men": 1.217057196570522,
        "disutil_ft_work_low_good_men": 0.8340523194181079,
        "disutil_ft_work_low_bad_men": 1.1512416736832638,
        "job_finding_logit_const_men": -1.2721731052897,
        "job_finding_logit_period_men": -0.0475811807969158,
        "job_finding_logit_high_educ_men": 0.554225840026031,
        "disutil_unemployed_high_women": 1.2743640821491782,
        "disutil_unemployed_low_women": 0.8015361075897802,
        "disutil_ft_work_high_good_women": 1.2859040196900708,
        "disutil_ft_work_high_bad_women": 1.318077558189196,
        "disutil_ft_work_low_good_women": 1.4252533668814757,
        "disutil_ft_work_low_bad_women": 1.3562066466642761,
        "disutil_children_ft_work_low": 0.33568016747636004,
        "disutil_children_ft_work_high": 0.4429893409723915,
        "disutil_pt_work_high_good_women": 1.2537655230153704,
        "disutil_pt_work_high_bad_women": 1.3078048776544922,
        "disutil_pt_work_low_good_women": 1.1779907410723167,
        "disutil_pt_work_low_bad_women": 1.3516938584463878,
        "job_finding_logit_const_women": -1.234836517650335,
        "job_finding_logit_period_women": -0.0540965788852066,
        "job_finding_logit_high_educ_women": 0.911287401832803,
    }


else:
    last_estimate = None

estimation_results = estimate_model(
    paths_dict,
    params_to_estimate_names=params_to_estimate_names,
    file_append=model_name,
    load_model=LOAD_SOL_MODEL,
    use_weights=USE_WEIGHTS,
    last_estimate=last_estimate,
    save_results=SAVE_RESULTS,
)
print(estimation_results)


# %%
