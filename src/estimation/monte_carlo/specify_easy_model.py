import copy
import functools
import inspect

import dcegm
import jax
import jax.numpy as jnp
import numpy as np

from model_code.policy_processes.informed_state_transition import informed_transition
from model_code.policy_processes.select_policy_belief import (
    select_solution_transition_func_and_update_specs,
)
from model_code.policy_processes.step_function import realized_policy_step_function
from model_code.specify_model import cpu_build, create_model_config_wo_informed
from model_code.state_space.experience import experience_grid_from_state
from model_code.state_space.state_space import create_state_space_functions
from model_code.stochastic_processes.health_transition import health_transition
from model_code.stochastic_processes.job_offers import job_offer_process_transition
from model_code.stochastic_processes.partner_transitions import partner_transition
from model_code.taste_shocks import shock_function_dict
from model_code.utility.bequest_utility import create_final_period_utility_functions
from model_code.utility.utility_functions_add import create_utility_functions
from model_code.wealth_and_budget.budget_equation import budget_constraint

# Index of SRA 67 in the policy state grid of the easy model. The degenerate
# policy state of retirees has index 1.
SRA_67_INDEX = 0


def specify_easy_model(
    path_dict,
    specs,
    load_model=False,
    only_informed=True,
    informed_from_age=None,
    disabled_must_retire=False,
):
    """Specify the Monte Carlo model for low educated men without SRA uncertainty.

    The model uses the model functions of the full model (see
    model_code/specify_model.py) and restricts it as follows:

    - Sex and education are fixed to men and low education.
    - With only_informed, the informed state is a deterministic state with the
      single value 1, so all individuals are informed in the solution and the
      simulation. Otherwise informed is treated as in the full model: a
      deterministic state (type) with values 0 and 1 in the solution, and a
      stochastic state with the informed transition in the simulation.
    - With informed_from_age, everybody is informed from this age on in the
      simulation, i.e. uninformed individuals become informed with certainty in
      the period before. The solution is unchanged, as individuals don't anticipate
      a change of their informed state.
    - With disabled_must_retire, disabled individuals who are not retired yet can
      only choose retirement (see model_code/state_space/choice_set.py). Everybody
      who becomes disabled takes up the disability pension, so the disability
      probability is the probability of entering the disability pension.
    - There is no subjective uncertainty about the SRA. The policy state has only
      two values: SRA 67 and the degenerate policy state, which retirees move to.
      The transition is the realized step function without steps, restricted to
      these two values. An individual starting with SRA 67 therefore keeps it until
      retirement.

    dcegm requires the values of a stochastic state to be 0, ..., n - 1, while the
    model functions expect the policy state of the full model's grid. The policy
    state of the easy model is therefore the index 0 (SRA 67) or 1 (degenerate),
    and every model function with a policy state argument is wrapped to translate
    the index to the policy state of the full model's grid (and back for states it
    returns). Data and initial states hold the index.

    Apart from the informed state, the model is solved and simulated with the same
    specification.

    """
    specs = {**specs, "disabled_must_retire": disabled_must_retire}

    # Without subjective uncertainty, the policy state transition is the realized
    # step function and the specs get no step periods.
    _, specs = select_solution_transition_func_and_update_specs(
        specs=specs,
        subj_unc=False,
        custom_resolution_age=None,
    )

    # Policy states of the full model's grid for the two values of the easy model
    SRA_67_state = int((67 - specs["min_SRA"]) / specs["SRA_grid_size"])
    degenerate_state = int(specs["n_policy_states"] - 1)
    policy_state_values = np.array([SRA_67_state, degenerate_state], dtype=int)

    wrap = functools.partial(
        _translate_policy_state,
        SRA_67_state=SRA_67_state,
        degenerate_state=degenerate_state,
    )

    def policy_state_transition(
        policy_state, period, lagged_choice, choice, model_specs
    ):
        trans_vec_all = realized_policy_step_function(
            policy_state=policy_state,
            period=period,
            lagged_choice=lagged_choice,
            choice=choice,
            model_specs=model_specs,
        )
        return trans_vec_all[jnp.asarray(policy_state_values)]

    stochastic_states_transitions = {
        "policy_state": policy_state_transition,
        "job_offer": job_offer_process_transition,
        "partner_state": partner_transition,
        "health": health_transition,
    }

    model_config = create_model_config_wo_informed(
        specs=specs,
        sex_type="men",
        edu_type="low",
    )
    model_config["stochastic_states"]["policy_state"] = np.arange(2, dtype=int)
    if only_informed:
        model_config["deterministic_states"]["informed"] = np.array([1], dtype=int)
        alternative_sim_specifications = None
        model_name = "model_monte_carlo_easy.pkl"
    else:
        model_config["deterministic_states"]["informed"] = np.arange(2, dtype=int)
        alternative_sim_specifications = _alternative_sim_specifications(
            model_config=model_config,
            stochastic_states_transitions=stochastic_states_transitions,
            wrap=wrap,
            informed_from_age=informed_from_age,
        )
        model_name = "model_monte_carlo_easy_uninformed.pkl"
    if disabled_must_retire:
        model_name = model_name.replace(".pkl", "_disabled_must_retire.pkl")

    model_path = path_dict["intermediate_data"] + model_name
    if load_model:
        path_kwargs = {"model_load_path": model_path}
    else:
        path_kwargs = {"model_save_path": model_path}

    with cpu_build():
        model = dcegm.setup_model(
            model_specs=specs,
            model_config=model_config,
            state_space_functions=_wrap_dict(create_state_space_functions(), wrap),
            utility_functions=_wrap_dict(create_utility_functions(), wrap),
            utility_functions_final_period=_wrap_dict(
                create_final_period_utility_functions(), wrap
            ),
            budget_constraint=wrap(budget_constraint),
            shock_functions=_wrap_dict(shock_function_dict(), wrap),
            stochastic_states_transitions=_wrap_dict(
                stochastic_states_transitions, wrap
            ),
            continuous_grid_functions=_wrap_dict(
                {"experience": experience_grid_from_state}, wrap
            ),
            alternative_sim_specifications=alternative_sim_specifications,
            use_stochastic_sparsity=True,
            **path_kwargs,
        )

    print("Monte Carlo model specified.", flush=True)
    return model


def _alternative_sim_specifications(
    model_config, stochastic_states_transitions, wrap, informed_from_age
):
    """Simulate informed as a stochastic state, as in the full model (see
    define_alternative_sim_specifications in model_code/specify_model.py)."""
    sim_model_config = copy.deepcopy(model_config)
    del sim_model_config["deterministic_states"]["informed"]
    sim_model_config["stochastic_states"]["informed"] = np.arange(2, dtype=int)

    sim_transitions = {
        **stochastic_states_transitions,
        "informed": informed_transition,
    }
    if informed_from_age is not None:
        sim_transitions["informed"] = _create_informed_transition_until_age(
            informed_from_age
        )
    return {
        "model_config": sim_model_config,
        "stochastic_states_transitions": _wrap_dict(sim_transitions, wrap),
        "state_space_functions": _wrap_dict(create_state_space_functions(), wrap),
        "budget_constraint": wrap(budget_constraint),
        "shock_functions": _wrap_dict(shock_function_dict(), wrap),
    }


def _create_informed_transition_until_age(informed_from_age):
    """Informed transition, but everybody is informed from informed_from_age on.

    The transition is to the next period, so individuals become informed with
    certainty in the period before informed_from_age.

    """

    def informed_transition_until_age(choice, education, informed, period, model_specs):
        prob_vector = informed_transition(
            choice=choice,
            education=education,
            informed=informed,
            model_specs=model_specs,
        )
        next_age = model_specs["start_age"] + period + 1
        return jax.lax.select(
            next_age >= informed_from_age,
            on_true=jnp.array([0, 1], dtype=prob_vector.dtype),
            on_false=prob_vector,
        )

    return informed_transition_until_age


def _wrap_dict(func_dict, wrap):
    return {name: wrap(func) for name, func in func_dict.items()}


def _translate_policy_state(func, SRA_67_state, degenerate_state):
    """Let func take the policy state index of the easy model.

    The index is translated to the policy state of the full model's grid before
    calling func. If func returns a dict with a policy state (the proxy states of
    the sparsity condition), it is translated back to the index. Functions without
    a policy state argument are returned unchanged.

    """
    if "policy_state" not in inspect.signature(func).parameters:
        return func

    step = degenerate_state - SRA_67_state

    @functools.wraps(func)
    def wrapped(**kwargs):
        kwargs["policy_state"] = SRA_67_state + kwargs["policy_state"] * step
        out = func(**kwargs)
        if isinstance(out, dict) and "policy_state" in out:
            out = {**out, "policy_state": (out["policy_state"] - SRA_67_state) // step}
        return out

    return wrapped
