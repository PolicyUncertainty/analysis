"""Expected retirement age computation module.

This module computes expected retirement age conditional on current state
using forward simulation of transition matrices.
"""
import numpy as np


def compute_expected_retirement_age(
    current_state_dict,
    transition_matrix_function,
    choice_state="partner_state",
    retired_state=2,
    start_age=30,
    max_periods=46
):
    """
    Compute expected retirement age conditional on current state.
    
    Args:
        current_state_dict: Dict with state variables (period, education, sex, partner_state, etc.)
        transition_matrix_function: Function that takes state elements and returns transition probs
        choice_state: State variable to track for retirement (default: 'partner_state')
        retired_state: Value indicating retirement (default: 2)
        start_age: Starting age in model (default: 30)
        max_periods: Maximum number of periods (default: 46 for ages 30-75)
    
    Returns:
        Expected retirement age as float
    """
    # Extract current state
    period = current_state_dict["period"]
    current_choice_state = current_state_dict[choice_state]
    
    # If already retired, return current age
    if current_choice_state == retired_state:
        return period + start_age
    
    # Number of states (single=0, working=1, retired=2)
    n_states = 3
    
    # Initialize probability distribution
    state_probs = np.zeros(n_states)
    state_probs[current_choice_state] = 1.0
    
    # Track expected retirement age
    expected_ret_age = 0.0
    cumulative_ret_prob = 0.0
    
    # Forward simulate
    for t in range(period, min(max_periods - 1, period + 50)):  # Cap lookahead
        # Build transition matrix for current period
        trans_matrix = np.zeros((n_states, n_states))
        state_dict_t = current_state_dict.copy()
        state_dict_t["period"] = t
        
        for s in range(n_states):
            state_dict_t[choice_state] = s
            trans_matrix[s, :] = transition_matrix_function(**state_dict_t)
        
        # Calculate probability of retiring exactly at age t+1
        # (was not retired at t, becomes retired at t+1)
        prob_retire_now = 0.0
        for s in range(n_states):
            if s != retired_state:  # From non-retired states
                prob_retire_now += state_probs[s] * trans_matrix[s, retired_state]
        
        # Add to expected retirement age
        retirement_age = t + 1 + start_age
        expected_ret_age += prob_retire_now * retirement_age
        cumulative_ret_prob += prob_retire_now
        
        # Update state distribution for next iteration
        state_probs = state_probs @ trans_matrix
        
        # Early stop if almost everyone retired
        if cumulative_ret_prob > 0.999:
            break
    
    # Handle case where some never retire
    if cumulative_ret_prob < 1.0 and cumulative_ret_prob > 0:
        # Assign max age to those who never retire
        expected_ret_age += (1.0 - cumulative_ret_prob) * (max_periods - 1 + start_age)
        cumulative_ret_prob = 1.0
    
    # Return expected age (normalized if needed)
    if cumulative_ret_prob > 0:
        return expected_ret_age / cumulative_ret_prob
    else:
        return max_periods - 1 + start_age