def informed_transition(period, education, informed_state, options):
    """Transition function for informed state. Not period-specific right now but can be amended."""
    if informed_state == 0:
        probability_informed = options["informed_hazard_rate"][education]
    else:
        probability_informed = 1
    return probability_informed