import jax.numpy as jnp


def expected_SRA_probs(policy_state, options):
    trans_mat = options["beliefs_trans_mat"]
    trans_vector = jnp.take(trans_mat, policy_state, axis=0)
    return trans_vector
