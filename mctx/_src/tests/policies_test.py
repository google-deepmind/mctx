# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for `policies.py`."""
import functools

from absl.testing import absltest
import jax
import jax.numpy as jnp
import mctx
from mctx._src import policies
import numpy as np


def _make_bandit_recurrent_fn(rewards, dummy_embedding=()):
  """Returns a recurrent_fn with discount=0."""

  def recurrent_fn(params, rng_key, action, embedding):
    del params, rng_key, embedding
    reward = rewards[jnp.arange(action.shape[0]), action]
    return mctx.RecurrentFnOutput(
        reward=reward,
        discount=jnp.zeros_like(reward),
        prior_logits=jnp.zeros_like(rewards),
        value=jnp.zeros_like(reward),
    ), dummy_embedding

  return recurrent_fn


def _make_bandit_decision_and_chance_fns(rewards, num_chance_outcomes):

  def decision_recurrent_fn(params, rng_key, action, embedding):
    del params, rng_key
    batch_size = action.shape[0]
    reward = rewards[jnp.arange(batch_size), action]
    dummy_chance_logits = jnp.full([batch_size, num_chance_outcomes],
                                   -jnp.inf).at[:, 0].set(1.0)
    afterstate_embedding = (action, embedding)
    return mctx.DecisionRecurrentFnOutput(
        chance_logits=dummy_chance_logits,
        afterstate_value=jnp.zeros_like(reward)), afterstate_embedding

  def chance_recurrent_fn(params, rng_key, chance_outcome,
                          afterstate_embedding):
    del params, rng_key, chance_outcome
    afterstate_action, embedding = afterstate_embedding
    batch_size = afterstate_action.shape[0]

    reward = rewards[jnp.arange(batch_size), afterstate_action]
    return mctx.ChanceRecurrentFnOutput(
        action_logits=jnp.zeros_like(rewards),
        value=jnp.zeros_like(reward),
        discount=jnp.zeros_like(reward),
        reward=reward), embedding

  return decision_recurrent_fn, chance_recurrent_fn


def _get_deepest_leaf(tree, node_index):
  """Returns `(leaf, depth)` with maximum depth and visit count.

  Args:
    tree: _unbatched_ MCTS tree state.
    node_index: the node of the inspected subtree.

  Returns:
    `(leaf, depth)` of a deepest leaf. If multiple leaves have the same depth,
    the leaf with the highest visit count is returned.
  """
  np.testing.assert_equal(len(tree.children_index.shape), 2)
  leaf = node_index
  max_found_depth = 0
  for action in range(tree.children_index.shape[-1]):
    next_node_index = tree.children_index[node_index, action]
    if next_node_index != tree.UNVISITED:
      found_leaf, found_depth = _get_deepest_leaf(tree, next_node_index)
      if ((1 + found_depth, tree.node_visits[found_leaf]) >
          (max_found_depth, tree.node_visits[leaf])):
        leaf = found_leaf
        max_found_depth = 1 + found_depth
  return leaf, max_found_depth


class PoliciesTest(absltest.TestCase):

  def test_apply_temperature_one(self):
    """Tests temperature=1."""
    logits = jnp.arange(6, dtype=jnp.float32)
    new_logits = policies._apply_temperature(logits, temperature=1.0)
    np.testing.assert_allclose(logits - logits.max(), new_logits)

  def test_apply_temperature_two(self):
    """Tests temperature=2."""
    logits = jnp.arange(6, dtype=jnp.float32)
    temperature = 2.0
    new_logits = policies._apply_temperature(logits, temperature)
    np.testing.assert_allclose((logits - logits.max()) / temperature,
                               new_logits)

  def test_apply_temperature_zero(self):
    """Tests temperature=0."""
    logits = jnp.arange(4, dtype=jnp.float32)
    new_logits = policies._apply_temperature(logits, temperature=0.0)
    np.testing.assert_allclose(
        jnp.array([-2.552118e+38, -1.701412e+38, -8.507059e+37, 0.0]),
        new_logits,
        rtol=1e-3)

  def test_apply_temperature_zero_on_large_logits(self):
    """Tests temperature=0 on large logits."""
    logits = jnp.array([100.0, 3.4028235e+38, -jnp.inf, -3.4028235e+38])
    new_logits = policies._apply_temperature(logits, temperature=0.0)
    np.testing.assert_allclose(
        jnp.array([-jnp.inf, 0.0, -jnp.inf, -jnp.inf]), new_logits)

  def test_mask_invalid_actions(self):
    """Tests action masking."""
    logits = jnp.array([1e6, -jnp.inf, 1e6 + 1, -100.0])
    invalid_actions = jnp.array([0.0, 1.0, 0.0, 1.0])
    masked_logits = policies._mask_invalid_actions(
        logits, invalid_actions)
    valid_probs = jax.nn.softmax(jnp.array([0.0, 1.0]))
    np.testing.assert_allclose(
        jnp.array([valid_probs[0], 0.0, valid_probs[1], 0.0]),
        jax.nn.softmax(masked_logits))

  def test_mask_all_invalid_actions(self):
    """Tests a state with no valid action."""
    logits = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf])
    invalid_actions = jnp.array([1.0, 1.0, 1.0, 1.0])
    masked_logits = policies._mask_invalid_actions(
        logits, invalid_actions)
    np.testing.assert_allclose(
        jnp.array([0.25, 0.25, 0.25, 0.25]),
        jax.nn.softmax(masked_logits))

  def test_muzero_policy(self):
    root = mctx.RootFnOutput(
        prior_logits=jnp.array([
            [-1.0, 0.0, 2.0, 3.0],
        ]),
        value=jnp.array([0.0]),
        embedding=(),
    )
    rewards = jnp.zeros_like(root.prior_logits)
    invalid_actions = jnp.array([
        [0.0, 0.0, 0.0, 1.0],
    ])

    policy_output = mctx.muzero_policy(
        params=(),
        rng_key=jax.random.PRNGKey(0),
        root=root,
        recurrent_fn=_make_bandit_recurrent_fn(rewards),
        num_simulations=1,
        invalid_actions=invalid_actions,
        dirichlet_fraction=0.0)
    expected_action = jnp.array([2], dtype=jnp.int32)
    np.testing.assert_array_equal(expected_action, policy_output.action)
    expected_action_weights = jnp.array([
        [0.0, 0.0, 1.0, 0.0],
    ])
    np.testing.assert_allclose(expected_action_weights,
                               policy_output.action_weights)

  def test_gumbel_muzero_policy(self):
    root_value = jnp.array([-5.0])
    root = mctx.RootFnOutput(
        prior_logits=jnp.array([
            [0.0, -1.0, 2.0, 3.0],
        ]),
        value=root_value,
        embedding=(),
    )
    rewards = jnp.array([
        [20.0, 3.0, -1.0, 10.0],
    ])
    invalid_actions = jnp.array([
        [1.0, 0.0, 0.0, 1.0],
    ])

    value_scale = 0.05
    maxvisit_init = 60
    num_simulations = 17
    max_depth = 3
    qtransform = functools.partial(
        mctx.qtransform_completed_by_mix_value,
        value_scale=value_scale,
        maxvisit_init=maxvisit_init,
        rescale_values=True)
    policy_output = mctx.gumbel_muzero_policy(
        params=(),
        rng_key=jax.random.PRNGKey(0),
        root=root,
        recurrent_fn=_make_bandit_recurrent_fn(rewards),
        num_simulations=num_simulations,
        invalid_actions=invalid_actions,
        max_depth=max_depth,
        qtransform=qtransform,
        gumbel_scale=1.0)
    # Testing the action.
    expected_action = jnp.array([1], dtype=jnp.int32)
    np.testing.assert_array_equal(expected_action, policy_output.action)

    # Testing the action_weights.
    probs = jax.nn.softmax(jnp.where(
        invalid_actions, -jnp.inf, root.prior_logits))
    mix_value = 1.0 / (num_simulations + 1) * (root_value + num_simulations * (
        probs[:, 1] * rewards[:, 1] + probs[:, 2] * rewards[:, 2]))

    completed_qvalues = jnp.array([
        [mix_value[0], rewards[0, 1], rewards[0, 2], mix_value[0]],
    ])
    max_value = jnp.max(completed_qvalues, axis=-1, keepdims=True)
    min_value = jnp.min(completed_qvalues, axis=-1, keepdims=True)
    total_value_scale = (maxvisit_init + np.ceil(num_simulations / 2)
                         ) * value_scale
    rescaled_qvalues = total_value_scale * (completed_qvalues - min_value) / (
        max_value - min_value)
    expected_action_weights = jax.nn.softmax(
        jnp.where(invalid_actions,
                  -jnp.inf,
                  root.prior_logits + rescaled_qvalues))
    np.testing.assert_allclose(expected_action_weights,
                               policy_output.action_weights,
                               atol=1e-6)

    # Testing the visit_counts.
    summary = policy_output.search_tree.summary()
    expected_visit_counts = jnp.array(
        [[0.0, np.ceil(num_simulations / 2), num_simulations // 2, 0.0]])
    np.testing.assert_array_equal(expected_visit_counts, summary.visit_counts)

    # Testing max_depth.
    leaf, max_found_depth = _get_deepest_leaf(
        jax.tree_util.tree_map(lambda x: x[0], policy_output.search_tree),
        policy_output.search_tree.ROOT_INDEX)
    self.assertEqual(max_depth, max_found_depth)
    self.assertEqual(6, policy_output.search_tree.node_visits[0, leaf])

  def test_gumbel_muzero_policy_without_invalid_actions(self):
    root_value = jnp.array([-5.0])
    root = mctx.RootFnOutput(
        prior_logits=jnp.array([
            [0.0, -1.0, 2.0, 3.0],
        ]),
        value=root_value,
        embedding=(),
    )
    rewards = jnp.array([
        [20.0, 3.0, -1.0, 10.0],
    ])

    value_scale = 0.05
    maxvisit_init = 60
    num_simulations = 17
    max_depth = 3
    qtransform = functools.partial(
        mctx.qtransform_completed_by_mix_value,
        value_scale=value_scale,
        maxvisit_init=maxvisit_init,
        rescale_values=True)
    policy_output = mctx.gumbel_muzero_policy(
        params=(),
        rng_key=jax.random.PRNGKey(0),
        root=root,
        recurrent_fn=_make_bandit_recurrent_fn(rewards),
        num_simulations=num_simulations,
        invalid_actions=None,
        max_depth=max_depth,
        qtransform=qtransform,
        gumbel_scale=1.0)
    # Testing the action.
    expected_action = jnp.array([3], dtype=jnp.int32)
    np.testing.assert_array_equal(expected_action, policy_output.action)

    # Testing the action_weights.
    summary = policy_output.search_tree.summary()
    completed_qvalues = rewards
    max_value = jnp.max(completed_qvalues, axis=-1, keepdims=True)
    min_value = jnp.min(completed_qvalues, axis=-1, keepdims=True)
    total_value_scale = (maxvisit_init + summary.visit_counts.max()
                         ) * value_scale
    rescaled_qvalues = total_value_scale * (completed_qvalues - min_value) / (
        max_value - min_value)
    expected_action_weights = jax.nn.softmax(
        root.prior_logits + rescaled_qvalues)
    np.testing.assert_allclose(expected_action_weights,
                               policy_output.action_weights,
                               atol=1e-6)

    # Testing the visit_counts.
    expected_visit_counts = jnp.array(
        [[6, 2, 2, 7]])
    np.testing.assert_array_equal(expected_visit_counts, summary.visit_counts)

  def test_stochastic_muzero_policy(self):
    """Tests that SMZ is equivalent to MZ with a dummy chance function."""
    root = mctx.RootFnOutput(
        prior_logits=jnp.array([
            [-1.0, 0.0, 2.0, 3.0],
            [0.0, 2.0, 5.0, -4.0],
        ]),
        value=jnp.array([1.0, 0.0]),
        embedding=jnp.zeros([2, 4])
    )
    rewards = jnp.zeros_like(root.prior_logits)
    invalid_actions = jnp.array([
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
    ])

    num_simulations = 10

    policy_output = mctx.muzero_policy(
        params=(),
        rng_key=jax.random.PRNGKey(0),
        root=root,
        recurrent_fn=_make_bandit_recurrent_fn(
            rewards,
            dummy_embedding=jnp.zeros_like(root.embedding)),
        num_simulations=num_simulations,
        invalid_actions=invalid_actions,
        dirichlet_fraction=0.0)

    num_chance_outcomes = 5

    decision_rec_fn, chance_rec_fn = _make_bandit_decision_and_chance_fns(
        rewards, num_chance_outcomes)

    stochastic_policy_output = mctx.stochastic_muzero_policy(
        params=(),
        rng_key=jax.random.PRNGKey(0),
        root=root,
        decision_recurrent_fn=decision_rec_fn,
        chance_recurrent_fn=chance_rec_fn,
        num_simulations=2 * num_simulations,
        num_actions=4,
        num_chance_outcomes=num_chance_outcomes,
        invalid_actions=invalid_actions,
        dirichlet_fraction=0.0)

    np.testing.assert_array_equal(stochastic_policy_output.action,
                                  policy_output.action)

    np.testing.assert_allclose(stochastic_policy_output.action_weights,
                               policy_output.action_weights)


if __name__ == "__main__":
  absltest.main()
