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
"""A collection of action selection functions."""
from typing import Optional, TypeVar

import chex
import jax
import jax.numpy as jnp

from mctx._src import base
from mctx._src import qtransforms
from mctx._src import seq_halving
from mctx._src import tree as tree_lib


def switching_action_selection_wrapper(
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn
) -> base.InteriorActionSelectionFn:
  """Wraps root and interior action selection fns in a conditional statement."""

  def switching_action_selection_fn(
      rng_key: chex.PRNGKey,
      tree: tree_lib.Tree,
      node_index: base.NodeIndices,
      depth: base.Depth) -> chex.Array:
    return jax.lax.cond(
        depth == 0,
        lambda x: root_action_selection_fn(*x[:3]),
        lambda x: interior_action_selection_fn(*x),
        (rng_key, tree, node_index, depth))

  return switching_action_selection_fn


def muzero_action_selection(
    rng_key: chex.PRNGKey,
    tree: tree_lib.Tree,
    node_index: chex.Numeric,
    depth: chex.Numeric,
    *,
    pb_c_init: float = 1.25,
    pb_c_base: float = 19652.0,
    qtransform: base.QTransform = qtransforms.qtransform_by_parent_and_siblings,
) -> chex.Array:
  """Returns the action selected for a node index.

  See Appendix B in https://arxiv.org/pdf/1911.08265.pdf for more details.

  Args:
    rng_key: random number generator state.
    tree: _unbatched_ MCTS tree state.
    node_index: scalar index of the node from which to select an action.
    depth: the scalar depth of the current node. The root has depth zero.
    pb_c_init: constant c_1 in the PUCT formula.
    pb_c_base: constant c_2 in the PUCT formula.
    qtransform: a monotonic transformation to convert the Q-values to [0, 1].

  Returns:
    action: the action selected from the given node.
  """
  visit_counts = tree.children_visits[node_index]
  node_visit = tree.node_visits[node_index]
  pb_c = pb_c_init + jnp.log((node_visit + pb_c_base + 1.) / pb_c_base)
  prior_logits = tree.children_prior_logits[node_index]
  prior_probs = jax.nn.softmax(prior_logits)
  policy_score = jnp.sqrt(node_visit) * pb_c * prior_probs / (visit_counts + 1)
  chex.assert_shape([node_index, node_visit], ())
  chex.assert_equal_shape([prior_probs, visit_counts, policy_score])
  value_score = qtransform(tree, node_index)

  # Add tiny bit of randomness for tie break
  node_noise_score = 1e-7 * jax.random.uniform(
      rng_key, (tree.num_actions,))
  to_argmax = value_score + policy_score + node_noise_score

  # Masking the invalid actions at the root.
  return masked_argmax(to_argmax, tree.root_invalid_actions * (depth == 0))


@chex.dataclass(frozen=True)
class GumbelMuZeroExtraData:
  """Extra data for Gumbel MuZero search."""
  root_gumbel: chex.Array


GumbelMuZeroExtraDataType = TypeVar(  # pylint: disable=invalid-name
    "GumbelMuZeroExtraDataType", bound=GumbelMuZeroExtraData)


def gumbel_muzero_root_action_selection(
    rng_key: chex.PRNGKey,
    tree: tree_lib.Tree[GumbelMuZeroExtraDataType],
    node_index: chex.Numeric,
    *,
    num_simulations: chex.Numeric,
    max_num_considered_actions: chex.Numeric,
    qtransform: base.QTransform = qtransforms.qtransform_completed_by_mix_value,
) -> chex.Array:
  """Returns the action selected by Sequential Halving with Gumbel.

  Initially, we sample `max_num_considered_actions` actions without replacement.
  From these, the actions with the highest `gumbel + logits + qvalues` are
  visited first.

  Args:
    rng_key: random number generator state.
    tree: _unbatched_ MCTS tree state.
    node_index: scalar index of the node from which to take an action.
    num_simulations: the simulation budget.
    max_num_considered_actions: the number of actions sampled without
      replacement.
    qtransform: a monotonic transformation for the Q-values.

  Returns:
    action: the action selected from the given node.
  """
  del rng_key
  chex.assert_shape([node_index], ())
  visit_counts = tree.children_visits[node_index]
  prior_logits = tree.children_prior_logits[node_index]
  chex.assert_equal_shape([visit_counts, prior_logits])
  completed_qvalues = qtransform(tree, node_index)

  table = jnp.array(seq_halving.get_table_of_considered_visits(
      max_num_considered_actions, num_simulations))
  num_valid_actions = jnp.sum(
      1 - tree.root_invalid_actions, axis=-1).astype(jnp.int32)
  num_considered = jnp.minimum(
      max_num_considered_actions, num_valid_actions)
  chex.assert_shape(num_considered, ())
  # At the root, the simulation_index is equal to the sum of visit counts.
  simulation_index = jnp.sum(visit_counts, -1)
  chex.assert_shape(simulation_index, ())
  considered_visit = table[num_considered, simulation_index]
  chex.assert_shape(considered_visit, ())
  gumbel = tree.extra_data.root_gumbel
  to_argmax = seq_halving.score_considered(
      considered_visit, gumbel, prior_logits, completed_qvalues,
      visit_counts)

  # Masking the invalid actions at the root.
  return masked_argmax(to_argmax, tree.root_invalid_actions)


def gumbel_muzero_interior_action_selection(
    rng_key: chex.PRNGKey,
    tree: tree_lib.Tree,
    node_index: chex.Numeric,
    depth: chex.Numeric,
    *,
    qtransform: base.QTransform = qtransforms.qtransform_completed_by_mix_value,
) -> chex.Array:
  """Selects the action with a deterministic action selection.

  The action is selected based on the visit counts to produce visitation
  frequencies similar to softmax(prior_logits + qvalues).

  Args:
    rng_key: random number generator state.
    tree: _unbatched_ MCTS tree state.
    node_index: scalar index of the node from which to take an action.
    depth: the scalar depth of the current node. The root has depth zero.
    qtransform: function to obtain completed Q-values for a node.

  Returns:
    action: the action selected from the given node.
  """
  del rng_key, depth
  chex.assert_shape([node_index], ())
  visit_counts = tree.children_visits[node_index]
  prior_logits = tree.children_prior_logits[node_index]
  chex.assert_equal_shape([visit_counts, prior_logits])
  completed_qvalues = qtransform(tree, node_index)

  # The `prior_logits + completed_qvalues` provide an improved policy,
  # because the missing qvalues are replaced by v_{prior_logits}(node).
  to_argmax = _prepare_argmax_input(
      probs=jax.nn.softmax(prior_logits + completed_qvalues),
      visit_counts=visit_counts)

  chex.assert_rank(to_argmax, 1)
  return jnp.argmax(to_argmax, axis=-1)


def masked_argmax(
    to_argmax: chex.Array,
    invalid_actions: Optional[chex.Array]) -> chex.Array:
  """Returns a valid action with the highest `to_argmax`."""
  if invalid_actions is not None:
    chex.assert_equal_shape([to_argmax, invalid_actions])
    # The usage of the -inf inside the argmax does not lead to NaN.
    # Do not use -inf inside softmax, logsoftmax or cross-entropy.
    to_argmax = jnp.where(invalid_actions, -jnp.inf, to_argmax)
  # If all actions are invalid, the argmax returns action 0.
  return jnp.argmax(to_argmax, axis=-1)


def _prepare_argmax_input(probs, visit_counts):
  """Prepares the input for the deterministic selection.

  When calling argmax(_prepare_argmax_input(...)) multiple times
  with updated visit_counts, the produced visitation frequencies will
  approximate the probs.

  For the derivation, see Section 5 "Planning at non-root nodes" in
  "Policy improvement by planning with Gumbel":
  https://openreview.net/forum?id=bERaNdoegnO

  Args:
    probs: a policy or an improved policy. Shape `[num_actions]`.
    visit_counts: the existing visit counts. Shape `[num_actions]`.

  Returns:
    The input to an argmax. Shape `[num_actions]`.
  """
  chex.assert_equal_shape([probs, visit_counts])
  to_argmax = probs - visit_counts / (
      1 + jnp.sum(visit_counts, keepdims=True, axis=-1))
  return to_argmax
