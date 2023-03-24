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
"""Search policies."""
import functools
from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp

from mctx._src import action_selection
from mctx._src import base
from mctx._src import qtransforms
from mctx._src import search
from mctx._src import seq_halving


def muzero_policy(
    params: base.Params,
    rng_key: chex.PRNGKey,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    num_simulations: int,
    invalid_actions: Optional[chex.Array] = None,
    max_depth: Optional[int] = None,
    loop_fn: base.LoopFn = jax.lax.fori_loop,
    *,
    qtransform: base.QTransform = qtransforms.qtransform_by_parent_and_siblings,
    dirichlet_fraction: chex.Numeric = 0.25,
    dirichlet_alpha: chex.Numeric = 0.3,
    pb_c_init: chex.Numeric = 1.25,
    pb_c_base: chex.Numeric = 19652,
    temperature: chex.Numeric = 1.0) -> base.PolicyOutput[None]:
  """Runs MuZero search and returns the `PolicyOutput`.

  In the shape descriptions, `B` denotes the batch dimension.

  Args:
    params: params to be forwarded to root and recurrent functions.
    rng_key: random number generator state, the key is consumed.
    root: a `(prior_logits, value, embedding)` `RootFnOutput`. The
      `prior_logits` are from a policy network. The shapes are
      `([B, num_actions], [B], [B, ...])`, respectively.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    num_simulations: the number of simulations.
    invalid_actions: a mask with invalid actions. Invalid actions
      have ones, valid actions have zeros in the mask. Shape `[B, num_actions]`.
    max_depth: maximum search tree depth allowed during simulation.
    loop_fn: Function used to run the simulations. It may be required to pass
      hk.fori_loop if using this function inside a Haiku module.
    qtransform: function to obtain completed Q-values for a node.
    dirichlet_fraction: float from 0 to 1 interpolating between using only the
      prior policy or just the Dirichlet noise.
    dirichlet_alpha: concentration parameter to parametrize the Dirichlet
      distribution.
    pb_c_init: constant c_1 in the PUCT formula.
    pb_c_base: constant c_2 in the PUCT formula.
    temperature: temperature for acting proportionally to
      `visit_counts**(1 / temperature)`.

  Returns:
    `PolicyOutput` containing the proposed action, action_weights and the used
    search tree.
  """
  rng_key, dirichlet_rng_key, search_rng_key = jax.random.split(rng_key, 3)

  # Adding Dirichlet noise.
  noisy_logits = _get_logits_from_probs(
      _add_dirichlet_noise(
          dirichlet_rng_key,
          jax.nn.softmax(root.prior_logits),
          dirichlet_fraction=dirichlet_fraction,
          dirichlet_alpha=dirichlet_alpha))
  root = root.replace(
      prior_logits=_mask_invalid_actions(noisy_logits, invalid_actions))

  # Running the search.
  interior_action_selection_fn = functools.partial(
      action_selection.muzero_action_selection,
      pb_c_base=pb_c_base,
      pb_c_init=pb_c_init,
      qtransform=qtransform)
  root_action_selection_fn = functools.partial(
      interior_action_selection_fn,
      depth=0)
  search_tree = search.search(
      params=params,
      rng_key=search_rng_key,
      root=root,
      recurrent_fn=recurrent_fn,
      root_action_selection_fn=root_action_selection_fn,
      interior_action_selection_fn=interior_action_selection_fn,
      num_simulations=num_simulations,
      max_depth=max_depth,
      invalid_actions=invalid_actions,
      loop_fn=loop_fn)

  # Sampling the proposed action proportionally to the visit counts.
  summary = search_tree.summary()
  action_weights = summary.visit_probs
  action_logits = _apply_temperature(
      _get_logits_from_probs(action_weights), temperature)
  action = jax.random.categorical(rng_key, action_logits)
  return base.PolicyOutput(
      action=action,
      action_weights=action_weights,
      search_tree=search_tree)


def gumbel_muzero_policy(
    params: base.Params,
    rng_key: chex.PRNGKey,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    num_simulations: int,
    invalid_actions: Optional[chex.Array] = None,
    max_depth: Optional[int] = None,
    loop_fn: base.LoopFn = jax.lax.fori_loop,
    *,
    qtransform: base.QTransform = qtransforms.qtransform_completed_by_mix_value,
    max_num_considered_actions: int = 16,
    gumbel_scale: chex.Numeric = 1.,
) -> base.PolicyOutput[action_selection.GumbelMuZeroExtraData]:
  """Runs Gumbel MuZero search and returns the `PolicyOutput`.

  This policy implements Full Gumbel MuZero from
  "Policy improvement by planning with Gumbel".
  https://openreview.net/forum?id=bERaNdoegnO

  At the root of the search tree, actions are selected by Sequential Halving
  with Gumbel. At non-root nodes (aka interior nodes), actions are selected by
  the Full Gumbel MuZero deterministic action selection.

  In the shape descriptions, `B` denotes the batch dimension.

  Args:
    params: params to be forwarded to root and recurrent functions.
    rng_key: random number generator state, the key is consumed.
    root: a `(prior_logits, value, embedding)` `RootFnOutput`. The
      `prior_logits` are from a policy network. The shapes are
      `([B, num_actions], [B], [B, ...])`, respectively.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    num_simulations: the number of simulations.
    invalid_actions: a mask with invalid actions. Invalid actions
      have ones, valid actions have zeros in the mask. Shape `[B, num_actions]`.
    max_depth: maximum search tree depth allowed during simulation.
    loop_fn: Function used to run the simulations. It may be required to pass
      hk.fori_loop if using this function inside a Haiku module.
    qtransform: function to obtain completed Q-values for a node.
    max_num_considered_actions: the maximum number of actions expanded at the
      root node. A smaller number of actions will be expanded if the number of
      valid actions is smaller.
    gumbel_scale: scale for the Gumbel noise. Evalution on perfect-information
      games can use gumbel_scale=0.0.

  Returns:
    `PolicyOutput` containing the proposed action, action_weights and the used
    search tree.
  """
  # Masking invalid actions.
  root = root.replace(
      prior_logits=_mask_invalid_actions(root.prior_logits, invalid_actions))

  # Generating Gumbel.
  rng_key, gumbel_rng = jax.random.split(rng_key)
  gumbel = gumbel_scale * jax.random.gumbel(
      gumbel_rng, shape=root.prior_logits.shape, dtype=root.prior_logits.dtype)

  # Searching.
  extra_data = action_selection.GumbelMuZeroExtraData(root_gumbel=gumbel)
  search_tree = search.search(
      params=params,
      rng_key=rng_key,
      root=root,
      recurrent_fn=recurrent_fn,
      root_action_selection_fn=functools.partial(
          action_selection.gumbel_muzero_root_action_selection,
          num_simulations=num_simulations,
          max_num_considered_actions=max_num_considered_actions,
          qtransform=qtransform,
      ),
      interior_action_selection_fn=functools.partial(
          action_selection.gumbel_muzero_interior_action_selection,
          qtransform=qtransform,
      ),
      num_simulations=num_simulations,
      max_depth=max_depth,
      invalid_actions=invalid_actions,
      extra_data=extra_data,
      loop_fn=loop_fn)
  summary = search_tree.summary()

  # Acting with the best action from the most visited actions.
  # The "best" action has the highest `gumbel + logits + q`.
  # Inside the minibatch, the considered_visit can be different on states with
  # a smaller number of valid actions.
  considered_visit = jnp.max(summary.visit_counts, axis=-1, keepdims=True)
  # The completed_qvalues include imputed values for unvisited actions.
  completed_qvalues = jax.vmap(qtransform, in_axes=[0, None])(  # pytype: disable=wrong-arg-types  # numpy-scalars  # pylint: disable=line-too-long
      search_tree, search_tree.ROOT_INDEX)
  to_argmax = seq_halving.score_considered(
      considered_visit, gumbel, root.prior_logits, completed_qvalues,
      summary.visit_counts)
  action = action_selection.masked_argmax(to_argmax, invalid_actions)

  # Producing action_weights usable to train the policy network.
  completed_search_logits = _mask_invalid_actions(
      root.prior_logits + completed_qvalues, invalid_actions)
  action_weights = jax.nn.softmax(completed_search_logits)
  return base.PolicyOutput(
      action=action,
      action_weights=action_weights,
      search_tree=search_tree)


def stochastic_muzero_policy(
    params: chex.ArrayTree,
    rng_key: chex.PRNGKey,
    root: base.RootFnOutput,
    decision_recurrent_fn: base.DecisionRecurrentFn,
    chance_recurrent_fn: base.ChanceRecurrentFn,
    num_simulations: int,
    num_actions: int,
    num_chance_outcomes: int,
    invalid_actions: Optional[chex.Array] = None,
    max_depth: Optional[int] = None,
    loop_fn: base.LoopFn = jax.lax.fori_loop,
    *,
    qtransform: base.QTransform = qtransforms.qtransform_by_parent_and_siblings,
    dirichlet_fraction: chex.Numeric = 0.25,
    dirichlet_alpha: chex.Numeric = 0.3,
    pb_c_init: chex.Numeric = 1.25,
    pb_c_base: chex.Numeric = 19652,
    temperature: chex.Numeric = 1.0) -> base.PolicyOutput[None]:
  """Runs Stochastic MuZero search.

  Implements search as described in the Stochastic MuZero paper:
    (https://openreview.net/forum?id=X6D9bAHhBQ1).

  In the shape descriptions, `B` denotes the batch dimension.
  Args:
    params: params to be forwarded to root and recurrent functions.
    rng_key: random number generator state, the key is consumed.
    root: a `(prior_logits, value, embedding)` `RootFnOutput`. The
      `prior_logits` are from a policy network. The shapes are `([B,
      num_actions], [B], [B, ...])`, respectively.
    decision_recurrent_fn: a callable to be called on the leaf decision nodes
      and unvisited actions retrieved by the simulation step, which takes as
      args `(params, rng_key, action, state_embedding)` and returns a
      `(DecisionRecurrentFnOutput, afterstate_embedding)`.
    chance_recurrent_fn:  a callable to be called on the leaf chance nodes and
      unvisited actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, afterstate_embedding)` and returns a
      `(ChanceRecurrentFnOutput, state_embedding)`.
    num_simulations: the number of simulations.
    num_actions: number of environment actions.
    num_chance_outcomes: number of chance outcomes following an afterstate.
    invalid_actions: a mask with invalid actions. Invalid actions have ones,
      valid actions have zeros in the mask. Shape `[B, num_actions]`.
    max_depth: maximum search tree depth allowed during simulation.
    loop_fn: Function used to run the simulations. It may be required to pass
      hk.fori_loop if using this function inside a Haiku module.
    qtransform: function to obtain completed Q-values for a node.
    dirichlet_fraction: float from 0 to 1 interpolating between using only the
      prior policy or just the Dirichlet noise.
    dirichlet_alpha: concentration parameter to parametrize the Dirichlet
      distribution.
    pb_c_init: constant c_1 in the PUCT formula.
    pb_c_base: constant c_2 in the PUCT formula.
    temperature: temperature for acting proportionally to `visit_counts**(1 /
      temperature)`.

  Returns:
    `PolicyOutput` containing the proposed action, action_weights and the used
    search tree.
  """

  rng_key, dirichlet_rng_key, search_rng_key = jax.random.split(rng_key, 3)

  # Adding Dirichlet noise.
  noisy_logits = _get_logits_from_probs(
      _add_dirichlet_noise(
          dirichlet_rng_key,
          jax.nn.softmax(root.prior_logits),
          dirichlet_fraction=dirichlet_fraction,
          dirichlet_alpha=dirichlet_alpha))

  root = root.replace(
      prior_logits=_mask_invalid_actions(noisy_logits, invalid_actions))

  # construct a dummy afterstate embedding
  batch_size = jax.tree_util.tree_leaves(root.embedding)[0].shape[0]
  dummy_action = jnp.zeros([batch_size], dtype=jnp.int32)
  _, dummy_afterstate_embedding = decision_recurrent_fn(params, rng_key,
                                                        dummy_action,
                                                        root.embedding)

  root = root.replace(
      # pad action logits with num_chance_outcomes so dim is A + C
      prior_logits=jnp.concatenate([
          root.prior_logits,
          jnp.full([batch_size, num_chance_outcomes], fill_value=-jnp.inf)
      ], axis=-1),
      # replace embedding with wrapper.
      embedding=base.StochasticRecurrentState(
          state_embedding=root.embedding,
          afterstate_embedding=dummy_afterstate_embedding,
          is_decision_node=jnp.ones([batch_size], dtype=bool)))

  # Stochastic MuZero Change: We need to be able to tell if different nodes are
  # decision or chance. This is accomplished by imposing a special structure
  # on the embeddings stored in each node. Each embedding is an instance of
  # StochasticRecurrentState which maintains this information.
  recurrent_fn = _make_stochastic_recurrent_fn(
      decision_node_fn=decision_recurrent_fn,
      chance_node_fn=chance_recurrent_fn,
      num_actions=num_actions,
      num_chance_outcomes=num_chance_outcomes,
  )

  # Running the search.

  interior_decision_node_selection_fn = functools.partial(
      action_selection.muzero_action_selection,
      pb_c_base=pb_c_base,
      pb_c_init=pb_c_init,
      qtransform=qtransform)

  interior_action_selection_fn = _make_stochastic_action_selection_fn(
      interior_decision_node_selection_fn, num_actions)

  root_action_selection_fn = functools.partial(
      interior_action_selection_fn, depth=0)

  search_tree = search.search(
      params=params,
      rng_key=search_rng_key,
      root=root,
      recurrent_fn=recurrent_fn,
      root_action_selection_fn=root_action_selection_fn,
      interior_action_selection_fn=interior_action_selection_fn,
      num_simulations=num_simulations,
      max_depth=max_depth,
      invalid_actions=invalid_actions,
      loop_fn=loop_fn)

  # Sampling the proposed action proportionally to the visit counts.
  search_tree = _mask_tree(search_tree, num_actions, 'decision')
  summary = search_tree.summary()
  action_weights = summary.visit_probs
  action_logits = _apply_temperature(
      _get_logits_from_probs(action_weights), temperature)
  action = jax.random.categorical(rng_key, action_logits)
  return base.PolicyOutput(
      action=action, action_weights=action_weights, search_tree=search_tree)


def _mask_invalid_actions(logits, invalid_actions):
  """Returns logits with zero mass to invalid actions."""
  if invalid_actions is None:
    return logits
  chex.assert_equal_shape([logits, invalid_actions])
  logits = logits - jnp.max(logits, axis=-1, keepdims=True)
  # At the end of an episode, all actions can be invalid. A softmax would then
  # produce NaNs, if using -inf for the logits. We avoid the NaNs by using
  # a finite `min_logit` for the invalid actions.
  min_logit = jnp.finfo(logits.dtype).min
  return jnp.where(invalid_actions, min_logit, logits)


def _get_logits_from_probs(probs):
  tiny = jnp.finfo(probs).tiny
  return jnp.log(jnp.maximum(probs, tiny))


def _add_dirichlet_noise(rng_key, probs, *, dirichlet_alpha,
                         dirichlet_fraction):
  """Mixes the probs with Dirichlet noise."""
  chex.assert_rank(probs, 2)
  chex.assert_type([dirichlet_alpha, dirichlet_fraction], float)

  batch_size, num_actions = probs.shape
  noise = jax.random.dirichlet(
      rng_key,
      alpha=jnp.full([num_actions], fill_value=dirichlet_alpha),
      shape=(batch_size,))
  noisy_probs = (1 - dirichlet_fraction) * probs + dirichlet_fraction * noise
  return noisy_probs


def _apply_temperature(logits, temperature):
  """Returns `logits / temperature`, supporting also temperature=0."""
  # The max subtraction prevents +inf after dividing by a small temperature.
  logits = logits - jnp.max(logits, keepdims=True, axis=-1)
  tiny = jnp.finfo(logits.dtype).tiny
  return logits / jnp.maximum(tiny, temperature)


def _make_stochastic_recurrent_fn(
    decision_node_fn: base.DecisionRecurrentFn,
    chance_node_fn: base.ChanceRecurrentFn,
    num_actions: int,
    num_chance_outcomes: int,
) -> base.RecurrentFn:
  """Make Stochastic Recurrent Fn."""

  def stochastic_recurrent_fn(
      params: base.Params,
      rng: chex.PRNGKey,
      action_or_chance: base.Action,  # [B]
      state: base.StochasticRecurrentState
  ) -> Tuple[base.RecurrentFnOutput, base.StochasticRecurrentState]:
    batch_size = jax.tree_util.tree_leaves(state.state_embedding)[0].shape[0]
    # Internally we assume that there are `A' = A + C` "actions";
    # action_or_chance can take on values in `{0, 1, ..., A' - 1}`,.
    # To interpret it as an action we can leave it as is:
    action = action_or_chance - 0
    # To interpret it as a chance outcome we subtract num_actions:
    chance_outcome = action_or_chance - num_actions

    decision_output, afterstate_embedding = decision_node_fn(
        params, rng, action, state.state_embedding)
    # Outputs from DecisionRecurrentFunction produce chance logits with
    # dim `C`, to respect our internal convention that there are `A' = A + C`
    # "actions" we pad with `A` dummy logits which are ultimately ignored:
    # see `_mask_tree`.
    output_if_decision_node = base.RecurrentFnOutput(
        prior_logits=jnp.concatenate([
            jnp.full([batch_size, num_actions], fill_value=-jnp.inf),
            decision_output.chance_logits], axis=-1),
        value=decision_output.afterstate_value,
        reward=jnp.zeros_like(decision_output.afterstate_value),
        discount=jnp.ones_like(decision_output.afterstate_value))

    chance_output, state_embedding = chance_node_fn(params, rng, chance_outcome,
                                                    state.afterstate_embedding)
    # Outputs from ChanceRecurrentFunction produce action logits with dim `A`,
    # to respect our internal convention that there are `A' = A + C` "actions"
    # we pad with `C` dummy logits which are ultimately ignored: see
    # `_mask_tree`.
    output_if_chance_node = base.RecurrentFnOutput(
        prior_logits=jnp.concatenate([
            chance_output.action_logits,
            jnp.full([batch_size, num_chance_outcomes], fill_value=-jnp.inf)
            ], axis=-1),
        value=chance_output.value,
        reward=chance_output.reward,
        discount=chance_output.discount)

    new_state = base.StochasticRecurrentState(
        state_embedding=state_embedding,
        afterstate_embedding=afterstate_embedding,
        is_decision_node=jnp.logical_not(state.is_decision_node))

    def _broadcast_where(decision_leaf, chance_leaf):
      extra_dims = [1] * (len(decision_leaf.shape) - 1)
      expanded_is_decision = jnp.reshape(state.is_decision_node,
                                         [-1] + extra_dims)
      return jnp.where(
          # ensure state.is_decision node has appropriate shape.
          expanded_is_decision,
          decision_leaf, chance_leaf)

    output = jax.tree_map(_broadcast_where,
                          output_if_decision_node,
                          output_if_chance_node)
    return output, new_state

  return stochastic_recurrent_fn


def _mask_tree(tree: search.Tree, num_actions: int, mode: str) -> search.Tree:
  """Masks out parts of the tree based upon node type.

  "Actions" in our tree can either be action or chance values: A' = A + C. This
  utility function masks the parts of the tree containing dimensions of shape
  A' to be either A or C depending upon `mode`.

  Args:
    tree: The tree to be masked.
    num_actions: The number of environment actions A.
    mode: Either "decision" or "chance".

  Returns:
    An appropriately masked tree.
  """

  def _take_slice(x):
    if mode == 'decision':
      return x[..., :num_actions]
    elif mode == 'chance':
      return x[..., num_actions:]
    else:
      raise ValueError(f'Unknown mode: {mode}.')

  return tree.replace(
      children_index=_take_slice(tree.children_index),
      children_prior_logits=_take_slice(tree.children_prior_logits),
      children_visits=_take_slice(tree.children_visits),
      children_rewards=_take_slice(tree.children_rewards),
      children_discounts=_take_slice(tree.children_discounts),
      children_values=_take_slice(tree.children_values),
      root_invalid_actions=_take_slice(tree.root_invalid_actions))


def _make_stochastic_action_selection_fn(
    decision_node_selection_fn: base.InteriorActionSelectionFn,
    num_actions: int,
) -> base.InteriorActionSelectionFn:
  """Make Stochastic Action Selection Fn."""

  # NOTE: trees are unbatched here.

  def _chance_node_selection_fn(
      tree: search.Tree,
      node_index: chex.Array,
  ) -> chex.Array:
    num_chance = tree.children_visits[node_index]
    chance_logits = tree.children_prior_logits[node_index]
    prob_chance = jax.nn.softmax(chance_logits)
    argmax_chance = jnp.argmax(prob_chance / (num_chance + 1), axis=-1)
    return argmax_chance

  def _action_selection_fn(key: chex.PRNGKey, tree: search.Tree,
                           node_index: chex.Array,
                           depth: chex.Array) -> chex.Array:
    is_decision = tree.embeddings.is_decision_node[node_index]
    chance_selection = _chance_node_selection_fn(
        tree=_mask_tree(tree, num_actions, 'chance'),
        node_index=node_index) + num_actions
    decision_selection = decision_node_selection_fn(
        key, _mask_tree(tree, num_actions, 'decision'), node_index, depth)
    return jax.lax.cond(is_decision, lambda: decision_selection,
                        lambda: chance_selection)

  return _action_selection_fn

