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
"""A JAX implementation of batched MCTS."""
import functools
from typing import Any, NamedTuple, Optional, Tuple, TypeVar

import chex
import jax
import jax.numpy as jnp

from mctx._src import action_selection
from mctx._src import base
from mctx._src import tree as tree_lib

Tree = tree_lib.Tree
T = TypeVar("T")


def search(
    params: base.Params,
    rng_key: chex.PRNGKey,
    *,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn,
    num_simulations: int,
    max_depth: Optional[int] = None,
    invalid_actions: Optional[chex.Array] = None,
    extra_data: Any = None,
    loop_fn: base.LoopFn = jax.lax.fori_loop) -> Tree:
  """Performs a full search and returns sampled actions.

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
    root_action_selection_fn: function used to select an action at the root.
    interior_action_selection_fn: function used to select an action during
      simulation.
    num_simulations: the number of simulations.
    max_depth: maximum search tree depth allowed during simulation, defined as
      the number of edges from the root to a leaf node.
    invalid_actions: a mask with invalid actions at the root. In the
      mask, invalid actions have ones, and valid actions have zeros.
      Shape `[B, num_actions]`.
    extra_data: extra data passed to `tree.extra_data`. Shape `[B, ...]`.
    loop_fn: Function used to run the simulations. It may be required to pass
      hk.fori_loop if using this function inside a Haiku module.

  Returns:
    `SearchResults` containing outcomes of the search, e.g. `visit_counts`
    `[B, num_actions]`.
  """
  action_selection_fn = action_selection.switching_action_selection_wrapper(
      root_action_selection_fn=root_action_selection_fn,
      interior_action_selection_fn=interior_action_selection_fn
  )

  # Do simulation, expansion, and backward steps.
  batch_size = root.value.shape[0]
  batch_range = jnp.arange(batch_size)
  if max_depth is None:
    max_depth = num_simulations
  if invalid_actions is None:
    invalid_actions = jnp.zeros_like(root.prior_logits)

  def body_fun(sim, loop_state):
    rng_key, tree = loop_state
    rng_key, simulate_key, expand_key = jax.random.split(rng_key, 3)
    # simulate is vmapped and expects batched rng keys.
    simulate_keys = jax.random.split(simulate_key, batch_size)
    parent_index, action = simulate(
        simulate_keys, tree, action_selection_fn, max_depth)
    # A node first expanded on simulation `i`, will have node index `i`.
    # Node 0 corresponds to the root node.
    next_node_index = tree.children_index[batch_range, parent_index, action]
    next_node_index = jnp.where(next_node_index == Tree.UNVISITED,
                                sim + 1, next_node_index)
    tree = expand(
        # pyrefly: ignore[bad-argument-type]
        params, expand_key, tree, recurrent_fn, parent_index,
        action, next_node_index)
    tree = backward(tree, next_node_index)
    loop_state = rng_key, tree
    return loop_state

  # Allocate all necessary storage.
  tree = instantiate_tree_from_root(root, num_simulations,
                                    root_invalid_actions=invalid_actions,
                                    extra_data=extra_data)
  _, tree = loop_fn(
      0, num_simulations, body_fun, (rng_key, tree))

  return tree


class _SimulationState(NamedTuple):
  """The state for the simulation while loop."""
  rng_key: chex.PRNGKey
  node_index: int
  action: int
  next_node_index: int
  depth: int
  is_continuing: bool


@functools.partial(jax.vmap, in_axes=[0, 0, None, None], out_axes=0)
def simulate(
    rng_key: chex.PRNGKey,
    tree: Tree,
    action_selection_fn: base.InteriorActionSelectionFn,
    max_depth: int) -> Tuple[chex.Array, chex.Array]:
  """Traverses the tree until reaching an unvisited action or `max_depth`.

  Each simulation starts from the root and keeps selecting actions traversing
  the tree until a leaf or `max_depth` is reached.

  Args:
    rng_key: random number generator state, the key is consumed.
    tree: _unbatched_ MCTS tree state.
    action_selection_fn: function used to select an action during simulation.
    max_depth: maximum search tree depth allowed during simulation.

  Returns:
    `(parent_index, action)` tuple, where `parent_index` is the index of the
    node reached at the end of the simulation, and the `action` is the action to
    evaluate from the `parent_index`.
  """
  def cond_fun(state):
    return state.is_continuing

  def body_fun(state):
    # Preparing the next simulation state.
    node_index = state.next_node_index
    rng_key, action_selection_key = jax.random.split(state.rng_key)
    action = action_selection_fn(action_selection_key, tree, node_index,
                                 state.depth)
    # pyrefly: ignore[bad-index]
    next_node_index = tree.children_index[node_index, action]
    # The returned action will be visited.
    depth = state.depth + 1
    is_before_depth_cutoff = depth < max_depth
    is_visited = next_node_index != Tree.UNVISITED
    is_continuing = jnp.logical_and(is_visited, is_before_depth_cutoff)
    return _SimulationState(  # pytype: disable=wrong-arg-types  # jax-types
        rng_key=rng_key,
        node_index=node_index,
        action=action,  # pyrefly: ignore[bad-argument-type]
        next_node_index=next_node_index,  # pyrefly: ignore[bad-argument-type]
        depth=depth,
        is_continuing=is_continuing)  # pyrefly: ignore[bad-argument-type]

  node_index = jnp.array(Tree.ROOT_INDEX, dtype=jnp.int32)
  depth = jnp.zeros((), dtype=tree.children_prior_logits.dtype)
  # pytype: disable=wrong-arg-types  # jnp-type
  initial_state = _SimulationState(
      rng_key=rng_key,
      node_index=tree.NO_PARENT,
      action=tree.NO_PARENT,
      next_node_index=node_index,
      depth=depth,
      is_continuing=jnp.array(True))
  # pytype: enable=wrong-arg-types
  end_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)

  # Returning a node with a selected action.
  # The action can be already visited, if the max_depth is reached.
  return end_state.node_index, end_state.action


def expand(
    params: chex.Array,
    rng_key: chex.PRNGKey,
    tree: Tree[T],
    recurrent_fn: base.RecurrentFn,
    parent_index: chex.Array,
    action: chex.Array,
    next_node_index: chex.Array) -> Tree[T]:
  """Create and evaluate child nodes from given nodes and unvisited actions.

  Args:
    params: params to be forwarded to recurrent function.
    rng_key: random number generator state.
    tree: the MCTS tree state to update.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    parent_index: the index of the parent node, from which the action will be
      expanded. Shape `[B]`.
    action: the action to expand. Shape `[B]`.
    next_node_index: the index of the newly expanded node. This can be the index
      of an existing node, if `max_depth` is reached. Shape `[B]`.

  Returns:
    tree: updated MCTS tree state.
  """
  batch_size = tree_lib.infer_batch_size(tree)
  batch_range = jnp.arange(batch_size)
  chex.assert_shape([parent_index, action, next_node_index], (batch_size,))

  # Retrieve states for nodes to be evaluated.
  embedding = jax.tree.map(
      lambda x: x[batch_range, parent_index], tree.embeddings)

  # Evaluate and create a new node.
  step, embedding = recurrent_fn(params, rng_key, action, embedding)
  chex.assert_shape(step.prior_logits, [batch_size, tree.num_actions])
  chex.assert_shape(step.reward, [batch_size])
  chex.assert_shape(step.discount, [batch_size])
  chex.assert_shape(step.value, [batch_size])
  tree = update_tree_node(
      tree, next_node_index, step.prior_logits, step.value, embedding)

  # Return updated tree topology.
  return tree.replace(  # pyrefly: ignore[missing-attribute]
      children_index=batch_update(
          tree.children_index, next_node_index, parent_index, action),
      children_rewards=batch_update(
          tree.children_rewards, step.reward, parent_index, action),
      children_discounts=batch_update(
          tree.children_discounts, step.discount, parent_index, action),
      parents=batch_update(tree.parents, parent_index, next_node_index),
      action_from_parent=batch_update(
          tree.action_from_parent, action, next_node_index))


class _BackwardState(NamedTuple):
  """The state for the backward while loop.

  The loop carry deliberately contains only O(num_nodes) data. Carrying the
  whole tree (with its [num_nodes, num_actions] children arrays) and updating
  it inside the loop body makes XLA:GPU insert full copies of
  `children_values` and `children_visits` on every loop iteration, because
  the same buffers are both read and scattered into within one iteration.
  With large action spaces this makes each simulation cost
  O(depth * num_nodes * num_actions) in memory traffic, i.e. the search
  becomes super-linear in `num_simulations`. Instead, the path from the leaf
  to the root is recorded here and the tree is updated by a single scatter
  per array after the loop.
  """
  step: chex.Array
  index: chex.Array
  leaf_value: chex.Array
  child_value: chex.Array
  path_parents: chex.Array
  path_actions: chex.Array
  path_node_values: chex.Array
  path_node_visits: chex.Array
  path_children_values: chex.Array
  path_children_visits: chex.Array


@jax.vmap
def backward(
    tree: Tree[T],
    leaf_index: chex.Numeric) -> Tree[T]:
  """Goes up and updates the tree until all nodes reached the root.

  Args:
    tree: the MCTS tree state to update, without the batch size.
    leaf_index: the node index from which to do the backward.

  Returns:
    Updated MCTS tree state.
  """
  num_nodes = tree.node_values.shape[0]
  leaf_index = jnp.asarray(leaf_index, dtype=jnp.int32)

  def cond_fun(state):
    return state.index != Tree.ROOT_INDEX

  def body_fun(state):
    # Here we update the value of our parent, so we start by reversing.
    index = state.index
    parent = tree.parents[index]
    count = tree.node_visits[parent]
    action = tree.action_from_parent[index]
    reward = tree.children_rewards[parent, action]
    leaf_value = reward + tree.children_discounts[parent, action] * (
        state.leaf_value)
    parent_value = (
        tree.node_values[parent] * count + leaf_value) / (count + 1.0)
    step = state.step
    # pyrefly: ignore[bad-argument-type]
    return _BackwardState(
        step=step + 1,
        index=parent,
        leaf_value=leaf_value,
        # The next edge reads the just-updated value of this path node; the
        # original in-loop version reads it from the freshly updated tree.
        child_value=parent_value,
        path_parents=state.path_parents.at[step].set(parent),
        path_actions=state.path_actions.at[step].set(action),
        path_node_values=state.path_node_values.at[step].set(parent_value),
        path_node_visits=state.path_node_visits.at[step].set(count + 1),
        path_children_values=state.path_children_values.at[step].set(
            state.child_value),
        path_children_visits=state.path_children_visits.at[step].set(
            tree.children_visits[parent, action] + 1))

  # Unused path slots keep index `num_nodes`, dropped by the scatters below.
  # pyrefly: ignore[bad-index,bad-argument-type]
  initial_state = _BackwardState(
      step=jnp.zeros((), dtype=jnp.int32),
      index=leaf_index,
      leaf_value=tree.node_values[leaf_index],
      child_value=tree.node_values[leaf_index],
      path_parents=jnp.full([num_nodes], num_nodes, dtype=jnp.int32),
      path_actions=jnp.zeros([num_nodes], dtype=jnp.int32),
      path_node_values=jnp.zeros([num_nodes], dtype=tree.node_values.dtype),
      path_node_visits=jnp.zeros([num_nodes], dtype=tree.node_visits.dtype),
      path_children_values=jnp.zeros(
          [num_nodes], dtype=tree.children_values.dtype),
      path_children_visits=jnp.zeros(
          [num_nodes], dtype=tree.children_visits.dtype))
  end_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)

  parents = end_state.path_parents
  actions = end_state.path_actions
  # pyrefly: ignore[missing-attribute]
  return tree.replace(
      node_values=tree.node_values.at[parents].set(
          end_state.path_node_values, mode="drop"),
      node_visits=tree.node_visits.at[parents].set(
          end_state.path_node_visits, mode="drop"),
      children_values=tree.children_values.at[parents, actions].set(
          end_state.path_children_values, mode="drop"),
      children_visits=tree.children_visits.at[parents, actions].set(
          end_state.path_children_visits, mode="drop"))


# Utility function to set the values of certain indices to prescribed values.
# This is vmapped to operate seamlessly on batches.
def update(x, vals, *indices):
  return x.at[indices].set(vals)


batch_update = jax.vmap(update)


def update_tree_node(
    tree: Tree[T],
    node_index: chex.Array,
    prior_logits: chex.Array,
    value: chex.Array,
    embedding: chex.Array) -> Tree[T]:
  """Updates the tree at node index.

  Args:
    tree: `Tree` to whose node is to be updated.
    node_index: the index of the expanded node. Shape `[B]`.
    prior_logits: the prior logits to fill in for the new node, of shape
      `[B, num_actions]`.
    value: the value to fill in for the new node. Shape `[B]`.
    embedding: the state embeddings for the node. Shape `[B, ...]`.

  Returns:
    The new tree with updated nodes.
  """
  batch_size = tree_lib.infer_batch_size(tree)
  batch_range = jnp.arange(batch_size)
  chex.assert_shape(prior_logits, (batch_size, tree.num_actions))

  # When using max_depth, a leaf can be expanded multiple times.
  # pyrefly: ignore[bad-index]
  new_visit = tree.node_visits[batch_range, node_index] + 1
  updates = dict(  # pylint: disable=use-dict-literal
      children_prior_logits=batch_update(
          tree.children_prior_logits, prior_logits, node_index),
      raw_values=batch_update(
          tree.raw_values, value, node_index),
      node_values=batch_update(
          tree.node_values, value, node_index),
      node_visits=batch_update(
          tree.node_visits, new_visit, node_index),
      embeddings=jax.tree.map(
          lambda t, s: batch_update(t, s, node_index),
          tree.embeddings, embedding))

  return tree.replace(**updates)  # pyrefly: ignore[missing-attribute]


def instantiate_tree_from_root(
    root: base.RootFnOutput,
    num_simulations: int,
    root_invalid_actions: chex.Array,
    extra_data: Any) -> Tree:
  """Initializes tree state at search root."""
  chex.assert_rank(root.prior_logits, 2)
  batch_size, num_actions = root.prior_logits.shape
  chex.assert_shape(root.value, [batch_size])
  num_nodes = num_simulations + 1
  data_dtype = root.value.dtype
  batch_node = (batch_size, num_nodes)
  batch_node_action = (batch_size, num_nodes, num_actions)

  def _zeros(x):
    return jnp.zeros(batch_node + x.shape[1:], dtype=x.dtype)

  # Create a new empty tree state and fill its root.
  tree = Tree(
      node_visits=jnp.zeros(batch_node, dtype=jnp.int32),
      raw_values=jnp.zeros(batch_node, dtype=data_dtype),
      node_values=jnp.zeros(batch_node, dtype=data_dtype),
      parents=jnp.full(batch_node, Tree.NO_PARENT, dtype=jnp.int32),
      action_from_parent=jnp.full(
          batch_node, Tree.NO_PARENT, dtype=jnp.int32),
      children_index=jnp.full(
          batch_node_action, Tree.UNVISITED, dtype=jnp.int32),
      children_prior_logits=jnp.zeros(
          batch_node_action, dtype=root.prior_logits.dtype),
      children_values=jnp.zeros(batch_node_action, dtype=data_dtype),
      children_visits=jnp.zeros(batch_node_action, dtype=jnp.int32),
      children_rewards=jnp.zeros(batch_node_action, dtype=data_dtype),
      children_discounts=jnp.zeros(batch_node_action, dtype=data_dtype),
      embeddings=jax.tree.map(_zeros, root.embedding),
      root_invalid_actions=root_invalid_actions,
      extra_data=extra_data)

  root_index = jnp.full([batch_size], Tree.ROOT_INDEX)
  tree = update_tree_node(
      tree, root_index, root.prior_logits, root.value, root.embedding)
  return tree
