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
"""Core types used in mctx."""

from typing import Any, Callable, Generic, TypeVar, Tuple

import chex

from mctx._src import tree


# Parameters are arbitrary nested structures of `chex.Array`.
# A nested structure is either a single object, or a collection (list, tuple,
# dictionary, etc.) of other nested structures.
Params = chex.ArrayTree


# The model used to search is expressed by a `RecurrentFn` function that takes
# `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput` and
# the new state embedding.
@chex.dataclass(frozen=True)
class RecurrentFnOutput:
  """The output of a `RecurrentFn`.

  reward: `[B]` an approximate reward from the state-action transition.
  discount: `[B]` the discount between the `reward` and the `value`.
  prior_logits: `[B, num_actions]` the logits produced by a policy network.
  value: `[B]` an approximate value of the state after the state-action
    transition.
  """
  reward: chex.Array
  discount: chex.Array
  prior_logits: chex.Array
  value: chex.Array


Action = chex.Array
RecurrentState = Any
RecurrentFn = Callable[
    [Params, chex.PRNGKey, Action, RecurrentState],
    Tuple[RecurrentFnOutput, RecurrentState]]


@chex.dataclass(frozen=True)
class RootFnOutput:
  """The output of a representation network.

  prior_logits: `[B, num_actions]` the logits produced by a policy network.
  value: `[B]` an approximate value of the current state.
  embedding: `[B, ...]` the inputs to the next `recurrent_fn` call.
  """
  prior_logits: chex.Array
  value: chex.Array
  embedding: RecurrentState


# Action selection functions specify how to pick nodes to expand in the tree.
NodeIndices = chex.Array
Depth = chex.Array
RootActionSelectionFn = Callable[
    [chex.PRNGKey, tree.Tree, NodeIndices], chex.Array]
InteriorActionSelectionFn = Callable[
    [chex.PRNGKey, tree.Tree, NodeIndices, Depth],
    chex.Array]
QTransform = Callable[[tree.Tree, chex.Array], chex.Array]
# LoopFn has the same interface as jax.lax.fori_loop.
LoopFn = Callable[
    [int, int, Callable[[Any, Any], Any], Tuple[chex.PRNGKey, tree.Tree]],
    Tuple[chex.PRNGKey, tree.Tree]]

T = TypeVar("T")


@chex.dataclass(frozen=True)
class PolicyOutput(Generic[T]):
  """The output of a policy.

  action: `[B]` the proposed action.
  action_weights: `[B, num_actions]` the targets used to train a policy network.
    The action weights sum to one. Usually, the policy network is trained by
    cross-entropy:
    `cross_entropy(labels=stop_gradient(action_weights), logits=prior_logits)`.
  search_tree: `[B, ...]` the search tree of the finished search.
  """
  action: chex.Array
  action_weights: chex.Array
  search_tree: tree.Tree[T]


@chex.dataclass(frozen=True)
class DecisionRecurrentFnOutput:
  """Output of the function for expanding decision nodes.

  Expanding a decision node takes an action and a state embedding and produces
  an afterstate, which represents the state of the environment after an action
  is taken but before the environment has updated its state. Accordingly, there
  is no discount factor or reward for transitioning from state `s` to afterstate
  `sa`.

  Attributes:
    chance_logits: `[B, C]` logits of `C` chance outcomes at the afterstate.
    afterstate_value: `[B]` values of the afterstates `v(sa)`.
  """
  chance_logits: chex.Array  # [B, C]
  afterstate_value: chex.Array  # [B]


@chex.dataclass(frozen=True)
class ChanceRecurrentFnOutput:
  """Output of the function for expanding chance nodes.

  Expanding a chance node takes a chance outcome and an afterstate embedding
  and produces a state, which captures a potentially stochastic environment
  transition. When this transition occurs reward and discounts are produced as
  in a normal transition.

  Attributes:
    action_logits: `[B, A]` logits of different actions from the state.
    value: `[B]` values of the states `v(s)`.
    reward: `[B]` rewards at the states.
    discount: `[B]` discounts at the states.
  """
  action_logits: chex.Array  # [B, A]
  value: chex.Array  # [B]
  reward: chex.Array  # [B]
  discount: chex.Array  # [B]


@chex.dataclass(frozen=True)
class StochasticRecurrentState:
  """Wrapper that enables different treatment of decision and chance nodes.

  In Stochastic MuZero tree nodes can either be decision or chance nodes, these
  nodes are treated differently during expansion, search and backup, and a user
  could also pass differently structured embeddings for each type of node. This
  wrapper enables treating chance and decision nodes differently and supports
  potential differences between chance and decision node structures.

  Attributes:
    state_embedding: `[B ...]` an optionally meaningful state embedding.
    afterstate_embedding: `[B ...]` an optionally meaningful afterstate
      embedding.
    is_decision_node: `[B]` whether the node is a decision or chance node. If it
      is a decision node, `afterstate_embedding` is a dummy value. If it is a
      chance node, `state_embedding` is a dummy value.
  """
  state_embedding: chex.ArrayTree  # [B, ...]
  afterstate_embedding: chex.ArrayTree  # [B, ...]
  is_decision_node: chex.Array  # [B]


RecurrentState = chex.ArrayTree

DecisionRecurrentFn = Callable[[Params, chex.PRNGKey, Action, RecurrentState],
                               Tuple[DecisionRecurrentFnOutput, RecurrentState]]

ChanceRecurrentFn = Callable[[Params, chex.PRNGKey, Action, RecurrentState],
                             Tuple[ChanceRecurrentFnOutput, RecurrentState]]
