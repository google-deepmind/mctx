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


# The search takes as input an initial set of predictions made directly
# from the raw data. These predictions are made by a `RootFn` with signature:
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


RawData = Any
RootFn = Callable[[Params, chex.PRNGKey, RawData], RootFnOutput]


# Action selection functions specify how to pick nodes to expand in the tree.
NodeIndices = chex.Array
Depth = chex.Array
RootActionSelectionFn = Callable[
    [chex.PRNGKey, tree.Tree, NodeIndices], chex.Array]
InteriorActionSelectionFn = Callable[
    [chex.PRNGKey, tree.Tree, NodeIndices, Depth],
    chex.Array]
QTransform = Callable[[tree.Tree, chex.Array], chex.Array]

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
