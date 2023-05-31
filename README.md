# Mctx: MCTS-in-JAX

Mctx is a library with a [JAX](https://github.com/google/jax)-native
implementation of Monte Carlo tree search (MCTS) algorithms such as
[AlphaZero](https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go),
[MuZero](https://deepmind.com/blog/article/muzero-mastering-go-chess-shogi-and-atari-without-rules), and
[Gumbel MuZero](https://openreview.net/forum?id=bERaNdoegnO). For computation
speed up, the implementation fully supports JIT-compilation. Search algorithms
in Mctx are defined for and operate on batches of inputs, in parallel. This
allows to make the most of the accelerators and enables the algorithms to work
with large learned environment models parameterized by deep neural networks.

## Installation

You can install the latest released version of Mctx from PyPI via:

```sh
pip install mctx
```

or you can install the latest development version from GitHub:

```sh
pip install git+https://github.com/deepmind/mctx.git
```

## Motivation

Learning and search have been important topics since the early days of AI
research. In the [words of Rich Sutton](http://www.incompleteideas.net/IncIdeas/BitterLesson.html):

> One thing that should be learned [...] is the great power of general purpose
> methods, of methods that continue to scale with increased computation even as
> the available computation becomes very great. The two methods that seem to
> scale arbitrarily in this way are *search* and *learning*.

Recently, search algorithms have been successfully combined with learned models
parameterized by deep neural networks, resulting in some of the most powerful
and general reinforcement learning algorithms to date (e.g. MuZero).
However, using search algorithms in combination with deep neural networks
requires efficient implementations, typically written in fast compiled
languages; this can come at the expense of usability and hackability,
especially for researchers that are not familiar with C++. In turn, this limits
adoption and further research on this critical topic.

Through this library, we hope to help researchers everywhere to contribute to
such an exciting area of research. We provide JAX-native implementations of core
search algorithms such as MCTS, that we believe strike a good balance between
performance and usability for researchers that want to investigate search-based
algorithms in Python. The search methods provided by Mctx are
heavily configurable to allow researchers to explore a variety of ideas in
this space, and contribute to the next generation of search based agents.

## Search in Reinforcement Learning

In Reinforcement Learning the *agent* must learn to interact with the
*environment* in order to maximize a scalar *reward* signal. On each step the
agent must select an action and receives in exchange an observation and a
reward. We may call whatever mechanism the agent uses to select the action the
agent's *policy*.

Classically, policies are parameterized directly by a function approximator (as
in REINFORCE), or policies are inferred by inspecting a set of learned estimates
of the value of each action (as in Q-learning). Alternatively, search allows to
select actions by constructing on the fly, in each state, a policy or a value
function local to the current state, by *searching* using a learned *model* of
the environment.

Exhaustive search over all possible future courses of actions is computationally
prohibitive in any non trivial environment, hence we need search algorithms
that can make the best use of a finite computational budget. Typically priors
are needed to guide which nodes in the search tree to expand (to reduce the
*breadth* of the tree that we construct), and value functions are used to
estimate the value of incomplete paths in the tree that don't reach an episode
termination (to reduce the *depth* of the search tree).

## Quickstart

Mctx provides a low-level generic `search` function and high-level concrete
policies: `muzero_policy` and `gumbel_muzero_policy`.

The user needs to provide several learned components to specify the
representation, dynamics and prediction used by [MuZero](https://deepmind.com/blog/article/muzero-mastering-go-chess-shogi-and-atari-without-rules).
In the context of the Mctx library, the representation of the *root* state is
specified by a `RootFnOutput`. The `RootFnOutput` contains the `prior_logits`
from a policy network, the estimated `value` of the root state, and any
`embedding` suitable to represent the root state for the environment model.

The dynamics environment model needs to be specified by a `recurrent_fn`.
A `recurrent_fn(params, rng_key, action, embedding)` call takes an `action` and
a state `embedding`. The call should return a tuple `(recurrent_fn_output,
new_embedding)` with a `RecurrentFnOutput` and the embedding of the next state.
The `RecurrentFnOutput` contains the `reward` and `discount` for the transition,
and `prior_logits` and `value` for the new state.

In [`examples/visualization_demo.py`](https://github.com/deepmind/mctx/blob/main/examples/visualization_demo.py), you can
see calls to a policy:

```python
policy_output = mctx.gumbel_muzero_policy(params, rng_key, root, recurrent_fn,
                                          num_simulations=32)
```

The `policy_output.action` contains the action proposed by the search. That
action can be passed to the environment. To improve the policy, the
`policy_output.action_weights` contain targets usable to train the policy
probabilities.

We recommend to use the `gumbel_muzero_policy`.
[Gumbel MuZero](https://openreview.net/forum?id=bERaNdoegnO) guarantees a policy
improvement if the action values are correctly evaluated. The policy improvement
is demonstrated in
[`examples/policy_improvement_demo.py`](https://github.com/deepmind/mctx/blob/main/examples/policy_improvement_demo.py).

### Example projects
The following projects demonstrate the Mctx usage:

- [Basic Learning Demo with Mctx](https://github.com/kenjyoung/mctx_learning_demo) —
  AlphaZero on random mazes.
- [a0-jax](https://github.com/NTT123/a0-jax) — AlphaZero on Connect Four,
  Gomoku, and Go.
- [muax](https://github.com/bwfbowen/muax) — MuZero on gym-style environments
(CartPole, LunarLander).
- [Classic MCTS](https://github.com/Carbon225/mctx-classic) — A simple example on Connect Four.

Tell us about your project.

## Citing Mctx

This is not an officially supported Google product. Mctx is part of the
[DeepMind JAX Ecosystem]; to cite Mctx, please use the [DeepMind JAX Ecosystem
citation].

[DeepMind JAX Ecosystem]: https://deepmind.com/blog/article/using-jax-to-accelerate-our-research "DeepMind JAX Ecosystem"
[DeepMind JAX Ecosystem citation]: https://github.com/deepmind/jax/blob/main/deepmind2020jax.txt "Citation"
