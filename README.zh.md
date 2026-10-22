# Mctx：基于 JAX 的蒙特卡洛树搜索（MCTS-in-JAX）

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

Mctx 是一个采用原生 [JAX](https://github.com/jax-ml/jax) 实现蒙特卡洛树搜索（MCTS）算法的库，支持包括 [AlphaZero](https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go)、[MuZero](https://deepmind.com/blog/article/muzero-mastering-go-chess-shogi-and-atari-without-rules) 和 [Gumbel MuZero](https://openreview.net/forum?id=bERaNdoegnO) 等经典算法。为加速计算，该实现完全支持 JIT 即时编译。Mctx 中的搜索算法专为批量（batch）输入设计并支持并行执行。这使得算法能够充分发挥硬件加速器（GPU/TPU）的算力，并能够与由深度神经网络参数化的大型已学习环境模型无缝协同工作。

## 安装

您可以通过 PyPI 安装 Mctx 的最新正式版本：

```sh
pip install mctx
```

或者直接从 GitHub 安装最新的开发版本：

```sh
pip install git+https://github.com/google-deepmind/mctx.git
```

## 动机

自人工智能研究早期以来，学习（Learning）与搜索（Search）一直是至关重要的核心课题。正如 Rich Sutton 在[《苦涩的教训》（The Bitter Lesson）](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)中所言：

> 应该吸取的教训是……通用方法所蕴含的巨大威力，即那些随着算力成倍增加而性能持续扩展的方法。能够以这种方式实现近乎无限扩展的两种方法正是：*搜索（Search）*与*学习（Learning）*。

近年来，搜索算法与深度神经网络参数化的学习模型成功结合，诞生了迄今为止最强大、最通用的强化学习算法（例如 MuZero）。然而，将搜索算法与深度神经网络结合使用需要极其高效的底层实现，通常需要用运行速度较快的编译型语言编写；这往往会牺牲易用性与灵活二次开发能力（hackability），对于不熟悉 C++ 的研究人员而言尤为困难。这反过来制约了这一关键方向的普及与深入研究。

通过本库，我们希望助力全球各地的研究人员投身于这一令人振奋的研究领域。我们提供了基于原生 JAX 的核心搜索算法（如 MCTS）实现，力求为希望在 Python 中探索基于搜索的算法的研究人员在性能与易用性之间取得良好平衡。Mctx 提供的搜索方法具备高度可配置性，方便研究人员在这一领域探索多种前沿构想，并为下一代基于搜索的智能体（Agent）做出贡献。

## 强化学习中的搜索

在强化学习（Reinforcement Learning）中，*智能体（Agent）* 必须通过与 *环境（Environment）* 交互来最大化标量 *奖励（Reward）* 信号。在每一步中，智能体必须选择一个动作，并随之获取一份观测（Observation）与奖励。我们将智能体用于选择动作的任何机制统称为智能体的 *策略（Policy）*。

经典方法中，策略要么直接由函数拟合器参数化（如 REINFORCE），要么通过评估各动作价值的学习估计值推导得出（如 Q-learning）。另一种替代方案是，搜索允许智能体在每个状态下，借助已学习的环境 *模型（Model）* 进行 *搜索*，动态即时构建出针对当前状态局部的策略或价值函数，从而实现动作选择。

在任何非平凡的环境中，对所有未来可能行动路径进行穷举搜索在计算上都是不可承受的，因此我们需要能够在有限计算预算内实现效益最大化的搜索算法。通常，需要先验（Priors）来指导扩展搜索树中的哪些节点（以缩小构建的搜索树 *宽度*），并利用价值函数（Value Functions）来估计搜索树中未到达回合终止点的未完整路径的价值（以减少搜索树的 *深度*）。

## 快速上手

Mctx 提供了一个底层的通用 `search` 函数，以及高层具体的即用策略：`muzero_policy` 与 `gumbel_muzero_policy`。

用户需要提供若干已学习的神经网络组件，以指定 [MuZero](https://deepmind.com/blog/article/muzero-mastering-go-chess-shogi-and-atari-without-rules) 所需的表征（Representation）、动力学（Dynamics）和预测（Prediction）模型。在 Mctx 库中，*根（Root）* 状态的表征由 `RootFnOutput` 指定。`RootFnOutput` 包含来自策略网络的 `prior_logits`、根状态的估计 `value`（价值），以及适合向环境模型表征根状态的任何 `embedding`（嵌入向量）。

动力学环境模型需要通过 `recurrent_fn` 指定。调用 `recurrent_fn(params, rng_key, action, embedding)` 时传入一个 `action` 和状态 `embedding`。该调用应返回一个元组 `(recurrent_fn_output, new_embedding)`，包含一个 `RecurrentFnOutput` 及下一状态的嵌入向量。`RecurrentFnOutput` 包含该转移过程的 `reward`（奖励）和 `discount`（折扣因子），以及新状态的 `prior_logits` 与 `value`。

在 [`examples/visualization_demo.py`](https://github.com/google-deepmind/mctx/blob/main/examples/visualization_demo.py) 中，您可以查看调用策略的示例：

```python
policy_output = mctx.gumbel_muzero_policy(params, rng_key, root, recurrent_fn,
                                          num_simulations=32)
```

`policy_output.action` 包含搜索推荐执行的动作。该动作可以直接传递给环境。为了改进策略，`policy_output.action_weights` 包含了可用于训练策略概率的目标分布。

我们推荐使用 `gumbel_muzero_policy`。[Gumbel MuZero](https://openreview.net/forum?id=bERaNdoegnO) 保证在动作价值得到准确评估的前提下实现策略改进（Policy Improvement）。在 [`examples/policy_improvement_demo.py`](https://github.com/google-deepmind/mctx/blob/main/examples/policy_improvement_demo.py) 中演示了策略改进的具体效果。

### 示例项目
以下项目展示了 Mctx 的实际应用：

- [Pgx](https://github.com/sotetsuk/pgx) — 包含 20 多个向量化 JAX 环境的集合，涵盖西洋双陆棋、国际象棋、将棋、围棋以及 AlphaZero 示例。
- [Basic Learning Demo with Mctx](https://github.com/kenjyoung/mctx_learning_demo) — 在随机迷宫上运行 AlphaZero。
- [a0-jax](https://github.com/NTT123/a0-jax) — 在四子棋（Connect Four）、五子棋（Gomoku）与围棋上运行 AlphaZero。
- [muax](https://github.com/bwfbowen/muax) — 在 Gym 风格环境（CartPole、LunarLander）上运行 MuZero。
- [Classic MCTS](https://github.com/Carbon225/mctx-classic) — 在四子棋上的简易示例。
- [mctx-az](https://github.com/lowrollr/mctx-az) — 带有 AlphaZero 子树持久化机制的 Mctx。

欢迎向我们分享您的项目。

## 引用 Mctx

本仓库属于 DeepMind JAX 生态体系的一部分。如需引用 Mctx，请使用以下 BibTeX 引用条目：

```bibtex
@software{deepmind2020jax,
  title = {The {D}eep{M}ind {JAX} {E}cosystem},
  author = {DeepMind and Babuschkin, Igor and Baumli, Kate and Bell, Alison and Bhupatiraju, Surya and Bruce, Jake and Buchlovsky, Peter and Budden, David and Cai, Trevor and Clark, Aidan and Danihelka, Ivo and Dedieu, Antoine and Fantacci, Claudio and Godwin, Jonathan and Jones, Chris and Hemsley, Ross and Hennigan, Tom and Hessel, Matteo and Hou, Shaobo and Kapturowski, Steven and Keck, Thomas and Kemaev, Iurii and King, Michael and Kunesch, Markus and Martens, Lena and Merzic, Hamza and Mikulik, Vladimir and Norman, Tamara and Papamakarios, George and Quan, John and Ring, Roman and Ruiz, Francisco and Sanchez, Alvaro and Sartran, Laurent and Schneider, Rosalia and Sezener, Eren and Spencer, Stephen and Srinivasan, Srivatsan and Stanojevi\'{c}, Milo\v{s} and Stokowiec, Wojciech and Wang, Luyu and Zhou, Guangyao and Viola, Fabio},
  url = {http://github.com/deepmind},
  year = {2020},
}
```

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月22日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
