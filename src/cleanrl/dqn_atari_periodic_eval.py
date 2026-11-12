# EECE576 controlled Rainbow-style DQN baseline with periodic evaluation.
#
# Configuration represented by this file:
#   (PER, Dueling, Multi-step) = (0, 0, 0)
#
# The goal is NOT a bit-for-bit reproduction of the 2015 Nature DQN code.
# Instead, this file uses one fixed Rainbow-style backbone so that future
# PER / Dueling / Multi-step variants can be compared fairly while changing
# only those components.
#
# Key Rainbow-style training settings used here:
#   - 84x84 grayscale observations
#   - 4 stacked frames
#   - action repeat 4
#   - reward clipping during training
#   - replay capacity 1,000,000
#   - batch size 32
#   - gamma 0.99
#   - learning update every 4 agent steps
#   - learning starts after 80K ALE frames = 20K agent steps
#   - hard target update every 32K ALE frames = 8K agent steps
#   - Adam learning rate 6.25e-5, eps 1.5e-4
#   - epsilon 1.0 -> 0.01 over 250K ALE frames = 62.5K agent steps
#   - full budget 200M ALE frames = 50M agent steps
#   - periodic evaluation every 1M ALE frames = 250K agent steps
#   - each periodic evaluation lasts 500K ALE frames = 125K agent steps
#
# Evaluation epsilon is kept configurable. The Rainbow paper passage that
# specifies periodic evaluation does not itself state the epsilon, so this
# file defaults to 0.05 to match the DQN-style evaluation already used in
# this repository.

import os
import random
import time
from dataclasses import dataclass

import ale_py
import gymnasium as gym

gym.register_envs(ale_py)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from cleanrl_utils.buffers import ReplayBuffer


FRAME_SKIP = 4


@dataclass
class Args:
    """Experiment and DQN hyperparameters."""

    # General experiment settings
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

    # W&B / TensorBoard
    track: bool = False
    wandb_project_name: str = "EECE576-DQN"
    wandb_entity: str | None = None
    capture_video: bool = False

    # Saving / session management
    save_model: bool = False
    checkpoint_frequency: int = 250_000
    max_hours: float = 0.0

    # Periodic evaluation
    eval_frequency: int = 250_000
    eval_steps: int = 31_250
    eval_epsilon: float = 0.05

    # Environment
    env_id: str = "BreakoutNoFrameskip-v4"
    num_envs: int = 1

    # 200M ALE frames / action-repeat 4 = 50M agent steps
    total_timesteps: int = 12_500_000

    # Rainbow-style common backbone
    learning_rate: float = 6.25e-5
    adam_epsilon: float = 1.5e-4
    buffer_size: int = 250_000
    gamma: float = 0.99
    batch_size: int = 32

    # Hard target-network update:
    # 32K ALE frames / 4 = 8K agent steps
    target_network_frequency: int = 8_000
    tau: float = 1.0

    # Epsilon-greedy because this project does not include Noisy Nets:
    # 250K ALE frames / 4 = 62.5K agent steps
    start_e: float = 1.0
    end_e: float = 0.01
    epsilon_decay_steps: int = 62_500

    # 80K ALE frames / 4 = 20K agent steps
    learning_starts: int = 20_000

    # One gradient update every 4 agent decisions
    train_frequency: int = 4


def make_env(env_id, seed, idx, capture_video, run_name):
    """Training environment using DQN/Rainbow Atari preprocessing."""

    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)

        # Keep this inside the reward-clipping wrapper so episode statistics
        # reflect the underlying game score, while the learner receives
        # clipped rewards.
        env = gym.wrappers.RecordEpisodeStatistics(env)

        # Standard Atari preprocessing used by DQN/Rainbow.
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=FRAME_SKIP)
        env = EpisodicLifeEnv(env)

        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)

        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk


def make_eval_env(env_id, seed):
    """Evaluation environment for raw full-game scores.

    Evaluation keeps the same observation preprocessing and action repeat,
    but it does NOT clip rewards and it does NOT treat a lost life as the end
    of the whole game. This lets eval/mean_return represent raw game score.
    """
    env = gym.make(env_id)

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=FRAME_SKIP)

    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)

    # Outermost wrapper: one recorded evaluation episode = one full game.
    env = gym.wrappers.RecordEpisodeStatistics(env)

    env.action_space.seed(seed)
    return env


class QNetwork(nn.Module):
    """Standard DQN convolutional network; Dueling is OFF."""

    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """Linearly anneal epsilon from start_e to end_e."""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def evaluate_for_steps(
    model,
    env_id,
    seed,
    eval_steps,
    device,
    epsilon,
):
    """Evaluate the frozen policy for a fixed number of agent steps.

    No learning or replay-buffer updates occur here. Returns the raw game
    scores of complete games that finish within the evaluation window.
    """
    env = make_eval_env(env_id, seed)
    obs, _ = env.reset(seed=seed)

    # Dedicated RNG so evaluation does not perturb the training Python RNG.
    eval_rng = random.Random(seed)

    episodic_returns = []
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            for _ in range(eval_steps):
                if eval_rng.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    obs_tensor = torch.as_tensor(
                        np.asarray(obs),
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0)

                    q_values = model(obs_tensor)
                    action = int(torch.argmax(q_values, dim=1).item())

                next_obs, _, terminated, truncated, info = env.step(action)
                obs = next_obs

                if terminated or truncated:
                    if "episode" in info:
                        episodic_return = float(
                            np.asarray(info["episode"]["r"]).item()
                        )
                        episodic_returns.append(episodic_return)

                    obs, _ = env.reset()
    finally:
        env.close()

        if was_training:
            model.train()

    return episodic_returns


if __name__ == "__main__":
    args = tyro.cli(Args)

    assert args.num_envs == 1, (
        "This implementation currently supports exactly one training env."
    )

    run_name = (
        f"{args.env_id}__{args.exp_name}__"
        f"{args.seed}__{int(time.time())}"
    )

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % "\n".join(
            [f"|{key}|{value}|" for key, value in vars(args).items()]
        ),
    )

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and args.cuda
        else "mps"
        if torch.backends.mps.is_available() and args.cuda
        else "cpu"
    )

    print(f"device={device}")

    # Training environment
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.seed + i,
                i,
                args.capture_video,
                run_name,
            )
            for i in range(args.num_envs)
        ],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )

    assert isinstance(
        envs.single_action_space,
        gym.spaces.Discrete,
    ), "Only discrete action spaces are supported."

    # ------------------------------------------------------------------
    # (PER, Dueling, Multi-step) = (0, 0, 0)
    # ------------------------------------------------------------------

    # Dueling = OFF
    q_network = QNetwork(envs).to(device)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # Rainbow-style optimizer settings.
    optimizer = optim.Adam(
        q_network.parameters(),
        lr=args.learning_rate,
        eps=args.adam_epsilon,
    )

    # PER = OFF -> uniform replay.
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )

    start_time = time.time()

    model_dir = f"runs/{run_name}"
    model_path = f"{model_dir}/{args.exp_name}.cleanrl_model"
    best_model_path = (
        f"{model_dir}/{args.exp_name}.best.cleanrl_model"
    )
    best_eval_return = -float("inf")

    obs, _ = envs.reset(seed=args.seed)

    for global_step in range(args.total_timesteps):
        # global_step is zero-based; agent_step is easier to interpret.
        agent_step = global_step + 1

        # Approximate raw ALE frames, because each selected action is
        # repeated for FRAME_SKIP=4 emulator frames.
        ale_frames = agent_step * FRAME_SKIP

        # --------------------------------------------------------------
        # Action selection
        # --------------------------------------------------------------
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.epsilon_decay_steps,
            global_step,
        )

        if random.random() < epsilon:
            actions = np.array(
                [
                    envs.single_action_space.sample()
                    for _ in range(envs.num_envs)
                ]
            )
        else:
            with torch.no_grad():
                q_values = q_network(
                    torch.as_tensor(
                        obs,
                        dtype=torch.float32,
                        device=device,
                    )
                )
                actions = torch.argmax(
                    q_values,
                    dim=1,
                ).cpu().numpy()

        # One agent decision advances ~4 ALE frames.
        (
            next_obs,
            rewards,
            terminations,
            truncations,
            infos,
        ) = envs.step(actions)

        # Training return is useful for monitoring, but the main
        # experimental curve is eval/mean_return.
        if (
            "final_info" in infos
            and "episode" in infos["final_info"]
        ):
            final_info = infos["final_info"]

            for idx in np.flatnonzero(final_info["_episode"]):
                episodic_return = final_info["episode"]["r"][idx]
                episodic_length = final_info["episode"]["l"][idx]

                print(
                    f"agent_step={agent_step}, "
                    f"ale_frames={ale_frames}, "
                    f"episodic_return={episodic_return}"
                )

                writer.add_scalar(
                    "charts/episodic_return",
                    episodic_return,
                    ale_frames,
                )
                writer.add_scalar(
                    "charts/episodic_length",
                    episodic_length,
                    ale_frames,
                )

        # --------------------------------------------------------------
        # Uniform replay storage
        # --------------------------------------------------------------
        real_next_obs = next_obs.copy()

        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_obs"][idx]

        rb.add(
            obs,
            real_next_obs,
            actions,
            rewards,
            terminations,
            infos,
        )

        obs = next_obs

        # --------------------------------------------------------------
        # Plain 1-step DQN update
        # Multi-step = OFF
        # --------------------------------------------------------------
        if agent_step > args.learning_starts:
            if agent_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)

                with torch.no_grad():
                    target_max, _ = target_network(
                        data.next_observations
                    ).max(dim=1)

                    # One-step TD target:
                    # r_t + gamma * max_a Q_target(s_{t+1}, a)
                    td_target = (
                        data.rewards.flatten()
                        + args.gamma
                        * target_max
                        * (1 - data.dones.flatten())
                    )

                old_val = q_network(
                    data.observations
                ).gather(
                    1,
                    data.actions,
                ).squeeze()

                loss = F.mse_loss(
                    td_target,
                    old_val,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if agent_step % 100 == 0:
                    writer.add_scalar(
                        "losses/td_loss",
                        loss.item(),
                        ale_frames,
                    )
                    writer.add_scalar(
                        "losses/q_values",
                        old_val.mean().item(),
                        ale_frames,
                    )

                    sps = int(
                        agent_step / (time.time() - start_time)
                    )

                    writer.add_scalar(
                        "charts/SPS",
                        sps,
                        ale_frames,
                    )

                    if agent_step % 10_000 == 0:
                        print("SPS:", sps)

            # Hard target-network update every 8K agent steps
            # after learning has started.
            steps_since_learning_start = (
                agent_step - args.learning_starts
            )

            if (
                steps_since_learning_start > 0
                and steps_since_learning_start
                % args.target_network_frequency
                == 0
            ):
                for target_param, online_param in zip(
                    target_network.parameters(),
                    q_network.parameters(),
                ):
                    target_param.data.copy_(
                        args.tau * online_param.data
                        + (1.0 - args.tau)
                        * target_param.data
                    )

        # --------------------------------------------------------------
        # Periodic evaluation
        # --------------------------------------------------------------
        if (
            args.eval_frequency > 0
            and agent_step % args.eval_frequency == 0
        ):
            eval_returns = evaluate_for_steps(
                model=q_network,
                env_id=args.env_id,
                seed=args.seed + 10_000,
                eval_steps=args.eval_steps,
                device=device,
                epsilon=args.eval_epsilon,
            )

            if eval_returns:
                mean_eval_return = float(
                    np.mean(eval_returns)
                )
                std_eval_return = float(
                    np.std(eval_returns)
                )

                print(
                    f"[EVAL] agent_step={agent_step}, "
                    f"ale_frames={ale_frames}, "
                    f"episodes={len(eval_returns)}, "
                    f"mean_return={mean_eval_return:.3f}, "
                    f"std_return={std_eval_return:.3f}"
                )

                writer.add_scalar(
                    "eval/mean_return",
                    mean_eval_return,
                    ale_frames,
                )
                writer.add_scalar(
                    "eval/std_return",
                    std_eval_return,
                    ale_frames,
                )
                writer.add_scalar(
                    "eval/episodes_completed",
                    len(eval_returns),
                    ale_frames,
                )
                writer.flush()

                if (
                    args.save_model
                    and mean_eval_return > best_eval_return
                ):
                    best_eval_return = mean_eval_return

                    torch.save(
                        q_network.state_dict(),
                        best_model_path + ".tmp",
                    )
                    os.replace(
                        best_model_path + ".tmp",
                        best_model_path,
                    )

                    print(
                        "[EVAL] new best evaluated model saved to "
                        f"{best_model_path}"
                    )
            else:
                print(
                    f"[EVAL] agent_step={agent_step}: "
                    "no complete games finished in the "
                    "evaluation window"
                )

        # --------------------------------------------------------------
        # Periodic checkpoint
        # --------------------------------------------------------------
        if (
            args.save_model
            and agent_step % args.checkpoint_frequency == 0
        ):
            torch.save(
                q_network.state_dict(),
                model_path + ".tmp",
            )
            os.replace(
                model_path + ".tmp",
                model_path,
            )

        # --------------------------------------------------------------
        # Optional session time limit
        # --------------------------------------------------------------
        if (
            args.max_hours > 0
            and time.time() - start_time
            > args.max_hours * 3600
        ):
            print(
                f"reached max_hours={args.max_hours} "
                f"at agent_step={agent_step}, "
                f"ale_frames={ale_frames}; stopping"
            )
            break

    if args.save_model:
        torch.save(
            q_network.state_dict(),
            model_path,
        )
        print(f"latest model saved to {model_path}")

        if os.path.exists(best_model_path):
            print(
                f"best evaluated model saved to "
                f"{best_model_path}"
            )

    envs.close()
    writer.close()