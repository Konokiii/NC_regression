import os

os.environ['MUJOCO_GL'] = 'egl'

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Dict, List, Optional, Callable, Union
import copy
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import random
import uuid
import warnings

from dm_env import specs
import gym
import numpy as np
from tqdm.auto import trange
import wandb

from vd4rl_buffer import EfficientReplayBuffer, load_offline_dataset_into_buffer
from vd4rl_utils import make

TensorBatch = List[torch.Tensor]

warnings.filterwarnings("ignore", category=DeprecationWarning)


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "walker_walk"
    dataset: str = "expert"
    data_size: Union[str, int] = "all"
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(1e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(3e5)  # Max time steps to run environment
    vd4rl_path: str = "./vd4rl"
    # Trainer
    buffer_size: int = 1_000_000  # Replay buffer size
    actor_lr: float = 3e-4
    encoder_lr: float = 3e-4
    actor_wd: float = 1e-4
    encoder_wd: float = 1e-4
    optimizer: str = "adam"
    hidden_layers: int = 3
    hidden_dim: int = 256
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    # Wandb logging
    project: str = "NC_new"
    group: str = "vd4rl"
    name: str = "test"

    # def __post_init__(self):
    #     self.name = f"{self.name}-{self.task_name}-{self.dataset_name}-{str(uuid.uuid4())[:8]}"
    #     if self.checkpoints_path is not None:
    #         self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def save_model(save_name, step, models, config):
    save_folder = os.path.join(config.vd4rl_path, "models", config.env, save_name)
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"checkpoint_{step}.pt")
    checkpoint = {"config": asdict(config)}
    for k, v in models.items():
        checkpoint[k] = v.cpu().state_dict()
        v.to(config.device)
    torch.save(checkpoint, save_path)


def set_seed(
        seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


@torch.no_grad()
def eval_actor(
        env: gym.Env,
        actor: nn.Module,
        encoder: nn.Module,
        device: str,
        n_episodes: int,
        seed: int
) -> np.ndarray:
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset().observation[None, ...], False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, encoder, device)
            env_out = env.step(action)
            state, reward, done = env_out.observation[None, ...], env_out.reward, env_out.last()
            episode_reward += reward
        episode_rewards.append(episode_reward)
    return np.asarray(episode_rewards)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class RandomShiftsAug(nn.Module):
    def __init__(self, pad=4):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(
            self, state_dim: int, action_dim: int, max_action: float,
            hidden_dim: int = 256,
    ):
        super(Actor, self).__init__()

        self.trunk = nn.Sequential(nn.Linear(state_dim, 50),
                                   nn.LayerNorm(50), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(50, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True))
        self.W = nn.Linear(hidden_dim, action_dim)

        self.apply(weight_init)

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        h = self.trunk(state)
        mu = self.W(self.policy(h))

        return mu

    @torch.no_grad()
    def act(self, state: np.ndarray, encoder: nn.Module, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state, device=device, dtype=torch.float32)
        state = encoder(state)
        action = torch.tanh(self(state))
        return action.cpu().data.numpy().flatten()


class BCAgent:
    def __init__(self,
                 aug,
                 encoder,
                 encoder_optimizer,
                 actor,
                 actor_optimizer,
                 device
                 ):
        self.device = device

        self.encoder = encoder
        self.actor = actor

        self.encoder_opt = encoder_optimizer
        self.actor_opt = actor_optimizer

        self.aug = aug

        self._train()

    def _train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)

    def update(self, batch):
        self._train(True)
        metrics = dict()
        obs, action, reward, _, _ = batch

        # augment & encode
        obs = self.aug(obs)
        obs = self.encoder(obs)

        # update actor
        action_prediction = self.actor(obs)
        actor_loss = F.mse_loss(action_prediction, action)

        self.encoder_opt.zero_grad(set_to_none=True)
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        self.encoder_opt.step()

        metrics['actor_bc_loss'] = actor_loss.item()

        self._train(False)

        return metrics


def run_vd4rl_bc(config: TrainConfig):
    env = make(config.env, 3, 2, config.seed)

    data_specs = (env.observation_spec(),
                  env.action_spec(),
                  specs.Array((1,), np.float32, 'reward'),
                  specs.Array((1,), np.float32, 'discount'))
    replay_buffer = EfficientReplayBuffer(
        config.buffer_size,
        config.batch_size,
        3,
        config.discount,
        3,
        data_specs=data_specs,
        # sarsa=True
    )

    data_path = f"{config.vd4rl_path}/datasets/{config.env}/{config.dataset}/84px"
    print("Trying to load data from:", data_path)
    load_offline_dataset_into_buffer(
        Path(data_path),
        replay_buffer,
        3,
        config.buffer_size,
    )

    max_action = 1.0

    # Set seeds
    seed = config.seed
    set_seed(seed)

    action_dim = env.action_spec().shape[0]

    aug = RandomShiftsAug(pad=4)

    optimizer = {'sgd': torch.optim.SGD,
                 'adam': torch.optim.Adam}[config.optimizer]

    encoder = Encoder((9, 84, 84)).to(config.device)
    encoder_optimizer = optimizer(encoder.parameters(), lr=config.encoder_lr, weight_decay=config.encoder_wd)

    actor = Actor(encoder.repr_dim, action_dim, max_action, config.hidden_dim).to(config.device)
    actor_optimizer = optimizer(actor.parameters(), lr=config.actor_lr, weight_decay=config.actor_wd)

    save_model(save_name=config.name,
               step=0,
               models={'encoder': encoder, 'actor': actor},
               config=config)

    kwargs = {
        "aug": aug,
        "encoder": encoder,
        "encoder_optimizer": encoder_optimizer,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "device": config.device,
    }

    print("---------------------------------------")
    print(f"Training VD4RL BCAgent, Env: {config.env}, {config.dataset} Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = BCAgent(**kwargs)

    # if config.load_model != "":
    #     policy_file = Path(config.load_model)
    #     trainer.load_state_dict(torch.load(policy_file))
    #     actor = trainer.actor

    wandb_init(asdict(config))

    def to_torch(xs, device):
        return tuple(torch.FloatTensor(x).to(device) for x in xs)

    for t in trange(config.max_timesteps, desc="BCAgent steps"):
        batch = list(to_torch(next(replay_buffer), config.device))
        train_log_dict = trainer.update(batch)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")

            save_model(save_name=config.name,
                       step=t + 1,
                       models={'encoder': encoder, 'actor': actor},
                       config=config)

            eval_scores = eval_actor(
                env,
                actor=actor,
                encoder=encoder,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            normalized_eval_scores = eval_scores / 10
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_scores.mean():.3f} , VD4RL score: {normalized_eval_scores.mean():.3f}"
            )
            print("---------------------------------------")
            wandb.log(train_log_dict, step=t+1)
            wandb.log(
                {
                    "normalized_score": normalized_eval_scores.mean(),
                    "normalized_score_std": normalized_eval_scores.std(),
                },
                step=t+1,
            )


if __name__ == "__main__":
    run_vd4rl_bc(TrainConfig())
