# inspiration:
# 1. https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py  # noqa
# 2. https://github.com/karpathy/minGPT
import os
import random
import uuid
import json
import pickle
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

import d4rl  # noqa
import gym
import numpy as np
# import pyrallis
import torch
import torch.nn as nn
import wandb
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm, trange  # noqa

from main.nrc_utils import compute_metrics


@dataclass
class TrainConfig:
    # wandb params
    project: str = "nrc4rl"
    group: str = "DT"
    name: str = "DT"
    task_id: int = 0
    # model params
    embedding_dim: int = 128
    num_layers: int = 3
    num_heads: int = 1
    seq_len: int = 20
    episode_len: int = 1000
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    max_action: float = 1.0
    # training params
    env: str = "hopper"
    dataset: str = 'medium'
    # env_name: str = "hopper-medium-v2"
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 3e-4
    clip_grad: Optional[float] = 0.25
    batch_size: int = 64
    update_steps: int = 2_000_000
    warmup_steps: int = 10_000
    reward_scale: float = 0.001
    num_workers: int = 4
    # evaluation params
    target_returns: Tuple[float, ...] = (3600.0, 1800.0)
    eval_episodes: int = 10
    eval_every: int = 10_000
    # general params
    checkpoints_path: Optional[str] = None
    deterministic_torch: bool = False
    train_seed: int = 10
    eval_seed: int = 42
    device: str = "cpu"
    # NRC relate_d
    num_NRC_batches: int = 50
    action_epsilon: float = 5e-8
    project_folder: str = "./"

    # def __post_init__(self):
    #     self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
    #     if self.checkpoints_path is not None:
    #         self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


# general utils
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
        settings=wandb.Settings(start_method="fork")
    )
    wandb.run.save()


def wrap_env(
        env: gym.Env,
        state_mean: Union[np.ndarray, float] = 0.0,
        state_std: Union[np.ndarray, float] = 1.0,
        reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


# some utils functionalities specific for Decision Transformer
def pad_along_axis(
        arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


def load_d4rl_trajectories(
        env_name: str, gamma: float = 1.0, max_action: float = 1.0, action_epsilon: float = 1e-7
) -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, Any]]:
    dataset = gym.make(env_name).get_dataset()
    # Modified: Clip the actions to have absolute value smaller than 1 so no inf appear in arctanh().
    dataset['actions'] = np.clip(dataset['actions'], a_min=-max_action + action_epsilon,
                                 a_max=max_action - action_epsilon)
    traj, traj_len = [], []

    data_ = defaultdict(list)
    for i in trange(dataset["rewards"].shape[0], desc="Processing trajectories"):
        data_["observations"].append(dataset["observations"][i])
        data_["actions"].append(dataset["actions"][i])
        data_["rewards"].append(dataset["rewards"][i])

        if dataset["terminals"][i] or dataset["timeouts"][i]:
            episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
            # return-to-go if gamma=1.0, just discounted returns else
            episode_data["returns"] = discounted_cumsum(
                episode_data["rewards"], gamma=gamma
            )
            traj.append(episode_data)
            traj_len.append(episode_data["actions"].shape[0])
            # reset trajectory buffer
            data_ = defaultdict(list)

    # needed for normalization, weighted sampling, other stats can be added also
    info = {
        "obs_mean": dataset["observations"].mean(0, keepdims=True),
        "obs_std": dataset["observations"].std(0, keepdims=True) + 1e-6,
        "traj_lens": np.array(traj_len),
    }

    # traj: List of dictionary, each of which is one traj
    return traj, info


class SequenceDataset(IterableDataset):
    def __init__(self,
                 env_name: str,
                 seq_len: int = 10,
                 reward_scale: float = 1.0,
                 max_action: float = 1.0,
                 action_epsilon: float = 1e-7, ):
        self.env_name = env_name
        self.max_action = max_action
        self.action_epsilon = action_epsilon

        self.dataset, info = load_d4rl_trajectories(env_name, gamma=1.0, max_action=max_action,
                                                    action_epsilon=action_epsilon)
        self.reward_scale = reward_scale
        self.seq_len = seq_len

        self.state_mean = info["obs_mean"]
        self.state_std = info["obs_std"]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L116 # noqa
        self.sample_prob = info["traj_lens"] / info["traj_lens"].sum()

    def __prepare_sample(self, traj_idx, start_idx):
        traj = self.dataset[traj_idx]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L128 # noqa
        states = traj["observations"][start_idx: start_idx + self.seq_len]
        actions = traj["actions"][start_idx: start_idx + self.seq_len]
        returns = traj["returns"][start_idx: start_idx + self.seq_len]
        time_steps = np.arange(start_idx, start_idx + self.seq_len)

        states = (states - self.state_mean) / self.state_std
        returns = returns * self.reward_scale
        # pad up to seq_len if needed
        mask = np.hstack(
            [np.ones(states.shape[0]), np.zeros(self.seq_len - states.shape[0])]
        )
        if states.shape[0] < self.seq_len:
            states = pad_along_axis(states, pad_to=self.seq_len)
            actions = pad_along_axis(actions, pad_to=self.seq_len)
            returns = pad_along_axis(returns, pad_to=self.seq_len)

        return states, actions, returns, time_steps, mask

    def __iter__(self):
        while True:
            traj_idx = np.random.choice(len(self.dataset), p=self.sample_prob)
            start_idx = random.randint(0, self.dataset[traj_idx]["rewards"].shape[0] - 1)
            yield self.__prepare_sample(traj_idx, start_idx)

    # Modified: compute NRC theory values
    def get_theoretic_values(self):
        d4rl_dataset = gym.make(self.env_name).get_dataset()
        Y = d4rl_dataset["actions"].T
        y_dim, M = Y.shape[0], Y.shape[1]
        Y_mean = Y.mean(axis=1, keepdims=True)
        Y_centered = Y - Y_mean
        Sigma = Y_centered @ Y_centered.T / M

        eig_vals, eig_vecs = np.linalg.eigh(Sigma)
        sqrt_eig_vals = np.sqrt(eig_vals)
        Sigma_sqrt = eig_vecs.dot(np.diag(sqrt_eig_vals)).dot(np.linalg.inv(eig_vecs))

        original_theory_values = {'Sigma': Sigma,
                                  'Sigma_sqrt': Sigma_sqrt,
                                  'Y_mean': Y_mean.squeeze(axis=-1),
                                  'min_eigval': eig_vals[0],
                                  'max_eigval': eig_vals[-1]}

        Y = np.clip(Y, a_min=-self.max_action + self.action_epsilon, a_max=self.max_action - self.action_epsilon)
        Y = np.arctanh(Y)
        Y_mean = Y.mean(axis=1, keepdims=True)
        Y_centered = Y - Y_mean
        Sigma = Y_centered @ Y_centered.T / M

        eig_vals, eig_vecs = np.linalg.eigh(Sigma)
        sqrt_eig_vals = np.sqrt(eig_vals)
        Sigma_sqrt = eig_vecs.dot(np.diag(sqrt_eig_vals)).dot(np.linalg.inv(eig_vecs))

        modified_theory_values = {'Sigma': Sigma,
                                  'Sigma_sqrt': Sigma_sqrt,
                                  'Y_mean': Y_mean.squeeze(axis=-1),
                                  'min_eigval': eig_vals[0],
                                  'max_eigval': eig_vals[-1]}

        return original_theory_values, modified_theory_values


# Decision Transformer implementation
class TransformerBlock(nn.Module):
    def __init__(
            self,
            seq_len: int,
            embedding_dim: int,
            num_heads: int,
            attention_dropout: float,
            residual_dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, attention_dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(residual_dropout),
        )
        # True value indicates that the corresponding position is not allowed to attend
        self.register_buffer(
            "causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool)
        )
        self.seq_len = seq_len

    # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]
    def forward(
            self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        causal_mask = self.causal_mask[: x.shape[1], : x.shape[1]]

        norm_x = self.norm1(x)
        attention_out = self.attention(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        # by default pytorch attention does not use dropout
        # after final attention weights projection, while minGPT does:
        # https://github.com/karpathy/minGPT/blob/7218bcfa527c65f164de791099de715b81a95106/mingpt/model.py#L70 # noqa
        x = x + self.drop(attention_out)
        x = x + self.mlp(self.norm2(x))
        return x


class DecisionTransformer(nn.Module):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            seq_len: int = 10,
            episode_len: int = 1000,
            embedding_dim: int = 128,
            num_layers: int = 4,
            num_heads: int = 8,
            attention_dropout: float = 0.0,
            residual_dropout: float = 0.0,
            embedding_dropout: float = 0.0,
            max_action: float = 1.0,
    ):
        super().__init__()
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        # additional seq_len embeddings for padding timesteps
        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=3 * seq_len,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # self.action_head = nn.Sequential(nn.Linear(embedding_dim, action_dim), nn.Tanh())
        # Modified: Remove tanh(), but predict arctanh(action) instead
        self.action_head = nn.Linear(embedding_dim, action_dim)

        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.max_action = max_action

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_embeddings(
            self,
            states: torch.Tensor,  # [batch_size, seq_len, state_dim]
            actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
            returns_to_go: torch.Tensor,  # [batch_size, seq_len]
            time_steps: torch.Tensor,  # [batch_size, seq_len]
            padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.FloatTensor:

        batch_size, seq_len = states.shape[0], states.shape[1]
        # [batch_size, seq_len, emb_dim]
        time_emb = self.timestep_emb(time_steps)
        state_emb = self.state_emb(states) + time_emb
        act_emb = self.action_emb(actions) + time_emb
        returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + time_emb

        # [batch_size, seq_len * 3, emb_dim], (r_0, s_0, a_0, r_1, s_1, a_1, ...)
        sequence = (
            torch.stack([returns_emb, state_emb, act_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.embedding_dim)
        )
        if padding_mask is not None:
            # [batch_size, seq_len * 3], stack mask identically to fit the sequence
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 3 * seq_len)
            )
        # LayerNorm and Dropout (!!!) as in original implementation,
        # while minGPT & huggingface uses only embedding dropout
        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        out = self.out_norm(out)

        # predict actions only from state embeddings
        return out[:, 1::3]

    def forward(
            self,
            states: torch.Tensor,  # [batch_size, seq_len, state_dim]
            actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
            returns_to_go: torch.Tensor,  # [batch_size, seq_len]
            time_steps: torch.Tensor,  # [batch_size, seq_len]
            padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.FloatTensor:
        embeddings = self.get_embeddings(states, actions, returns_to_go, time_steps, padding_mask)
        # [batch_size, seq_len, action_dim]
        out = self.action_head(embeddings) * self.max_action
        return out

    def project(self, embeddings):
        return self.action_head(embeddings) * self.max_action

    def act(
            self,
            states: torch.Tensor,  # [batch_size, seq_len, state_dim]
            actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
            returns_to_go: torch.Tensor,  # [batch_size, seq_len]
            time_steps: torch.Tensor,  # [batch_size, seq_len]
            padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ):
        return F.tanh(self(states, actions, returns_to_go, time_steps, padding_mask))


# Training and evaluation logic
@torch.no_grad()
def eval_rollout(
        model: DecisionTransformer,
        env: gym.Env,
        target_return: float,
        device: str = "cpu",
) -> Tuple[float, float]:
    states = torch.zeros(
        1, model.episode_len + 1, model.state_dim, dtype=torch.float, device=device
    )
    actions = torch.zeros(
        1, model.episode_len, model.action_dim, dtype=torch.float, device=device
    )
    returns = torch.zeros(1, model.episode_len + 1, dtype=torch.float, device=device)
    time_steps = torch.arange(model.episode_len, dtype=torch.long, device=device)
    time_steps = time_steps.view(1, -1)

    states[:, 0] = torch.as_tensor(env.reset(), device=device)
    returns[:, 0] = torch.as_tensor(target_return, device=device)

    # cannot step higher than model episode len, as timestep embeddings will crash
    episode_return, episode_len = 0.0, 0.0
    for step in range(model.episode_len):
        # first select history up to step, then select last seq_len states,
        # step + 1 as : operator is not inclusive, last action is dummy with zeros
        # (as model will predict last, actual last values are not important)
        # Modified: model.forward() gives arctanh(a); need model.act()
        predicted_actions = model.act(  # fix this noqa!!!
            states[:, : step + 1][:, -model.seq_len:],
            actions[:, : step + 1][:, -model.seq_len:],
            returns[:, : step + 1][:, -model.seq_len:],
            time_steps[:, : step + 1][:, -model.seq_len:],
        )
        predicted_action = predicted_actions[0, -1].cpu().numpy()
        next_state, reward, done, info = env.step(predicted_action)
        # at step t, we predict a_t, get s_{t + 1}, r_{t + 1}
        actions[:, step] = torch.as_tensor(predicted_action)
        states[:, step + 1] = torch.as_tensor(next_state)
        returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward)

        episode_return += reward
        episode_len += 1

        if done:
            break

    return episode_return, episode_len


# @pyrallis.wrap()
def train_DT(config: TrainConfig):
    config.env_name = f"{config.env}-{config.dataset}-v2"
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)

    # data & dataloader setup
    dataset = SequenceDataset(
        config.env_name,
        seq_len=config.seq_len,
        reward_scale=config.reward_scale,
        max_action=config.max_action,
        action_epsilon=config.action_epsilon
    )
    trainloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        num_workers=config.num_workers,
    )
    # evaluation environment with state & reward preprocessing (as in dataset above)
    eval_env = wrap_env(
        env=gym.make(config.env_name),
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        reward_scale=config.reward_scale,
    )
    # model & optimizer & scheduler setup
    config.state_dim = eval_env.observation_space.shape[0]
    config.action_dim = eval_env.action_space.shape[0]
    model = DecisionTransformer(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        embedding_dim=config.embedding_dim,
        seq_len=config.seq_len,
        episode_len=config.episode_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        max_action=config.max_action,
    ).to(config.device)

    # Check for existing trained model and load if present
    model_save_folder = os.path.join(config.project_folder, 'models')
    os.makedirs(model_save_folder, exist_ok=True)
    # model_save_name = "%s_%s_WD%s" % (config.env,
    #                                   config.dataset,
    #                                   config.weight_decay,
    #                                   )
    model_save_path = os.path.join(model_save_folder, f"{config.name}_checkpoint.pt")

    start_step = 0
    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint["model_state"])
        start_step = checkpoint["step"]
        print(f"Resumed training from step {start_step}")
    if start_step >= config.update_steps:
        raise ValueError("Total number of updates is smaller than 'start_step'.")

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda steps: min((steps + 1) / config.warmup_steps, 1),
    )
    # # save config to the checkpoint
    # if config.checkpoints_path is not None:
    #     print(f"Checkpoints path: {config.checkpoints_path}")
    #     os.makedirs(config.checkpoints_path, exist_ok=True)
    #     with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
    #         pyrallis.dump(config, f)

    # init wandb session for logging
    wandb_init(asdict(config))

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Prepare logging for NRC/training metrics
    all_WWT = []
    W = model.action_head.weight.detach().clone().cpu().numpy()
    WWT = W @ W.T
    all_WWT.append(WWT.reshape(1, -1))
    all_metrics = {}

    trainloader_iter = iter(trainloader)
    for step in trange(start_step, config.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        states, actions, returns, time_steps, mask = [b.to(config.device) for b in batch]
        # True value indicates that the corresponding key value will be ignored
        padding_mask = ~mask.to(torch.bool)

        # Modified: predict arctanh(action) instead
        predicted_actions = model(
            states=states,
            actions=actions,
            returns_to_go=returns,
            time_steps=time_steps,
            padding_mask=padding_mask,
        )

        arctanh_actions = torch.arctanh(actions.detach())

        loss = F.mse_loss(predicted_actions, arctanh_actions, reduction="none")
        # [batch_size, seq_len, action_dim] * [batch_size, seq_len, 1]
        loss = (loss * mask.unsqueeze(-1)).mean()

        optim.zero_grad()
        loss.backward()
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optim.step()
        scheduler.step()

        wandb.log(
            {
                "train_loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
            },
            step=step,
        )

        # validation in the env for the actual online performance
        if step % config.eval_every == 0 or step == config.update_steps - 1:
            model.eval()
            # *********************************************
            # ****************RL Evaluation****************
            # *********************************************
            for target_return in config.target_returns:
                eval_env.seed(config.eval_seed)
                eval_returns = []
                for _ in trange(config.eval_episodes, desc="Evaluation", leave=False):
                    eval_return, eval_len = eval_rollout(
                        model=model,
                        env=eval_env,
                        target_return=target_return * config.reward_scale,
                        device=config.device,
                    )
                    # unscale for logging & correct normalized score computation
                    eval_returns.append(eval_return / config.reward_scale)

                normalized_scores = (
                        eval_env.get_normalized_score(np.array(eval_returns)) * 100
                )
                rl_log = {
                        f"eval/{target_return}_return_mean": np.mean(eval_returns),
                        f"eval/{target_return}_return_std": np.std(eval_returns),
                        f"eval/{target_return}_normalized_score_mean": np.mean(
                            normalized_scores
                        ),
                        f"eval/{target_return}_normalized_score_std": np.std(
                            normalized_scores
                        ),
                    }
                wandb.log(rl_log, step=step)

            # **********************************************
            # ****************NRC Evaluation****************
            # **********************************************
            with torch.no_grad():
                y = torch.empty((0,), device=config.device)
                H = torch.empty((0,), device=config.device)
                WH = torch.empty((0,), device=config.device)
                W = model.action_head.weight.detach().clone()

                for _ in range(config.num_NRC_batches):
                    NRC_batch = next(trainloader_iter)
                    states, actions, returns, time_steps, mask = [b.to(config.device) for b in NRC_batch]
                    # True value indicates that the corresponding key value will be ignored
                    padding_mask = ~mask.to(torch.bool)
                    bool_mask = mask.to(torch.bool).flatten()

                    features = model.get_embeddings(
                        states=states,
                        actions=actions,
                        returns_to_go=returns,
                        time_steps=time_steps,
                        padding_mask=padding_mask,
                    ).reshape(-1, config.embedding_dim)[bool_mask]
                    predictions = model.project(features)
                    targets = torch.arctanh(actions.detach()).reshape(-1, config.action_dim)[bool_mask]

                    y = torch.cat((y, targets), dim=0)
                    H = torch.cat((H, features), dim=0)
                    WH = torch.cat((WH, predictions), dim=0)

                res = {'targets': y,
                       'features': H,
                       'predictions': WH,
                       'weights': W
                       }
                nrc_log = compute_metrics(res, config.device, info=None)

                wandb.log({"NRC/" + k: v for k, v in nrc_log.items()}, step=step)

                all_metrics[str(step)] = nrc_log
                all_metrics[str(step)].update(rl_log)

                W = model.action_head.weight.detach().clone().cpu().numpy()
                WWT = W @ W.T
                all_WWT.append(WWT.reshape(1, -1))

            model.train()

    # if config.checkpoints_path is not None:
    #     checkpoint = {
    #         "model_state": model.state_dict(),
    #         "state_mean": dataset.state_mean,
    #         "state_std": dataset.state_std,
    #     }
    #     torch.save(checkpoint, os.path.join(config.checkpoints_path, "dt_checkpoint.pt"))

    # **********************************************
    # ****************Post Training*****************
    # **********************************************
    original_theory_values, modified_theory_values = dataset.get_theoretic_values()
    Sigma_sqrt = modified_theory_values['Sigma_sqrt']
    min_eigval = modified_theory_values['min_eigval']

    # Log NRC3 related curves to wandb
    WWT = all_WWT[-1].reshape(config.action_dim, -1)
    WWT_normalized = WWT / np.linalg.norm(WWT)

    c_to_plot = np.linspace(0, min_eigval, num=1000)
    NRC3_to_plot = []
    for c in c_to_plot:
        c_sqrt = c ** 0.5
        A = Sigma_sqrt - c_sqrt * np.eye(Sigma_sqrt.shape[0])
        A_normalized = A / np.linalg.norm(A)
        diff_mat = WWT_normalized - A_normalized
        NRC3_to_plot.append((np.linalg.norm(diff_mat).item()) ** 2)
    best_c = c_to_plot[np.argmin(NRC3_to_plot)]

    data = [[a, b] for (a, b) in zip(c_to_plot, NRC3_to_plot)]
    table = wandb.Table(data=data, columns=["gamma", "NRC3"])
    wandb.log(
        {
            "NRC3_vs_gamma": wandb.plot.line(
                table, "gamma", "NRC3", title="NRC3_vs_gamma"
            )
        }
    )

    wandb.log({'NRC/best_gamma': best_c})

    # ===================================================
    all_WWT = np.concatenate(all_WWT, axis=0)
    all_WWT_normalized = all_WWT / np.linalg.norm(all_WWT, axis=1, keepdims=True)
    A_c = Sigma_sqrt - (best_c ** 0.5) * np.eye(Sigma_sqrt.shape[0])
    A_c = A_c / np.linalg.norm(A_c)
    all_NCR3 = np.linalg.norm(all_WWT_normalized - A_c.reshape(1, -1), axis=1) ** 2

    ep_to_plot = np.arange(all_WWT.shape[0])
    data = [[a, b] for (a, b) in zip(ep_to_plot, all_NCR3)]
    table = wandb.Table(data=data, columns=["step", "NRC3"])
    wandb.log(
        {
            "NRC3_vs_step": wandb.plot.line(
                table, "step", "NRC3", title="NRC3_vs_step"
            )
        }
    )

    # Save all results to local
    metrics_save_folder = os.path.join(config.project_folder, 'metrics')
    os.makedirs(metrics_save_folder, exist_ok=True)
    metrics_save_path = os.path.join(metrics_save_folder, f"{config.name}_metrics.pkl")

    exist_metrics = {}
    if os.path.exists(metrics_save_path):
        with open(metrics_save_path, "rb") as f:
            exist_metrics = pickle.load(f)
    exist_metrics.update(all_metrics)
    with open(metrics_save_path, "wb") as f:
        pickle.dump(exist_metrics, f)

    # Save dataset related theoretical values to local
    dataset_save_folder = os.path.join(config.project_folder, 'results/dt')
    os.makedirs(dataset_save_folder, exist_ok=True)
    dataset_save_path = os.path.join(dataset_save_folder, config.name + '.json')

    readable_original = {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in original_theory_values.items()
    }
    readable_modified = {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in modified_theory_values.items()
    }

    with open(dataset_save_path, 'w') as f:
        json.dump(readable_original, f, indent=4)
        json.dump(readable_modified, f, indent=4)

    # Save trained model to local
    checkpoint = {
        "model_state": model.state_dict(),
        "step": config.update_steps,
        "config": asdict(config),
    }
    torch.save(checkpoint, model_save_path)
    print(f"Model checkpoint saved at {model_save_path}")


if __name__ == "__main__":
    train_DT(TrainConfig())
