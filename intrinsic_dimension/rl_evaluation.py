import os
import pickle
import random
import uuid
import argparse
from typing import Optional
import gym
import d4rl
import wandb
from tqdm.auto import trange

import numpy as np
import torch
from torch import nn
from compute_ID import compute_intrinsic_dimension


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
        settings=wandb.Settings(start_method="fork", init_timeout=360)
    )
    wandb.run.save()


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


def get_setting_dt(settings, setting_number, random_setting_seed=0, random_order=False):
    np.random.seed(random_setting_seed)
    hypers, lognames, values_list = [], [], []
    hyper2logname = {}
    n_settings = int(len(settings) / 3)
    for i in range(n_settings):
        hypers.append(settings[i * 3])
        lognames.append(settings[i * 3 + 1])
        values_list.append(settings[i * 3 + 2])
        hyper2logname[hypers[-1]] = lognames[-1]

    total = 1
    for values in values_list:
        total *= len(values)
    max_job = total

    new_indexes = np.random.choice(total, total, replace=False) if random_order else np.arange(total)
    new_index = new_indexes[setting_number]

    indexes = []  ## this says which hyperparameter we use
    remainder = new_index
    for values in values_list:
        division = int(total / len(values))
        index = int(remainder / division)
        remainder = remainder % division
        indexes.append(index)
        total = division
    actual_setting = {}
    for j in range(len(indexes)):
        actual_setting[hypers[j]] = values_list[j][indexes[j]]

    return indexes, actual_setting, max_job, hyper2logname


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0, arch: str = '3-256-R'):
        super(Actor, self).__init__()

        depth, width, activation = arch.split('-')
        arch = []
        for _ in range(int(depth)):
            arch.append(width)
            arch.append(activation)

        use_bias = True

        in_dim = state_dim
        module_list = []
        for i, layer in enumerate(arch):
            if layer == 'R':
                module_list.append(nn.ReLU())
            elif layer == 'T':
                module_list.append(nn.Tanh())
            elif layer == 'G':
                module_list.append(nn.GELU())
            elif layer == 'L':
                module_list.append(nn.LeakyReLU())
            elif layer == 'BL':
                module_list.append(nn.BatchNorm1d(in_dim))
                module_list.append(nn.LeakyReLU())
            elif layer == 'BR':
                module_list.append(nn.BatchNorm1d(in_dim))
                module_list.append(nn.ReLU())
            else:
                out_dim = int(layer)
                module_list.append(nn.Linear(in_dim, out_dim))
                in_dim = out_dim

        self.feature_map = nn.Sequential(*module_list)
        self.W = nn.Linear(in_dim, action_dim, bias=use_bias)

        self.max_action = max_action

    def get_feature(self, state: torch.Tensor):
        return self.feature_map(state)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        H = self.get_feature(state)
        return self.W(H)

    def project(self, feature):
        return self.W(feature)

    # @torch.no_grad()
    # def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
    #     state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
    #     # Modified: Clip the actions, since we do not have a tanh in the actor.
    #     action = self(state).clamp(min=-self.max_action, max=self.max_action)
    #     return action.cpu().data.numpy().flatten()

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        # Modified: Apply tanh(), since the actor predicts arctanh of actions.
        action = torch.tanh(self(state))
        return action.cpu().data.numpy().flatten()


def to_tensor(data: np.ndarray, device='cpu') -> torch.Tensor:
    return torch.tensor(data, dtype=torch.float32, device=device)


@torch.no_grad()
def RL_eval(
        env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


if __name__ == '__main__':
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=int, default=0)
    args = parser.parse_args()
    setting = args.setting

    settings = [
        'env', 'E', ['halfcheetah'],
        'max_epochs', 'MaxEp', [int(6e4)],
        'data_size', 'DS', [0.1],
        'lamW', 'WD', [0, 0.0001, 0.001, 0.01],
        'arch', 'A', ['3-64-R', '3-128-R', '3-256-R', '3-512-R', '3-1024-R',
                      '1-256-R', '2-256-R', '4-256-R', '5-256-R', '5-1024-R'],
        # 'arch', 'A', ['3-256-R'],
        'seed', '', list(range(5)),

        'group', '', ['10%BC'],
    ]

    indexes, config, total, hyper2logname = get_setting_dt(settings, setting)
    config['name'] = '_'.join([v + str(config[k]) for k, v in hyper2logname.items() if v != ''])
    # config['group'] = 'test'
    config['project'] = 'Intrinsic_Dim'
    config['device'] = DEVICE

    env = gym.make(f"{config['env']}-expert-v2")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    set_seed(config['seed'], env)

    MODEL_FOLDER = f"./models/{config['env']}/{config['name']}"
    DATA_PATH = f"./dataset/{config['env']}_top{config['data_size']}.pkl"
    with open(DATA_PATH, 'rb') as f:
        train_dataset = pickle.load(f)
    X = to_tensor(train_dataset['observations'], config['device'])
    Y = to_tensor(train_dataset['actions'], config['device'])
    Y = torch.arctanh(torch.clamp(Y, min=-1 + 1e-7, max=1 - 1e-7))
    print('Computing target ID...')
    target_ID = compute_intrinsic_dimension(Y, batch_size=2000, epsilon=0.)
    print('Finished computing target ID!')

    wandb_init(config)

    for epoch in trange(config['max_epochs'], desc='Epochs'):
        if epoch == 0 or (epoch + 1) in list(range(0, config['max_epochs'] + 1, config['max_epochs'] // 100)):
            print('===================================================')
            model_name = config['name'] + f'_ep{epoch + 1 if epoch > 0 else 0}.pt'
            model_path = os.path.join(MODEL_FOLDER, model_name)
            print('Loading model from the path: ', model_path)
            model_log = torch.load(model_path, map_location=config['device'])
            model_info = model_log['config']
            model_state_dict = model_log['model_state']
            model = Actor(state_dim, action_dim, max_action=1, arch=config['arch']).to(config['device'])
            model.load_state_dict(model_state_dict)
            print('Model sucessfully loaded.')
            print('===================================================')

            # RL evaluation
            eval_scores = RL_eval(
                env,
                model,
                device=config['device'],
                n_episodes=10,
                seed=config['seed'],
            )
            eval_score = eval_scores.mean()
            normalized_eval_score = env.get_normalized_score(eval_score)

            # ID evaluation
            with torch.no_grad():
                H = model.get_feature(X)
                feature_ID = compute_intrinsic_dimension(H, batch_size=2000, epsilon=0.)

            wandb.log(
                {"d4rl_normalized_score": normalized_eval_score,
                 "feature_id": feature_ID,
                 "target_id": target_ID},
                step=epoch,
            )

    print('Full experiment config for the last checkpoint:', model_info, sep='\n')
