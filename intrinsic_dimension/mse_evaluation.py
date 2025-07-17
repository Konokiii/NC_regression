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


def gram_schmidt(W: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    n_vecs = W.size(0)
    U = torch.zeros_like(W)

    for i in range(n_vecs):
        v = W[i].clone()

        for j in range(i):
            v = v - torch.dot(U[j], v) * U[j]

        norm = v.norm(p=2)
        # if norm < eps:
        #     raise ValueError(
        #         f"Vector {i} is linearly dependent (‖v‖={norm:.2e} < {eps})."
        #     )
        U[i] = v / norm

    return U


def compute_NRC1(features, target_dim):
    result_dict = {}

    H = features
    H_normalized = H / (torch.norm(H, dim=1, keepdim=True) + 1e-6)
    n_components = max(target_dim, 10)

    # Center the data
    H_mean = H.mean(dim=0, keepdim=True)
    H_centered = H - H_mean
    H_normalized_centered = H_centered / (torch.norm(H_centered, dim=1, keepdim=True) + 1e-6)

    U, S, Vh = torch.linalg.svd(H_centered, full_matrices=False)

    # Explained variance ratio
    explained_variance = (S ** 2) / (H_centered.shape[0] - 1)
    total_variance = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_variance
    for k in range(n_components):
        result_dict[f'EVR{k + 1}'] = explained_variance_ratio[k].item()

    H_pca = Vh[:n_components, :]
    H_U = gram_schmidt(H_pca)
    H_P = torch.mm(H_U[:target_dim, :].T, H_U[:target_dim, :])

    H_proj_PCA_normalized = torch.mm(H_normalized, H_P)
    result_dict['NRC1'] = torch.norm(H_proj_PCA_normalized - H_normalized).item() ** 2 / H.shape[0]

    H_proj_PCA_normalized_centered = torch.mm(H_normalized_centered, H_P)
    result_dict['NRC1c'] = torch.norm(H_proj_PCA_normalized_centered - H_normalized_centered).item() ** 2 / H.shape[0]

    return result_dict


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
        'env', 'E', ['ant'],
        'max_epochs', 'MaxEp', [int(3e5)],
        'data_size', 'DS', [20000],
        'lamW', 'WD', [0, 0.00001, 0.0001, 0.0003, 0.0005,
                       0.0007, 0.001, 0.003, 0.005, 0.007,
                       0.01, 0.03, 0.05],
        'arch', 'A', ['3-64-R', '3-128-R', '3-256-R', '3-512-R', '3-1024-R',
                      '1-256-R', '2-256-R', '4-256-R', '5-256-R', '5-1024-R'],
        'seed', '', list(range(1)),

        'group', '', ['mse_eval'],
    ]

    indexes, config, total, hyper2logname = get_setting_dt(settings, setting)
    config['name'] = '_'.join([v + str(config[k]) for k, v in hyper2logname.items() if v != ''])
    config['project'] = 'Intrinsic_Dim'
    config['device'] = DEVICE

    env = gym.make(f"{config['env']}-expert-v2")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    set_seed(config['seed'], env)

    MODEL_FOLDER = f"./models/{config['env']}/{config['name']}"
    TRAIN_DATA_PATH = f"./dataset/{config['env']}_train.pkl"
    TEST_DATA_PATH = f"./dataset/{config['env']}_test.pkl"
    DATA_SIZE = config['data_size']

    with open(TRAIN_DATA_PATH, 'rb') as f:
        train_dataset = pickle.load(f)
        train_dataset = {k: v[:DATA_SIZE] for k, v in train_dataset.items()}

    X_train = to_tensor(train_dataset['observations'], config['device'])
    Y_train = to_tensor(train_dataset['actions'], config['device'])
    print('Computing train target ID...')
    train_target_ID = compute_intrinsic_dimension(Y_train, batch_size=2000, epsilon=0.)
    print('Finished computing train target ID!')

    with open(TEST_DATA_PATH, 'rb') as f:
        test_dataset = pickle.load(f)
        test_dataset = {k: v[:max(DATA_SIZE // 5, 1000)] for k, v in test_dataset.items()}

    X_test = to_tensor(test_dataset['observations'], config['device'])
    Y_test = to_tensor(test_dataset['actions'], config['device'])
    print('Computing tset target ID...')
    test_target_ID = compute_intrinsic_dimension(Y_test, batch_size=2000, epsilon=0.)
    print('Finished computing test target ID!')

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

            # ID evaluation
            with torch.no_grad():
                H_train = model.get_feature(X_train)
                train_feature_ID = compute_intrinsic_dimension(H_train, batch_size=2000, epsilon=1e-6)
                NRC1_related = compute_NRC1(H_train, target_dim=action_dim)
                Y_train_hat = model.project(H_train)
                train_mse = torch.sum((Y_train - Y_train_hat) ** 2).item() / Y_train.shape[0]

                H_test = model.get_feature(X_test)
                test_feature_ID = compute_intrinsic_dimension(H_test, batch_size=2000, epsilon=1e-6)
                Y_test_hat = model.project(H_test)
                test_mse = torch.sum((Y_test - Y_test_hat) ** 2).item() / Y_test.shape[0]

            log_metrics = {"test_mse": test_mse,
                           "test_feature_id": test_feature_ID,
                           "test_target_id": test_target_ID,
                           "train_mse": train_mse,
                           "train_feature_id": train_feature_ID,
                           "train_target_id": train_target_ID}
            log_metrics.update(NRC1_related)
            wandb.log(log_metrics, step=epoch)

    print('Full experiment config for the last checkpoint:', model_info, sep='\n')
