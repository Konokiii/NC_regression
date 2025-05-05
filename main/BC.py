import os
import pickle
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from tqdm import tqdm

import numpy as np
from sklearn.decomposition import PCA
from scipy.linalg import sqrtm
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import Dataset, DataLoader

TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "swimmer"  # OpenAI gym environment name. Choose from 'swimmer' and 'reacher'.
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = 5  # How often (epochs) we evaluate
    max_epochs: int = 200  # How many epochs to run
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    batch_size: int = 256  # Batch size for all networks
    num_eval_batch: int = 400  # Do NC evaluation over a subset of the whole dataset
    data_size: int = 1000  # Number of samples to use
    normalize: str = 'none'  # Choose from 'none', 'normal', 'standard', 'center'

    arch: str = '3-256-R'  # Actor architecture
    optimizer: str = 'sgd'
    lamH: float = -1  # If it is -1, then the model is not UFM.
    lamW: float = 5e-2
    lr: float = 1e-2

    mode: str = 'null'
    data_folder: str = '/NC_regression/dataset/mujoco'
    project_folder: str = '/NC_regression'
    saved_model: Optional[str] = None

    # Wandb logging
    project: str = "NC_regression"
    group: str = "test"
    name: str = "test"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


# Save model
def save_model(save_name, model, config):
    save_folder = os.path.join(config.project_folder, f'models/{config.env}')
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, save_name + '.pt')
    checkpoint = {
        "model_state": model.cpu().state_dict(),
        "config": asdict(config),
    }
    torch.save(checkpoint, save_path)
    model.to(config.device)


def cosine_similarity_gpu(a, b):
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1) if not torch.equal(a, b) else a_norm

    return torch.mm(a_norm, b_norm.T)


def gram_schmidt(W):
    dim = W.shape[0]

    U = torch.empty_like(W)
    U[0, :] = W[0, :] / torch.norm(W[0, :], p=2)

    for i in range(1, dim):
        j = i - 1
        ortho_vector = W[i, :]
        while j >= 0:
            proj = torch.dot(U[j, :], W[i, :]) * U[j, :]
            ortho_vector -= proj
            j -= 1
        U[i, :] = ortho_vector / torch.norm(ortho_vector, p=2)

    return U


def compute_NRC1(features, target_dim, result_dict):
    H = features
    H_normalized = H / (torch.norm(H, dim=1, keepdim=True) + 1e-8)
    n_components = max(target_dim, 10)

    # Center the data
    H_mean = H.mean(dim=0, keepdim=True)
    H_centered = H - H_mean
    H_normalized_centered = H_centered / (torch.norm(H_centered, dim=1, keepdim=True) + 1e-8)

    U, S, Vh = torch.linalg.svd(H_centered, full_matrices=False)

    # Explained variance ratio
    explained_variance = (S ** 2) / (H_centered.shape[0] - 1)
    total_variance = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_variance

    H_pca = Vh[:n_components, :]
    H_U = gram_schmidt(H_pca)

    for k in range(n_components):
        result_dict[f'EVR{k + 1}'] = explained_variance_ratio[k].item()
        H_P = torch.mm(H_U[:k + 1, :].T, H_U[:k + 1, :])

        H_proj_PCA = torch.mm(H, H_P)
        result_dict[f'NRC1_pca{k + 1}'] = torch.norm(H_proj_PCA - H).item() ** 2 / H.shape[0]

        H_proj_PCA_normalized = torch.mm(H_normalized, H_P)
        result_dict[f'NRC1n_pca{k + 1}'] = torch.norm(H_proj_PCA_normalized - H_normalized).item() ** 2 / H.shape[0]

        H_proj_PCA_normalized_centered = torch.mm(H_normalized_centered, H_P)
        result_dict[f'NRC1nc_pca{k + 1}'] = torch.norm(H_proj_PCA_normalized_centered - H_normalized_centered).item() ** 2 / H.shape[0]

    result_dict['NRC1'] = result_dict[f'NRC1_pca{target_dim}']
    result_dict['NRC1n'] = result_dict[f'NRC1n_pca{target_dim}']
    result_dict['NRC1nc'] = result_dict[f'NRC1nc_pca{target_dim}']


def compute_metrics(metrics, split, device, info=None):
    result = {}
    y = metrics['targets']  # (B,2)
    yhat = metrics['outputs']  # (B,2)
    W = metrics['weights']  # (2,256)
    H = metrics['embeddings']  # (B,256)
    H_normalized = H / (torch.norm(H, dim=1, keepdim=True) + 1e-8)
    B = H.shape[0]

    result['prediction_error'] = F.mse_loss(y, yhat).item()
    # result['cosSIM_y_yhat'] = cosine_similarity_gpu(y, yhat).mean().item()

    if split == 'test':
        return result

    y_dim = y.shape[1]

    # R^2
    SS_res = torch.sum((y-yhat)**2).item()
    SS_tot = torch.sum((y-y.mean(dim=0))**2).item()
    result['R_sq'] = 1 - SS_res/SS_tot

    result['H_norm'] = torch.norm(H, p=2, dim=1).mean().item()
    result['W_norm'] = torch.norm(W, p=2, dim=1).mean().item()

    # WWT
    WWT = W @ W.T
    if y_dim == 2:
        result['W11'] = WWT[0, 0].item()
        result['W22'] = WWT[1, 1].item()
        result['W12'] = WWT[0, 1].item()
        result['cosSIM_W12'] = F.cosine_similarity(W[0], W[1], dim=0).item()
    elif y_dim == 3:
        result['W11'] = WWT[0, 0].item()
        result['W22'] = WWT[1, 1].item()
        result['W33'] = WWT[2, 2].item()
        result['W12'] = WWT[0, 1].item()
        result['W13'] = WWT[0, 2].item()
        result['W23'] = WWT[1, 2].item()
        result['cosSIM_W12'] = F.cosine_similarity(W[0], W[1], dim=0).item()
        result['cosSIM_W13'] = F.cosine_similarity(W[0], W[2], dim=0).item()
        result['cosSIM_W23'] = F.cosine_similarity(W[1], W[2], dim=0).item()
    elif y_dim > 3:
        W_normalized = W / (torch.norm(W, dim=1, keepdim=True) + 1e-8)
        WWT_normalized = W_normalized @ W_normalized.T
        result['cosSIM_avg'] = (WWT_normalized.sum() - WWT_normalized.trace()) / (y_dim**2-y_dim)

    result['WWT_norm'] = torch.norm(WWT).item()

    # NRC1 & explained variance ratio
    compute_NRC1(H, y_dim, result)
    # H_np = H.cpu().numpy()
    # n_components = max(y_dim, 5)
    # pca_for_H = PCA(n_components=n_components)
    # try:
    #     pca_for_H.fit(H_np)
    # except Exception as e:
    #     print('Initial PCA failed with error:', e)
    #     print('Try PCA with full SVD solver.')
    #     pca_for_H = PCA(n_components=n_components, svd_solver='full')
    #     pca_for_H.fit(H_np)
    #     # for k in range(n_components):
    #     #     result[f'NRC1_pca{k+1}'] = -1
    #     #     result[f'NRC1n_pca{k + 1}'] = -1
    # finally:
    #     H_pca = torch.tensor(pca_for_H.components_[:n_components, :], device=device)
    #     H_U = gram_schmidt(H_pca)
    #     for k in range(n_components):
    #         result[f'EVR{k+1}'] = pca_for_H.explained_variance_ratio_[k]
    #
    #         H_P = torch.mm(H_U[:k+1, :].T, H_U[:k+1, :])
    #         H_proj_PCA = torch.mm(H, H_P)
    #         result[f'NRC1_pca{k + 1}'] = torch.norm(H_proj_PCA - H).item() ** 2 / B
    #
    #         H_proj_PCA_normalized = torch.mm(H_normalized, H_P)
    #         result[f'NRC1n_pca{k + 1}'] = torch.norm(H_proj_PCA_normalized - H_normalized).item() ** 2 / B
    # del H_pca, H_U, H_P, H_proj_PCA, H_proj_PCA_normalized, pca_for_H, H_np
    #
    # result['NRC1'] = result[f'NRC1_pca{y_dim}']
    # result['NRC1n'] = result[f'NRC1n_pca{y_dim}']

    # NRC2 with Gram-Schmidt
    U = gram_schmidt(W)
    P_E = torch.mm(U.T, U)
    H_proj_W = torch.mm(H, P_E)
    H_proj_W_normalized = torch.mm(H_normalized, P_E)
    result['NRC2'] = torch.norm(H - H_proj_W).item() ** 2 / B
    result['NRC2n'] = torch.norm(H_normalized - H_proj_W_normalized).item() ** 2 / B
    del H_proj_W, H_proj_W_normalized

    if info:
        assert isinstance(info, dict), 'Extra info used to compute should be a dictionary.'
        NRC3_target = info['NRC3_target']
        NRC3n_target = info['NRC3n_target']

        # NRC3 with UFM assumption
        result['NRC3n_ufm'] = torch.norm(WWT / torch.norm(WWT) - torch.tensor(NRC3n_target, device=device)).item() ** 2
        result['NRC3_ufm'] = torch.norm(WWT - torch.tensor(NRC3_target, device=device)).item() ** 2

    # NRC2: Project H to span(w1, w2)
    # try:
    #     inverse_mat = torch.inverse(W @ W.T)
    # except Exception as e:
    #     print(e)
    #     result['NRC2_old'] = -1
    # else:
    #     H_proj_W = H @ W.T @ inverse_mat @ W
    #     result['NRC2_old'] = (torch.norm(H - H_proj_W)**2 / B).item()
    #     del H_proj_W

    return result


def set_seed(
        seed: int, env=None, deterministic_torch: bool = False
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
        settings=wandb.Settings(start_method="fork")  # Workaround for ERROR Run initialization has timed out after 60.0 sec.
    )
    wandb.run.save()


# @torch.no_grad()
# def RL_eval(
#         env, actor: nn.Module, device: str, n_episodes: int, seed: int
# ) -> np.ndarray:
#     env.seed(seed)
#     actor.eval()
#     episode_rewards = []
#     for _ in range(n_episodes):
#         state, done = env.reset(), False
#         episode_reward = 0.0
#         while not done:
#             action = actor.act(state, device)
#             state, reward, done, _ = env.step(action)
#             episode_reward += reward
#         episode_rewards.append(episode_reward)
#
#     actor.train()
#     return np.asarray(episode_rewards)


class MujocoBuffer(Dataset):
    def __init__(
            self,
            data_folder: str,
            env: str,
            split: str,
            data_size: int,
            normalize: str,
            device: str = "cpu",
    ):
        self.device = device

        self.size = 0
        self.state_dim = 0
        self.action_dim = 0

        self.states, self.actions = None, None
        self._load_dataset(data_folder, env, split, data_size, normalize)

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self.device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def _load_dataset(self, data_folder: str, env: str, split: str, data_size: int, normalize: str):
        file_name = '%s_%s.pkl' % (env, split)
        file_path = os.path.join(data_folder, file_name)
        try:
            with open(file_path, 'rb') as file:
                dataset = pickle.load(file)
                if split == 'test':
                    if env in ['swimmer', 'hopper']:
                        data_size = max(data_size // 5, 1000)
                    elif env == 'reacher':
                        data_size = max(data_size // 5, 50)

                self.size = data_size
                self.states = dataset['observations'][:self.size, :]
                self.actions = dataset['actions'][:self.size, :]
            print('Successfully load dataset from: ', file_path)
        except Exception as e:
            print(e)

        self.state_dim = self.states.shape[1]
        self.action_dim = self.actions.shape[1]

        if normalize == 'none':
            pass
        elif normalize == 'standard':
            mean = self.actions.mean(0)
            std = np.maximum(self.actions.std(0), 1e-4)
            self.actions = (self.actions - mean) / std
        elif normalize == 'normal':
            min_action = self.actions.min(axis=0)
            max_action = self.actions.max(axis=0)
            self.actions = (self.actions - min_action) / (max_action - min_action)
        elif normalize == 'center':
            mean = self.actions.mean(0)
            self.actions = self.actions - mean

        print(f"Dataset size: {self.size}; State Dim: {self.state_dim}; Action_Dim: {self.action_dim}.")

        self.states = self._to_tensor(self.states)
        self.actions = self._to_tensor(self.actions)

    def get_state_dim(self):
        return self.state_dim

    def get_action_dim(self):
        return self.action_dim

    def get_theory_stats(self):
        Y = self.actions.T
        y_dim = Y.shape[0]

        Y_mean = Y.mean(dim=1, keepdim=True)
        Y = Y - Y_mean
        M = Y.shape[1]
        Sigma = Y @ Y.T / M

        # Eigen decomposition (for symmetric matrix, use symeig or eigh)
        eig_vals, eig_vecs = torch.linalg.eigh(Sigma)
        sqrt_eig_vals = torch.sqrt(eig_vals)
        Sigma_sqrt = eig_vecs @ torch.diag(sqrt_eig_vals) @ torch.linalg.inv(eig_vecs)

        # Convert to numpy to save
        min_eigval = eig_vals[0].cpu().numpy()
        max_eigval = eig_vals[-1].cpu().numpy()
        Y_mean = Y_mean.cpu().numpy()
        Sigma_sqrt = Sigma_sqrt.cpu().numpy()
        Sigma = Sigma.cpu().numpy()

        # Y = self.actions.T
        # y_dim = Y.shape[0]
        # Y_mean = Y.mean(axis=1, keepdims=True)
        # Y = Y - Y_mean
        # M = Y.shape[1]
        # Sigma = Y @ Y.T / M
        #
        # # Sigma_sqrt = sqrtm(Sigma)
        # # eig_vals = np.linalg.eigvalsh(Sigma)
        # eig_vals, eig_vecs = np.linalg.eigh(Sigma)
        # sqrt_eig_vals = np.sqrt(eig_vals)
        # Sigma_sqrt = eig_vecs.dot(np.diag(sqrt_eig_vals)).dot(np.linalg.inv(eig_vecs))
        #
        # min_eigval = eig_vals[0]
        # max_eigval = eig_vals[-1]

        if y_dim == 2:
            return {
                'mu11': Sigma[0, 0],
                'mu12': Sigma[0, 1],
                'mu22': Sigma[1, 1],
                'min_eigval': min_eigval,
                'max_eigval': max_eigval,
                # 'gamma1': gamma1,
                # 'gamma2': gamma2,
                'sigma11': Sigma_sqrt[0, 0],
                'sigma12': Sigma_sqrt[0, 1],
                'sigma21': Sigma_sqrt[1, 0],
                'sigma22': Sigma_sqrt[1, 1],
                'm1': Y_mean[0, 0],
                'm2': Y_mean[1, 0]
            }, Sigma, Sigma_sqrt
        elif y_dim == 3:
            return {
                'mu11': Sigma[0, 0],
                'mu22': Sigma[1, 1],
                'mu33': Sigma[2, 2],
                'mu12': Sigma[0, 1],
                'mu13': Sigma[0, 2],
                'mu23': Sigma[1, 2],
                'min_eigval': min_eigval,
                'max_eigval': max_eigval,
                # 'gamma1': gamma1,
                # 'gamma2': gamma2,
                # 'sigma11': Sigma_sqrt[0, 0],
                # 'sigma12': Sigma_sqrt[0, 1],
                # 'sigma21': Sigma_sqrt[1, 0],
                # 'sigma22': Sigma_sqrt[1, 1],
                'm1': Y_mean[0, 0],
                'm2': Y_mean[1, 0],
                'm3': Y_mean[2, 0]
            }, Sigma, Sigma_sqrt
        elif y_dim > 3:
            return {
                'min_eigval': min_eigval,
                'max_eigval': max_eigval,
            }, Sigma, Sigma_sqrt

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {'states': self.states[idx],
                'actions': self.actions[idx]}

        # states = self.states[idx]
        # actions = self.actions[idx]
        # return {
        #     'states': self._to_tensor(states),
        #     'actions': self._to_tensor(actions)
        # }


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

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        # Modified: Clip the actions, since we do not have a tanh in the actor.
        action = self(state).clamp(min=-self.max_action, max=self.max_action)
        return action.cpu().data.numpy().flatten()


class BC:
    def __init__(
            self,
            actor: nn.Module,
            actor_optimizer: torch.optim.Optimizer,
            lamH: float,
            lamW: float,
            num_eval_batch: int,
            device: str = "cpu",
    ):
        self.actor = actor
        self.actor.train()

        self.actor_optimizer = actor_optimizer

        self.total_it = 0
        self.lamH = lamH
        self.lamW = lamW
        self.num_eval_batch = num_eval_batch
        self.device = device

    def train(self, batch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        states, actions = batch['states'], batch['actions']

        # Compute actor loss
        if self.lamH == -1:
            preds = self.actor(states)
            mse_loss = 0.5 * F.mse_loss(preds, actions)
            reg_loss = 0
            for param in self.actor.parameters():
                reg_loss += torch.norm(param) ** 2
            reg_loss = 0.5 * self.lamW * reg_loss
            train_loss = mse_loss + reg_loss
        else:
            H = self.actor.get_feature(states)
            preds = self.actor.project(H)
            mse_loss = 0.5 * F.mse_loss(preds, actions)
            reg_H_loss = 0.5 * self.lamH * (torch.norm(H, p=2) ** 2) / H.shape[0]
            reg_W_loss = 0.5 * self.lamW * torch.norm(self.actor.W.weight) ** 2
            train_loss = mse_loss + reg_H_loss + reg_W_loss

        log_dict["train_mse_loss"] = mse_loss.item()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        train_loss.backward()
        self.actor_optimizer.step()

        return log_dict

    @torch.no_grad()
    def NC_eval(self, dataloader, split, info=None):
        self.actor.eval()
        y = torch.empty((0,), device=self.device)
        H = torch.empty((0,), device=self.device)
        Wh = torch.empty((0,), device=self.device)
        W = self.actor.W.weight.detach().clone()

        for i, batch in enumerate(dataloader):
            if split == 'train' and i + 1 > self.num_eval_batch:
                break
            states, actions = batch['states'], batch['actions']
            features = self.actor.get_feature(states)
            preds = self.actor.project(features)

            y = torch.cat((y, actions), dim=0)
            H = torch.cat((H, features), dim=0)
            Wh = torch.cat((Wh, preds), dim=0)

        res = {'targets': y,
               'embeddings': H,
               'outputs': Wh,
               'weights': W
               }
        log_dict = compute_metrics(res, split, self.device, info=info)
        self.actor.train()

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.total_it = state_dict["total_it"]


def run_BC(config: TrainConfig):
    train_dataset = MujocoBuffer(
        data_folder=config.data_folder,
        env=config.env,
        split='train',
        data_size=config.data_size,
        normalize=config.normalize,
        device=config.device
    )
    val_dataset = MujocoBuffer(
        data_folder=config.data_folder,
        env=config.env,
        split='test',
        data_size=config.data_size,
        normalize=config.normalize,
        device=config.device
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    # Set seeds
    seed = config.seed
    set_seed(seed)

    state_dim = train_dataset.get_state_dim()
    action_dim = train_dataset.get_action_dim()

    # if config.saved_model is not None:
    #     load_model_path = os.path.join(config.project_folder, f'models/{config.env}/{config.saved_model}.pth')
    #     if os.path.exists(load_model_path):
    #         actor = torch.load(load_model_path).to(config.device)
    #     else:
    #         raise FileNotFoundError('Try to load pretrained model, but file not found.')

    actor = Actor(state_dim, action_dim, arch=config.arch).to(config.device)
    # Save the initial model
    save_model(save_name=config.name+'_ep0', model=actor, config=config)

    actor_optimizer = {'adam': torch.optim.Adam,
                       'sgd': torch.optim.SGD}[config.optimizer](actor.parameters(), lr=config.lr)

    kwargs = {
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "lamH": config.lamH,
        "lamW": config.lamW,
        'num_eval_batch': config.num_eval_batch,
        "device": config.device
    }

    print("---------------------------------------")
    print(f"Training BC, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize policy
    trainer = BC(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    # TODO: fix and optimize wandb log.
    train_theory_stats, Sigma, Sigma_sqrt = train_dataset.get_theory_stats()
    info = None
    if config.lamH != -1:
        # lamW can be zero, so take maximum between it and 1e-10.
        NRC3_target = (config.lamH / max(config.lamW, 1e-10)) ** 0.5 * (Sigma_sqrt - (config.lamH * config.lamW) ** 0.5 * np.eye(Sigma_sqrt.shape[0]))

        NRC3n_target = Sigma_sqrt - (config.lamH * config.lamW) ** 0.5 * np.eye(Sigma_sqrt.shape[0])
        NRC3n_target = NRC3n_target / np.linalg.norm(NRC3n_target)
        info = {'NRC3_target': NRC3_target,
                'NRC3n_target': NRC3n_target}

    train_log = trainer.NC_eval(train_loader, split='train', info=info)
    val_log = trainer.NC_eval(val_loader, split='test', info=info)
    wandb.log({'train': train_log,
               'validation': val_log,
               'C': train_theory_stats,
               })

    all_WWT = []
    W = actor.W.weight.detach().clone().cpu().numpy()
    WWT = W @ W.T
    all_WWT.append(WWT.reshape(1, -1))

    for epoch in range(config.max_epochs):
        epoch_train_loss = 0
        count = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{config.max_epochs} Training"):
            log_dict = trainer.train(batch)
            epoch_train_loss += log_dict['train_mse_loss']
            count += 1
            # wandb.log(trainer.train(batch), step=trainer.total_it)
        epoch_train_loss /= count
        if (epoch + 1) % config.eval_freq == 0:
            W = actor.W.weight.detach().clone().cpu().numpy()
            WWT = W @ W.T
            all_WWT.append(WWT.reshape(1, -1))
            train_log = trainer.NC_eval(train_loader, split='train', info=info)
            val_log = trainer.NC_eval(val_loader, split='test', info=info)
            wandb.log({'train_mse_loss': epoch_train_loss,
                       'train': train_log,
                       'validation': val_log,
                       'C': train_theory_stats,
                       })

        # if (epoch + 1) in [config.max_epochs // (i+1) for i in range(4)]:
        #     # Save NRC3 related data to local for later plots
        #     save_folder = os.path.join(config.project_folder, 'results/wwt')
        #     os.makedirs(save_folder, exist_ok=True)
        #     save_path = os.path.join(save_folder, config.name + '.pkl')
        #     with open(save_path, 'wb') as file:
        #         to_plot_nrc3 = {'WWT': np.concatenate(all_WWT, axis=0),
        #                         'Sigma_sqrt': Sigma_sqrt,
        #                         'min_eigval': train_theory_stats['min_eigval']}
        #         pickle.dump(to_plot_nrc3, file)

        if (epoch + 1) in list(range(0, config.max_epochs+1, config.max_epochs // 100)):
            # Save the initial model
            save_model(save_name=config.name + f'_ep{epoch+1}', model=actor, config=config)

    # NRC3
    W = actor.W.weight.detach().clone().cpu().numpy()
    WWT = W @ W.T
    all_WWT.append(WWT.reshape(1, -1))
    all_WWT = np.concatenate(all_WWT, axis=0)

    # # Save NRC3 related data to local for later plots
    # save_folder = os.path.join(config.project_folder, 'results/wwt')
    # os.makedirs(save_folder, exist_ok=True)
    # save_path = os.path.join(save_folder, config.name + '.pkl')
    # with open(save_path, 'wb') as file:
    #     to_plot_nrc3 = {'WWT': all_WWT,
    #                     'Sigma_sqrt': Sigma_sqrt,
    #                     'min_eigval': train_theory_stats['min_eigval']}
    #     pickle.dump(to_plot_nrc3, file)

    # Log NRC3 related curves to wandb
    WWT_normalized = WWT / np.linalg.norm(WWT)
    min_eigval = train_theory_stats['min_eigval']

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
    table = wandb.Table(data=data, columns=["c", "NRC3"])
    wandb.log(
        {
            "NRC3_vs_c": wandb.plot.line(
                table, "c", "NRC3", title="NRC3_vs_c"
            )
        }
    )

    wandb.log({'C': {'best_c': best_c}})

    # ===================================================
    all_WWT_normalized = all_WWT / np.linalg.norm(all_WWT, axis=1, keepdims=True)
    A_c = Sigma_sqrt - (best_c ** 0.5) * np.eye(Sigma_sqrt.shape[0])
    A_c = A_c / np.linalg.norm(A_c)
    all_NCR3 = np.linalg.norm(all_WWT_normalized - A_c.reshape(1, -1), axis=1) ** 2

    ep_to_plot = np.arange(all_WWT.shape[0])
    data = [[a, b] for (a, b) in zip(ep_to_plot, all_NCR3)]
    table = wandb.Table(data=data, columns=["epoch", "NRC3"])
    wandb.log(
        {
            "NRC3_vs_epoch": wandb.plot.line(
                table, "epoch", "NRC3", title="NRC3_vs_epoch"
            )
        }
    )

