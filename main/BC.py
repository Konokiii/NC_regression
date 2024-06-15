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
    # n_episodes: int = 10  # How many episodes run during RL evaluation
    max_epochs: int = 200  # How many epochs to run
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    batch_size: int = 256  # Batch size for all networks
    num_eval_batch: int = 100  # Do NC evaluation over a subset of the whole dataset
    data_size: int = 10  # Number of episodes to use
    normalize: str = 'none'  # Choose from 'none', 'normal', 'standard', 'center'

    arch: str = '256-R-256-R|T'  # Actor architecture
    optimizer: str = 'sgd'
    lamH: float = 1e-5  # If it is -1, then the model is not UFM.
    lamW: float = 5e-2
    lr: float = 1e-2

    mode: str = 'null'
    data_folder: str = '/NC_regression/dataset/mujoco'

    # Wandb logging
    project: str = "NC_regression"
    group: str = "test"
    name: str = "test"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def cosine_similarity_gpu(a, b):
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1) if not torch.equal(a, b) else a_norm

    return torch.mm(a_norm, b_norm.T)


def gram_schmidt(W):
    dim = W.shape[0]

    U = torch.empty_like(W)
    U[0, :] = W[0, :] / torch.norm(W[0, :], p=2)

    for i in range(1, dim):
        proj = torch.dot(U[i-1, :], W[i, :]) * U[i-1, :]
        ortho_vector = W[i, :] - proj
        U[i, :] = ortho_vector / torch.norm(ortho_vector, p=2)

    return U


def compute_metrics(metrics, split, device, extra_info=None):
    result = {}
    y = metrics['targets']  # (B,2)
    yhat = metrics['outputs']  # (B,2)
    W = metrics['weights']  # (2,256)
    H = metrics['embeddings']  # (B,256)
    B = H.shape[0]

    result['prediction_error'] = F.mse_loss(y, yhat).item()
    # result['cosSIM_y_yhat'] = cosine_similarity_gpu(y, yhat).mean().item()

    if split == 'test':
        return result

    y_dim = y.shape[1]

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
    result['WWT_norm'] = torch.norm(WWT).item()
    del WWT

    # NRC1
    H_np = H.cpu().numpy()
    pca_for_H = PCA(n_components=y_dim)
    try:
        pca_for_H.fit(H_np)
    except Exception as e:
        print(e)
        result['NRC1'] = -0.5
        result['NRC2'] = -0.5
    else:
        H_pca = pca_for_H.components_[:y_dim, :]  # First two principal components

        try:
            inverse_mat = np.linalg.inv(H_pca @ H_pca.T)
        except Exception as e:
            print(e)
            result['NRC1'] = -1
            result['NRC2'] = -1
        else:
            P = H_pca.T @ inverse_mat @ H_pca
            del pca_for_H
            del H_pca
            del inverse_mat
            H_proj_PCA = H_np @ P
            result['NRC1'] = (np.linalg.norm(H_np - H_proj_PCA)**2 / B).item()
            del H_np
            del H_proj_PCA

            P = torch.tensor(P, dtype=torch.float32, device=device)
            W_proj_PCA = W @ P
            result['NRC2'] = torch.norm(W - W_proj_PCA).item()

    # NRC2_old
    try:
        inverse_mat = torch.inverse(W @ W.T)
    except Exception as e:
        print(e)
        result['NRC2_old'] = -1
    else:
        H_proj_W = H @ W.T @ inverse_mat @ W
        result['NRC2_old'] = (torch.norm(H - H_proj_W)**2 / B).item()
        del H_proj_W

    # Projection error with Gram-Schmidt
    U = gram_schmidt(W)
    P_E = torch.mm(U.T, U)
    H_proj = torch.mm(H, P_E)
    # H_projected_E_norm = F.normalize(torch.tensor(H_projected_E).float().to(device), p=2, dim=1)
    result['proj_error_H2W'] = F.mse_loss(H_proj, H).item()
    del H_proj

    if extra_info:
        if extra_info.get('A_case2') is not None:
            WWT = W @ W.T
            WWT_normalized = WWT / torch.norm(WWT)

            A_case2 = extra_info
            A_case2 = torch.tensor(A_case2, dtype=torch.float32, device=device)
            A_case2_normalized = A_case2 / torch.norm(A_case2)
            result['NRC3'] = (torch.norm(WWT_normalized - A_case2_normalized).item()) ** 2
            result['NRC3_unnorm'] = (torch.norm(WWT - A_case2).item()) ** 2

    # # MSE between cosine similarities of embeddings and targets with norm
    # cos_H_norm = cosine_similarity_gpu(H, H)
    # result['cosSIM_H'] = cos_H_norm.fill_diagonal_(float('nan')).nanmean().item()
    # cos_y_norm = cosine_similarity_gpu(y, y)
    # indices = torch.triu_indices(cos_H_norm.size(0), cos_H_norm.size(0), offset=1)
    # cos_H_norm = cos_H_norm[indices[0], indices[1]]
    # cos_y_norm = cos_y_norm[indices[0], indices[1]]
    # result['MSE_cosSIM_y_H_norm'] = F.mse_loss(cos_H_norm, cos_y_norm).item()
    # del cos_H_norm
    # del cos_y_norm

    # # MSE between cosine similarities of embeddings and targets
    # cos_H = torch.mm(H, H.T)
    # cos_y = torch.mm(y, y.T)
    # indices = torch.triu_indices(cos_H.size(0), cos_H.size(0), offset=1)
    # cos_H = cos_H[indices[0], indices[1]]
    # cos_y = cos_y[indices[0], indices[1]]
    # result['MSE_cosSIM_y_H'] = F.mse_loss(cos_H, cos_y).item()
    # del cos_H
    # del cos_y
    # del indices

    # MSE between cosine similarities of PCA embeddings and targets
    # cos_H_pca = torch.mm(H_pca_norm, H_pca_norm.transpose(0, 1))
    # indices = torch.triu_indices(cos_H_pca.size(0), cos_H_pca.size(0), offset=1)
    # upper_tri_embeddings_pca = cos_H_pca[indices[0], indices[1]]
    # result['mse_cos_sim_PCA'] = F.mse_loss(upper_tri_embeddings_pca, upper_tri_targets).item()

    # # Cosine similarity of Y and H with H2W
    # H_coordinates = torch.mm(F.normalize(H), U.T)
    # result['cosSIM_y_H2W'] = F.cosine_similarity(H_coordinates, y).mean().item()

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
        self.size = 0
        self.state_dim = 0
        self.action_dim = 0

        self.states, self.actions = None, None
        self._load_dataset(data_folder, env, split, data_size, normalize)

        self.device = device

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
                    data_size //= 5
                # self.size = data_size * 1000 if env == 'swimmer' else data_size * 50
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

    def get_state_dim(self):
        return self.state_dim

    def get_action_dim(self):
        return self.action_dim

    def get_theory_stats(self):
        Y = self.actions.T
        y_dim = Y.shape[0]
        Y_mean = Y.mean(axis=1, keepdims=True)
        Y = Y - Y_mean
        M = Y.shape[1]
        Sigma = Y @ Y.T / M

        # Sigma_sqrt = sqrtm(Sigma)
        # eig_vals = np.linalg.eigvalsh(Sigma)
        eig_vals, eig_vecs = np.linalg.eigh(Sigma)
        sqrt_eig_vals = np.sqrt(eig_vals)
        Sigma_sqrt = eig_vecs.dot(np.diag(sqrt_eig_vals)).dot(np.linalg.inv(eig_vecs))

        min_eigval = eig_vals[0]
        max_eigval = eig_vals[-1]

        # mu11 = Sigma[0, 0]
        # mu12 = Sigma[0, 1]
        # mu22 = Sigma[1, 1]
        # sqrt = np.sqrt((mu22 - mu11) ** 2 + 4 * mu12 ** 2)
        # gamma1 = (mu22 - mu11 + sqrt) / (2 * mu12)
        # gamma2 = (mu22 - mu11 - sqrt) / (2 * mu12)

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

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        states = self.states[idx]
        actions = self.actions[idx]
        return {
            'states': self._to_tensor(states),
            'actions': self._to_tensor(actions)
        }


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0, arch: str = '256-R-256-R|T'):
        super(Actor, self).__init__()

        arch, use_bias = arch.split('|')
        arch = arch.split('-')
        use_bias = True if use_bias == 'T' else False

        in_dim = state_dim
        module_list = []
        for i, layer in enumerate(arch):
            if layer == 'R':
                module_list.append(nn.ReLU())
            elif layer == 'T':
                module_list.append(nn.Tanh())
            elif layer == 'G':
                module_list.append(nn.GELU())
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
    def NC_eval(self, dataloader, split, extra_info=None):
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
        log_dict = compute_metrics(res, split, self.device, extra_info=extra_info)
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
    actor = Actor(state_dim, action_dim, arch=config.arch).to(config.device)

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
    # val_theory_states = val_dataset.get_theory_stats()
    A_case2 = None
    if config.lamH != -1 and config.lamW != 0:
        for d in [train_theory_stats]:
            A_case2 = (config.lamH / config.lamW) ** 0.5 * Sigma_sqrt - config.lamH * np.eye(Sigma_sqrt.shape[0])
            d['A11'] = A_case2[0, 0]
            d['A22'] = A_case2[1, 1]
            d['A12'] = A_case2[0, 1]

    train_log = trainer.NC_eval(train_loader, split='train', extra_info={'A_case2': A_case2})
    val_log = trainer.NC_eval(val_loader, split='test')
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
            train_log = trainer.NC_eval(train_loader, split='train', extra_info={'A_case2': A_case2})
            val_log = trainer.NC_eval(val_loader, split='test')
            wandb.log({'train_mse_loss': epoch_train_loss,
                       'train': train_log,
                       'validation': val_log,
                       'C': train_theory_stats,
                       })

        if (epoch + 1) in [config.max_epochs // (i+1) for i in range(4)]:
            # Save NRC3 related data to local for later plots
            save_folder = '/NC_regression/results/wwt'
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, config.name + '.pkl')
            with open(save_path, 'wb') as file:
                to_plot_nrc3 = {'WWT': np.concatenate(all_WWT, axis=0),
                                'Sigma_sqrt': Sigma_sqrt,
                                'min_eigval': train_theory_stats['min_eigval']}
                pickle.dump(to_plot_nrc3, file)

    # # Compute residuals
    # if config.env in ['reacher', 'swimmer']:
    #     with torch.no_grad():
    #         actor.eval()
    #         y = torch.empty((0,), device=config.device)
    #         H = torch.empty((0,), device=config.device)
    #         Wh = torch.empty((0,), device=config.device)
    #         W = actor.W.weight.detach().clone()
    #
    #         for i, batch in enumerate(train_loader):
    #             states, actions = batch['states'], batch['actions']
    #             features = actor.get_feature(states)
    #             preds = actor.project(features)
    #
    #             y = torch.cat((y, actions), dim=0)
    #             H = torch.cat((H, features), dim=0)
    #             Wh = torch.cat((Wh, preds), dim=0)
    #
    #     U = gram_schmidt(W)
    #     coeff = H @ U.T
    #     w1 = coeff[:, 0]
    #     w2 = coeff[:, 1]
    #     y1y2 = y[:, 0] / y[:, 1]
    #     data = [[a, b, c] for (a, b, c) in zip(w1, w2, y1y2)]
    #     table = wandb.Table(data=data, columns=["w1", "w2", 'y1/y2'])
    #     try:
    #         wandb.log({'Residual_table': table})
    #     except Exception as e:
    #         print(e)
    #
    #     try:
    #         wandb.log(
    #             {'Residual Plot': wandb.plots.HeatMap(list(w1), list(w2), y1y2.cpu().numpy(), show_text=False)})
    #     except Exception as e:
    #         print(e)

    if config.lamH != -1:
        return

    # NRC3
    W = actor.W.weight.detach().clone().cpu().numpy()
    WWT = W @ W.T
    all_WWT.append(WWT.reshape(1, -1))
    all_WWT = np.concatenate(all_WWT, axis=0)

    # Save NRC3 related data to local for later plots
    save_folder = '/NC_regression/results/wwt'
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, config.name + '.pkl')
    with open(save_path, 'wb') as file:
        to_plot_nrc3 = {'WWT': all_WWT,
                        'Sigma_sqrt': Sigma_sqrt,
                        'min_eigval': train_theory_stats['min_eigval']}
        pickle.dump(to_plot_nrc3, file)

    # Log NRC3 related curves to wandb
    WWT_normalized = WWT / np.linalg.norm(WWT)
    min_eigval = train_theory_stats['min_eigval']
    # Sigma_sqrt = np.array([train_theory_stats[k] for k in ['sigma11', 'sigma12', 'sigma21', 'sigma22']]).reshape(2, 2)

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

    # Save model
    save_folder = f'/NC_regression/models/{config.env}'
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, config.name + '.pth')
    torch.save(actor, save_path)

    # c_to_plot = np.linspace(0.000001, min_eigval, num=1000)
    # lamH_to_plot = np.linspace(0.0001, 0.1, num=1000)
    # NC2_to_plot = []
    # lamW_to_plot = []
    # for lamH in lamH_to_plot:
    #     NC2_row = []
    #     lamW_row = []
    #     for c in c_to_plot:
    #         A = lamH * Sigma_sqrt / (c ** 0.5) - lamH * np.eye(2)
    #         A_normalized = A / np.maximum(np.linalg.norm(A), 1e-6)
    #         diff_mat = WWT_normalized - A_normalized
    #         lamW = c / lamH
    #         NC2_row.append(np.linalg.norm(diff_mat).item())
    #         lamW_row.append(lamW)
    #     idx = np.argmin(NC2_row)
    #     NC2_to_plot.append(NC2_row[idx])
    #     lamW_to_plot.append(lamW_row[idx])
    #
    # data = [[a, b, c] for (a, b, c) in zip(lamH_to_plot, NC2_to_plot, lamW_to_plot)]
    # table = wandb.Table(data=data, columns=["lamH", "NC2", "lamW"])
    # wandb.log(
    #     {
    #         "NC2(c, lamH)": wandb.plot.line(
    #             table, "lamH", "NC2", title="NC2 as a Function of c and lamH"
    #         )
    #     }
    # )

    # c_to_plot = np.linspace(0.000001, min_eigval, num=80)
    # lamH_to_plot = np.linspace(0.0001, 0.1, num=80)
    # NC2_to_plot = []
    # lamW_to_plot = []
    # for c in c_to_plot:
    #     NC2_row = []
    #     lamW_row = []
    #     for lamH in lamH_to_plot:
    #         A = lamH * Sigma_sqrt / (c ** 0.5) - lamH * np.eye(2)
    #         A_normalized = A / np.maximum(np.linalg.norm(A), 1e-6)
    #         diff_mat = WWT_normalized - A_normalized
    #         lamW = c / lamH
    #         NC2_row.append(np.linalg.norm(diff_mat).item())
    #         lamW_row.append(lamW)
    #     NC2_to_plot.append(NC2_row)
    #     lamW_to_plot.append(lamW_row)
    #
    # wandb.log({'NC2(c, lamH)': wandb.plots.HeatMap(list(lamH_to_plot), list(c_to_plot), NC2_to_plot, show_text=False)})
    # wandb.log({'lamW(c, lamH)': wandb.plots.HeatMap(list(lamH_to_plot), list(c_to_plot), lamW_to_plot, show_text=False)})
