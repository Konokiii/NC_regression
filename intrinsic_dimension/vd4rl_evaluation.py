import argparse
import gym
import os
import numpy as np

import random
from typing import Optional
import pickle
import wandb
import uuid
from tqdm.auto import trange

import torch
from dm_env import specs

from agent.vd4rl_utils import make
from agent.vd4rl_buffer import add_offline_data_to_buffer, EfficientReplayBuffer
from intrinsic_dimension.compute_ID import compute_intrinsic_dimension
from agent.vd4rl_bc import Encoder, Actor, eval_actor

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


def to_torch(xs, device):
    return tuple(torch.FloatTensor(x).to(device) for x in xs)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )


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


def extract_features(model, data, policy_layer_name='policy'):
    features = []

    def hook_fn(module, input, output):
        features.append(output.detach())

    # Register forward hook
    handle = None
    for name, module in model.named_modules():
        if policy_layer_name in name:
            handle = module.register_forward_hook(hook_fn)
            break

    if handle is None:
        raise ValueError(f"Policy layer '{policy_layer_name}' not found in model.")

    # Run a forward pass
    _ = model(data)

    # Remove hook
    handle.remove()

    # Return features
    return features[0] if features else None



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
        'env', '', ['cheetah_run'],
        'dataset', '', ['expert'],
        'max_timesteps', '', [int(1e6)],
        'data_size', '', ['all'],
        'actor_wd', '', [0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
        'hidden_layers', '', [3],
        'hidden_dim', '', [256],
        'optimizer', '', ['adam', 'sgd'],
        'eval_freq', '', [int(5e4)],

        'seed', '', [0],
        'group', '', ['vd4rl'],
    ]

    hyper2logname = {'env': 'E',
                     'max_timesteps': 'MaxT',
                     'data_size': 'DS',
                     'actor_wd': 'WD',
                     'hidden_layers': 'Depth',
                     'hidden_dim': 'Width',
                     'optimizer': 'Opt'}

    indexes, config, total, _ = get_setting_dt(settings, setting)
    config['name'] = '_'.join([v + str(getattr(config, k)) for k, v in hyper2logname.items()])
    config['project'] = 'Intrinsic_Dim'
    config['device'] = DEVICE

    env = make(config['env'], 3, 2, config['seed'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    set_seed(config['seed'], env)

    MODEL_FOLDER = f"./vd4rl/models/{config['env']}/{config['name']}"
    TRAIN_DATA_PATH = f"./vd4rl/dataset/{config['env']}/expert/84px/{config['env']}_expert_train.pkl"
    TEST_DATA_PATH = f"./vd4rl/dataset/{config['env']}/expert/84px/{config['env']}_expert_test.pkl"

    data_specs = (env.observation_spec(),
                  env.action_spec(),
                  specs.Array((1,), np.float32, 'reward'),
                  specs.Array((1,), np.float32, 'discount'))

    with open(TRAIN_DATA_PATH, 'rb') as f:
        train_dataset = pickle.load(f)
        train_dataset['action'] = np.arctanh(np.clip(train_dataset['action'], a_min=-1+1e-7, a_max=1-1e-7))
        TRAIN_DATA_SIZE = train_dataset['reward'].shape[0]

    replay_buffer_train = EfficientReplayBuffer(
        100000,
        TRAIN_DATA_SIZE,
        3,
        0.99,
        3,
        data_specs=data_specs,
        # sarsa=True
    )

    add_offline_data_to_buffer(train_dataset, replay_buffer_train, framestack=3)

    print('Computing train target ID...')
    train_target_ID = compute_intrinsic_dimension(to_torch(next(replay_buffer_train), DEVICE)[1], batch_size=1000, epsilon=0.)
    print('Finished computing train target ID!')

    with open(TEST_DATA_PATH, 'rb') as f:
        test_dataset = pickle.load(f)
        test_dataset['action'] = np.arctanh(np.clip(test_dataset['action'], a_min=-1 + 1e-7, a_max=1 - 1e-7))
        TEST_DATA_SIZE = test_dataset['reward'].shape[0]

    replay_buffer_test = EfficientReplayBuffer(
        100000,
        TEST_DATA_SIZE,
        3,
        0.99,
        3,
        data_specs=data_specs,
        # sarsa=True
    )

    add_offline_data_to_buffer(test_dataset, replay_buffer_test, framestack=3)

    print('Computing tset target ID...')
    test_target_ID = compute_intrinsic_dimension(to_torch(next(replay_buffer_test), DEVICE)[1], batch_size=1000, epsilon=0.)
    print('Finished computing test target ID!')

    wandb_init(config)

    for epoch in trange(config['max_timesteps'], desc='Timesteps'):
        if epoch == 0 or (epoch + 1) % config['eval_freq'] == 0:
            print('===================================================')
            model_name = f'checkpoint_{epoch + 1 if epoch > 0 else 0}.pt'
            model_path = os.path.join(MODEL_FOLDER, model_name)
            print('Loading model from the path: ', model_path)
            model_log = torch.load(model_path, map_location=config['device'])
            model_info = model_log['config']
            encoder_state_dict = model_log['encoder']
            actor_state_dict = model_log['actor']
            encoder = Encoder((9, 84, 84)).to(config['device'])
            actor = Actor(encoder.repr_dim, action_dim, max_action=1, hidden_dim=config['hidden_dim']).to(config['device'])
            encoder.load_state_dict(encoder_state_dict)
            actor.load_state_dict(actor_state_dict)
            print('Model sucessfully loaded.')
            print('===================================================')

            # ID evaluation
            with torch.no_grad():
                X_train, Y_train = list(to_torch(next(replay_buffer_train), DEVICE))
                X_train = encoder(X_train)
                X_test, Y_test = list(to_torch(next(replay_buffer_test), DEVICE))
                X_test = encoder(X_test)

                H_train = extract_features(actor, X_train, policy_layer_name='policy')
                train_feature_ID = compute_intrinsic_dimension(H_train, batch_size=1000, epsilon=1e-6)
                NRC1_related = compute_NRC1(H_train, target_dim=action_dim)
                Y_train_hat = actor.W(H_train)
                train_mse = torch.sum((Y_train - Y_train_hat) ** 2).item() / Y_train.shape[0]

                H_test = extract_features(actor, X_test, policy_layer_name='policy')
                test_feature_ID = compute_intrinsic_dimension(H_test, batch_size=1000, epsilon=1e-6)
                Y_test_hat = actor.W(H_test)
                test_mse = torch.sum((Y_test - Y_test_hat) ** 2).item() / Y_test.shape[0]

            # RL evaluation
            eval_scores = eval_actor(
                env,
                actor=actor,
                encoder=encoder,
                device=DEVICE,
                n_episodes=10,
                seed=config['seed'],
            )
            normalized_eval_scores = eval_scores / 10

            log_metrics = {"test_mse": test_mse,
                           "test_feature_id": test_feature_ID,
                           "test_target_id": test_target_ID,
                           "train_mse": train_mse,
                           "train_feature_id": train_feature_ID,
                           "train_target_id": train_target_ID,
                           "rl_scores": normalized_eval_scores.mean()}
            log_metrics.update(NRC1_related)
            wandb.log(log_metrics, step=epoch)

    print('Full experiment config for the last checkpoint:', model_info, sep='\n')