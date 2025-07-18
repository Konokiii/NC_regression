import os
import torch
import argparse

from agent.vd4rl_bc import TrainConfig, run_vd4rl_bc
from agent.utils import get_setting_dt

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


def main():
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
        'eval_freq', '', [int(5e4)]
    ]

    indexes, actual_setting, total, _ = get_setting_dt(settings, setting)

    """replace values"""
    config = TrainConfig(**actual_setting)
    config.device = DEVICE
    config.vd4rl_path = "./vd4rl"
    config.encoder_wd = config.actor_wd
    config.batch_size = 256

    config.project = 'NC_new'
    config.group = 'vd4rl'

    hyper2logname = {'env': 'E',
                     'max_timesteps': 'MaxT',
                     'data_size': 'DS',
                     'actor_wd': 'WD',
                     'hidden_layers': 'Depth',
                     'hidden_dim': 'Width',
                     'optimizer': 'Opt'}
    config.name = '_'.join([v + str(getattr(config, k)) for k, v in hyper2logname.items()])

    run_vd4rl_bc(config)


if __name__ == '__main__':
    main()

