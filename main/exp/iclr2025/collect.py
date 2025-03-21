import os
import torch
import argparse

from main.BC import TrainConfig, run_BC
from main.utils import get_setting_dt

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
        {'env': 'reacher',
         'max_epochs': int(2e5),
         'data_size': int(1e4),
         'lamW': 1.5e-3,
         'eval_freq': int(1e3)
         },
        {'env': 'reacher',
         'max_epochs': int(2e5),
         'data_size': int(2e4),
         'lamW': 1.5e-3,
         'eval_freq': int(1e3)
         },
        {'env': 'reacher',
         'max_epochs': int(2e5),
         'data_size': int(5e4),
         'lamW': 1.5e-3,
         'eval_freq': int(1e3)
         },

        {'env': 'swimmer',
         'max_epochs': int(2e5),
         'data_size': int(1e4),
         'lamW': 1e-2,
         'eval_freq': int(1e3)
         },
        {'env': 'swimmer',
         'max_epochs': int(2e5),
         'data_size': int(2e4),
         'lamW': 1e-2,
         'eval_freq': int(1e3)
         },
        {'env': 'swimmer',
         'max_epochs': int(2e5),
         'data_size': int(5e4),
         'lamW': 1e-2,
         'eval_freq': int(1e3)
         },

        {'env': 'hopper',
         'max_epochs': int(2e5),
         'data_size': int(1e4),
         'lamW': 1e-2,
         'eval_freq': int(1e3)
         },
        {'env': 'hopper',
         'max_epochs': int(2e5),
         'data_size': int(2e4),
         'lamW': 1e-2,
         'eval_freq': int(1e3)
         },
        {'env': 'hopper',
         'max_epochs': int(2e5),
         'data_size': int(5e4),
         'lamW': 1e-2,
         'eval_freq': int(1e3)
         },
    ]

    # indexes, actual_setting, total, hyper2logname = get_setting_dt(settings, setting)

    # exp_name_full = get_auto_exp_name(actual_setting, hyper2logname, exp_prefix='')
    # config = pyrallis.load(TrainConfig, '/configs/offline/iql/%s/%s_v2.yaml'
    #                        % (actual_setting['env'], actual_setting['dataset'].replace('-', '_')))

    """replace values"""
    config = TrainConfig(**settings[setting])
    config.device = DEVICE
    config.batch_size = 2048

    config.data_folder = './dataset'
    config.project_folder = './'
    config.project = 'NC_new'
    config.group = 'test'

    hyper2logname = {'env': 'E',
                     'max_epochs': 'MaxEp',
                     'data_size': 'DS',
                     'lamW': 'WD'}
    config.name = '_'.join([v + str(getattr(config, k)) for k, v in hyper2logname.items()])

    run_BC(config)


if __name__ == '__main__':
    main()
