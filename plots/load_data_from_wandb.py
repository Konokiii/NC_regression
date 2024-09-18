import pandas as pd
import os
import wandb

api = wandb.Api()
entity, project = "d_konoki", "NC_rebuttal"

name_convertor = {'_step': 'Step',
                  'train.NRC1_pca2': 'NRC1_pca2',
                  'train.NRC1_pca3': 'NRC1_pca3',
                  'train.NRC2': 'NRC2',
                  'train.prediction_error': 'trainError',
                  'validation.prediction_error': 'testError',
                  'train.R_sq': 'R_sq',
                  'train.EVR1': 'EVR1',
                  'train.EVR2': 'EVR2',
                  'train.EVR3': 'EVR3',
                  'train.EVR4': 'EVR4',
                  'train.EVR5': 'EVR5',}
max_epochs = {'reacher': 3000000, 'swimmer': 3000000, 'hopper': 500000}
for env in ['reacher', 'swimmer']:
    my_filter = {'config.group': 'long',
                 'config.env': env,
                 'config.max_epochs': max_epochs[env],
                 'config.mode': 'null',
                 'config.lamW': 0}

    runs = api.runs(path=entity + "/" + project,
                    filters=my_filter,
                    per_page=50)
    print(f'Load {len(runs)} runs for {env}.')

    for i, run in enumerate(runs):
        config = run.config
        exp_name = config['name'][:-3]
        seed = config['seed']
        print(f'Reading experiment: {exp_name} on seed {seed}.')
        if i == 0:
            print(config)

        history = run.scan_history(keys=[k for k in name_convertor.keys()])
        return_data = {k: [] for k in name_convertor.values()}
        for row in history:
            for k, v in row.items():
                return_data[name_convertor[k]].append(v)

        df = pd.DataFrame(return_data)
        save_folder = os.path.join('../results/rebuttal', env, exp_name)
        os.makedirs(save_folder, exist_ok=True)
        df.to_csv(os.path.join(save_folder, f'progress_s{seed}.csv'), sep=',', index=False)





