import pandas as pd
import json
import os
import wandb

api = wandb.Api()
entity, project = "d_konoki", "NC_new"

name_convertor = {'_step': 'Step',
                  'train.NRC1n': 'NRC1',
                  # 'train.NRC1n_pca2': 'NRC1_pca2',
                  # 'train.NRC1n_pca3': 'NRC1_pca3',
                  'train.NRC2n': 'NRC2',
                  'train.NRC3_ufm': 'NRC3',
                  'train.prediction_error': 'trainError',
                  'validation.prediction_error': 'testError',
                  # 'train.R_sq': 'R_sq',
                  # 'train.EVR1': 'EVR1',
                  # 'train.EVR2': 'EVR2',
                  # 'train.EVR3': 'EVR3',
                  # 'train.EVR4': 'EVR4',
                  # 'train.EVR5': 'EVR5',
                  }
max_epochs = {'reacher': int(1e4), 'swimmer': int(1e4), 'hopper': int(1e4)}
lamW_range = {'reacher': [1e-2, 1e-3, 1e-4, 1e-5, 0],
              'swimmer': [1e-2, 1e-3, 1e-4, 1e-5, 0],
              'hopper': [1e-2, 1e-3, 1e-4, 1e-5, 0]}

for env in ['reacher', 'swimmer', 'hopper']:
    for lamW in lamW_range[env]:
        my_filter = {'config.group': 'bn_case2',
                     'config.env': env,
                     'config.max_epochs': max_epochs[env],
                     'config.mode': 'bn_no_relu',
                     'config.lamW': lamW}

        runs = api.runs(path=entity + "/" + project,
                        filters=my_filter,
                        per_page=50)
        print(f'Load {len(runs)} runs for {env}.')

        for i, run in enumerate(runs):
            config = run.config
            exp_name = config['name'][:-3]
            seed = config['seed']

            if i == 0:
                print(config)
            print(f'Reading experiment: {exp_name} on seed {seed}.')

            return_data = {v: [] for v in name_convertor.values()}

            history = run.scan_history(keys=[k for k in name_convertor.keys()])
            for row in history:
                for k, v in row.items():
                    return_data[name_convertor[k]].append(v)
            df = pd.DataFrame(return_data)

            for table_name in ["NRC3_vs_c_table", "NRC3_vs_epoch_table"]:
                for table in run.logged_artifacts():
                    if table_name not in table.name:
                        continue

                    table_dir = table.download()  # This downloads the table to the same folder and return the path.
                    table_path = f"{table_dir}/{table_name}.table.json"
                    with open(table_path) as file:
                        json_dict = json.load(file)  # A dictionary containing keys: "data", "columns". The values are: List[List[row_data]], List[column_names].
                    df_table = pd.DataFrame(json_dict["data"], columns=json_dict["columns"])

                    if table_name == "NRC3_vs_c_table":
                        df['gamma'] = df_table['c']
                        df['NRC3(gamma)'] = df_table['NRC3']
                    if table_name == "NRC3_vs_epoch_table":
                        df['NRC3'] = df_table['NRC3']

            save_folder = os.path.join('../results/camera_ready/case2', env, exp_name)
            os.makedirs(save_folder, exist_ok=True)
            df.to_csv(os.path.join(save_folder, f'progress_s{seed}.csv'), sep=',', index=False)


