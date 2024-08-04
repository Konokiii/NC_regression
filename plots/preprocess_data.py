import pandas as pd
import os

name_convertor = {'train.NRC1_pca1': 'NRC1_pca1',
                  'train.NRC1_pca2': 'NRC1_pca2',
                  'train.NRC1_pca3': 'NRC1_pca3',
                  'train.NRC1_pca4': 'NRC1_pca4',
                  'train.NRC1_pca5': 'NRC1_pca5',
                  'train.NRC2': 'NRC2',
                  'train.prediction_error': 'trainError',
                  'validation.prediction_error': 'testError',
                  'train.R_sq': 'R_sq',
                  'train.EVR1': 'EVR1',
                  'train.EVR2': 'EVR2',
                  'train.EVR3': 'EVR3',
                  'train.EVR4': 'EVR4',
                  'train.EVR5': 'EVR5',}

for root, subdirs, files in os.walk('../results/rebuttal'):
    if not subdirs:
        print(root)
        for s in range(3):
            with open(os.path.join(root, f'nrc3_s{s}.csv'), 'r') as f:
                df = pd.read_csv(f, header=0, delimiter=',')
                exp_name = df['name'][0]
                nrc3 = df['NRC3']
            print('exp_name:', exp_name)
            with open(os.path.join(root, f'progress_s{s}.csv'), 'r') as f:
                df = pd.read_csv(f, header=0, delimiter=',')
                columns_to_delete = [col for col in df.columns if 'MIN' in col or 'MAX' in col]
                df.drop(columns=columns_to_delete, inplace=True)

                df.rename(columns={exp_name + ' - ' + k: v for k, v in name_convertor.items()}, inplace=True)

                df['NRC3'] = nrc3
                if 'hopper' in str(root):
                    df['NRC1'] = df['NRC1_pca3']
                else:
                    df['NRC1'] = df['NRC1_pca2']

            df.to_csv(os.path.join(root, f'progress_combined_s{s}.csv'), index=False)



