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

# for root, subdirs, files in os.walk('../results/rebuttal'):
#     if any(d in root for d in ['carla1d', 'carla2d', 'utkface']):
#         continue
#     if not subdirs:
#         print(root)
#         for s in range(3):
#             with open(os.path.join(root, f'nrc3_s{s}.csv'), 'r') as f:
#                 df = pd.read_csv(f, header=0, delimiter=',')
#                 nrc3 = df['NRC3']
#             with open(os.path.join(root, f'progress_s{s}.csv'), 'r') as f:
#                 df = pd.read_csv(f, header=0, delimiter=',')
#                 df['NRC3'] = nrc3
#                 if 'hopper' in str(root):
#                     df['NRC1'] = df['NRC1_pca3']
#                 else:
#                     df['NRC1'] = df['NRC1_pca2']
#
#             save_to = os.path.join(root, f'progress_combined_s{s}.csv')
#             if os.path.exists(save_to):
#                 print(f'The data file exists in {save_to}')
#             else:
#                 df.to_csv(save_to, index=False)

# for root, subdirs, files in os.walk('../results/rebuttal'):
#     if root == '../results/rebuttal/reacher/Ereacher_Mnull_Eps6000000_DS1000_wd0' or \
#             root == '../results/rebuttal/swimmer/Eswimmer_Mnull_Eps6000000_DS1000_wd0':
#         for s in range(3):
#             with open(os.path.join(root, f'progress_combined_s{s}-2.csv'), 'r') as f:
#                 df_former = pd.read_csv(f, header=0, delimiter=',')
#             with open(os.path.join(root, f'progress_combined_s{s}.csv'), 'r') as f:
#                 df_latter = pd.read_csv(f, header=0, delimiter=',')
#             df_latter['Step'] = df_latter['Step'] + len(df_former)
#             df_combined = pd.concat([df_former, df_latter], ignore_index=True)
#             df_combined.to_csv(os.path.join(root, f'progress_combined_s{s}-3.csv'), index=False)
