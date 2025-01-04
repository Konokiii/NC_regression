import pandas as pd
import os

# name_convertor = {'mse_train': 'trainError',
#                   'mse_val': 'testError',
#                   'nc1n': 'NRC1',
#                   'nc2n': 'NRC2',
#                   'nc3': 'NRC3'}
#
# seed = 2023
# root_path = f'../results/camera_ready/case1/carla2d/ufm/seed{seed}'

# # Delete MAX and MIN columns:
# for f in os.listdir(root_path):
#     f_path = os.path.join(root_path, f)
#     if 'EVR' in f or '.csv' not in f:
#         continue
#
#     print(f_path)
#     with open(f_path, 'r') as file:
#         df = pd.read_csv(file, header=0, delimiter=',')
#
#     column_names = df.columns.tolist()
#     col_to_del = [col for col in column_names if 'MAX' in col or 'MIN' in col]
#     df.drop(col_to_del, axis=1, inplace=True)
#
#     df.to_csv(f_path, index=False)
#
# # Remove Unnamed columns; and rename columns:
# for f in os.listdir(root_path):
#     f_path = os.path.join(root_path, f)
#     if 'EVR' in f or '.csv' not in f:
#         continue
#
#     print(f_path)
#     with open(f_path, 'r') as file:
#         df = pd.read_csv(file, header=0, delimiter=',')
#
#     col_to_del = [col for col in df.columns.tolist() if 'Unnamed' in col]
#     df.drop(col_to_del, axis=1, inplace=True)
#
#     column_names = df.columns.tolist()
#     name_mapping = {}
#     for col in column_names:
#         if 'res18_WD' not in col:
#             continue
#         wd_idx = col.find('WD')
#         name_mapping[col] = col[wd_idx:wd_idx+6]
#     df.rename(columns=name_mapping, inplace=True)
#
#     save_name = f
#     for i in name_convertor.keys():
#         if i in f:
#             save_name = f[:16]+name_convertor[i]+'.csv'
#
#     df.to_csv(os.path.join(root_path, save_name), index=False)

# # Group statistics of the same weight decay in one csv file:
# for wd in ['WD5e-2', 'WD1e-1', 'WD5e-3', 'WD1e-2', 'WD1e-3', 'WD5e-4', 'WD1e-4']:
#     data = {}
#     for f in os.listdir(root_path):
#         f_path = os.path.join(root_path, f)
#         if 'EVR' in f or not os.path.isfile(f_path) or '.csv' not in f:
#             continue
#
#         print(f_path)
#         col_name = f[16:][:-4]
#         with open(f_path, 'r') as file:
#             df = pd.read_csv(file, header=0, delimiter=',')
#         if 'Step' not in data.keys():
#             data['Step'] = df['Step']
#         data[col_name] = df[wd]
#
#     df_new = pd.DataFrame(data)
#     os.makedirs(os.path.join(root_path, wd), exist_ok=True)
#     save_path = os.path.join(root_path, wd, f'progress_s{seed}.csv')
#     df_new.to_csv(save_path, index=False)

# # Include EVR and NRC3_gamma
# for root, dirs, files in os.walk(root_path):
#     for subdir in dirs:
#         path = os.path.join(root, subdir)
#         for f in os.listdir(path):
#             if 'EVR' in f or 'gamma' in f:
#                 with open(os.path.join(path, f), 'r') as file:
#                     df1 = pd.read_csv(file, header=0, sep=',')
#                 with open(os.path.join(path, f'progress_s{seed}.csv'), 'r') as file:
#                     df2 = pd.read_csv(file, header=0, sep=',')
#                 df_combined = pd.concat([df1, df2.loc[:, ~df2.columns.isin(df1.columns)]], axis=1)
#                 df_combined.to_csv(os.path.join(path, f'progress_s{seed}.csv'), index=False)


# Add R^2 for Carla:
root_path = '../results/camera_ready/case1/carla2d'
for root, dirs, files in os.walk(root_path):
    for subdir in dirs:
        path = os.path.join(root, subdir)
        for f in os.listdir(path):
            with open(os.path.join(path, f), 'r') as file:
                df = pd.read_csv(file)
            mse = df['trainError'].to_numpy()
            df['R_sq'] = 1 - 2 * mse / (528 + 0.02)
            df.to_csv(os.path.join(path, f), index=False)

