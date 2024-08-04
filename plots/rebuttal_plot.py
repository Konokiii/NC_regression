import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import seaborn as sns
import pandas as pd
import csv

import numpy as np
import torch

from log_alias import *

DATA_FOLDER = '../results/rebuttal'
N_BOOT = 3
LEGEND_FONTSIZE = 8


def load_data_from_csv(data_folder, dataset, variant):
    data = {}
    root_path = os.path.join(data_folder, dataset, variant)
    for f in os.listdir(root_path):
        path = os.path.join(root_path, f)
        if not os.path.isfile(path):
            continue

        if '.csv' in f and 'progress_combined' in f:
            with open(path, 'r') as csv_file:
                df = pd.read_csv(csv_file, delimiter=",", header=0)
                for col in df.columns:
                    if col == 'Step':
                        data[col] = np.concatenate([data.get(col, []), df[col].to_numpy() * 100])
                    else:
                        data[col] = np.concatenate([data.get(col, []), df[col].to_numpy()])
        elif '.csv' in f and 'gamma' in f:
            raise NotImplementedError

    return data


def plot_different_metrics(ax, y_metrics_dict, data, x_metric='Step'):  # metrics_dict, key=metric_name, value=legend
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, num=len(y_metrics_dict)))[::-1]
    x_to_plot = data[x_metric]
    for i, m in enumerate(y_metrics_dict.keys()):
        y_to_plot = data[m]
        sns.lineplot(ax=ax, x=x_to_plot, y=y_to_plot, n_boot=N_BOOT, color=colors[i], label=y_metrics_dict[m])


def plot_different_variants(ax, data_dict, y_metric, x_metrix='Step'):  # data_dict, key=legend, value=full_data
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, num=len(data_dict)))[::-1]
    for i, variant in enumerate(data_dict.keys()):
        x_to_plot = data_dict[variant][x_metrix]
        y_to_plot = data_dict[variant][y_metric]
        sns.lineplot(ax=ax, x=x_to_plot, y=y_to_plot, n_boot=N_BOOT, color=colors[i], label=variant)


def figure2():
    data_plot = {}
    for dataset, variant in {'reacher': reacher, 'swimmer': swimmer, 'hopper': hopper}.items():
        data_plot[dataset] = load_data_from_csv(DATA_FOLDER, dataset, variant)

    fig = plt.figure(figsize=(10, 8))

    gs = fig.add_gridspec(2, 1, hspace=0.3)  # spacing between the two groups

    gs0 = gs[0].subgridspec(2, 3, hspace=0.2)  # spacing within the groups
    gs1 = gs[1].subgridspec(2, 3, hspace=0.2)

    data_name = {'reacher': "Reacher", 'swimmer': 'Swimmer', 'hopper': "Hopper", 'carla2d': 'Carla 2D',
                 'carla1d': 'Carla 1D', 'age': 'UTKFace'}

    for i, dataset in enumerate(['reacher', 'swimmer', 'hopper']):
        print(f'Plotting figure 2 for {dataset}.')

        # ==== first plot nrc
        row = 0 if dataset in ['reacher', 'swimmer', 'hopper'] else 2
        if dataset in ['reacher', 'swimmer', 'hopper']:
            ax = fig.add_subplot(gs0[0, i % 3])
        else:
            ax = fig.add_subplot(gs1[0, i % 3])

        if dataset in ['reacher', 'swimmer', 'hopper', 'carla2d']:
            sns.lineplot(ax=ax, x=data_plot[dataset]['Step'], y=data_plot[dataset]['NRC1'], n_boot=N_BOOT, color='C0',
                         label='NRC1')
            print('NRC1 DONE')
            sns.lineplot(ax=ax, x=data_plot[dataset]['Step'], y=data_plot[dataset]['NRC2'], n_boot=N_BOOT, color='C0',
                         label='NRC2',
                         linestyle='--')
            print('NRC2 DONE')
            ax.tick_params(axis='y', colors='C0')
            ax.set_ylabel('NRC1/NRC2', color='C0')

            ax2 = ax.twinx()
            sns.lineplot(ax=ax2, x=data_plot[dataset]['Step'], y=data_plot[dataset]['NRC3'], n_boot=N_BOOT, color='C1',
                         label='NRC3')
            print('NRC3 DONE')
            ax2.tick_params(axis='y', colors='C1')
            ax2.set_ylabel('NRC3', color='C1')

            handles, labels = [], []
            for a in [ax, ax2]:
                for h, l in zip(*a.get_legend_handles_labels()):
                    handles.append(h)
                    labels.append(l)
            ax.legend(handles, labels, fontsize=LEGEND_FONTSIZE)
            ax2.legend().set_visible(False)
            ax.grid(True, linestyle='--')

            if dataset == 'reacher':
                ax.xaxis.set_ticks([0, int(2.5e5), int(5e5), int(7.5e5),int(1e6)])
                ax.set_xticklabels([0, '250K', '500K', '750K', '1M'])
            if dataset == 'swimmer':
                ax.xaxis.set_ticks([0, int(2.5e5), int(5e5), int(7.5e5), int(1e6)])
                ax.set_xticklabels([0, '250K', '500K', '750K', '1M'])
            if dataset == 'hopper':
                ax.xaxis.set_ticks([0, int(0.5e5), int(1e5), int(1.5e5), int(2e5)])
                ax.set_xticklabels([0, '50K', '100K', '150K', '200K'])

        else:
            sns.lineplot(ax=ax, x=data_plot[dataset]['Step'], y=data_plot[dataset]['NRC1'], n_boot=N_BOOT, color='C0',
                         label='NRC1')
            print('NRC1 DONE')
            sns.lineplot(ax=ax, x=data_plot[dataset]['Step'], y=data_plot[dataset]['NRC2'], n_boot=N_BOOT, color='C0',
                         label='NRC2',
                         linestyle='--')
            print('NRC2 DONE')
            ax.set_ylabel('NRC1/NRC2')
            ax.grid(True, linestyle='--')

        ax.set_title(data_name[dataset], fontweight='bold')

        # ==== then plot Train and test MSE
        row = 1 if dataset in ['reacher', 'swimmer', 'hopper'] else 3
        if dataset in ['reacher', 'swimmer', 'hopper']:
            ax = fig.add_subplot(gs0[1, i % 3])
        else:
            ax = fig.add_subplot(gs1[1, i % 3])
        sns.lineplot(ax=ax, x=data_plot[dataset]['Step'], y=data_plot[dataset]['trainError'], n_boot=N_BOOT, color='C0',
                     label='Train MSE')
        print('Train MSE done')
        sns.lineplot(ax=ax, x=data_plot[dataset]['Step'], y=data_plot[dataset]['testError'], n_boot=N_BOOT, color='C1',
                     label='Test MSE')
        print('Test MSE done')

        ax.legend(fontsize=LEGEND_FONTSIZE)
        ax.set_ylabel('MSE')
        ax.grid(True, linestyle='--')

        if dataset == 'reacher':
            ax.xaxis.set_ticks([0, int(2.5e5), int(5e5), int(7.5e5), int(1e6)])
            ax.set_xticklabels([0, '250K', '500K', '750K', '1M'])
        if dataset == 'swimmer':
            ax.xaxis.set_ticks([0, int(2.5e5), int(5e5), int(7.5e5), int(1e6)])
            ax.set_xticklabels([0, '250K', '500K', '750K', '1M'])
        if dataset == 'hopper':
            ax.xaxis.set_ticks([0, int(0.5e5), int(1e5), int(1.5e5), int(2e5)])
            ax.set_xticklabels([0, '50K', '100K', '150K', '200K'])

        if dataset == 'carla1d':
            # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.set_xlabel('Epoch')

    for i in range(2, 4):
        for j in range(3):
            gs.update(bottom=0.1)

    positions_row_3_4 = [ax.get_position() for ax in fig.axes if ax.get_subplotspec().rowspan.start >= 2]

    # Move all subplots in rows 3 and 4 down
    for pos in positions_row_3_4:
        pos.y0 -= 0.2  # Move down by 0.2
        pos.y1 -= 0.2  # Move down by 0.2

    plt.show()
    plt.close()


def figure4():
    variants = {'reacher': [reacher_wd0, reacher_wd5e6, reacher_wd5e5, reacher_wd5e4,
                            reacher_wd15e3],
                'swimmer': [swimmer_wd0, swimmer_wd5e5, swimmer_wd5e4, swimmer_wd5e3,
                            swimmer_wd1e2],
                'hopper': [hopper_wd0, hopper_wd5e6, hopper_wd5e5, hopper_wd5e4,
                           hopper_wd1e3]}
    legends = {'reacher': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=5e-6$', r'$\lambda_{WD}=5e-5$', r'$\lambda_{WD}=5e-4$',
                           r'$\lambda_{WD}=1.5e-3$'],
               'swimmer': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=5e-5$', r'$\lambda_{WD}=5e-4$', r'$\lambda_{WD}=5e-3$',
                           r'$\lambda_{WD}=1e-2$'],
               'hopper': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=5e-6$', r'$\lambda_{WD}=5e-5$', r'$\lambda_{WD}=5e-4$',
                          r'$\lambda_{WD}=1e-3$']}
    metrics = {'trainError': 'Train MSE', 'testError': 'Test MSE', 'R_sq': r'$\mathbf{R^2}$', 'NRC1': 'NRC1', 'NRC2': 'NRC2', 'NRC3': 'NRC3'}
    titles = {'reacher': 'Reacher', 'swimmer': 'Swimmer', 'hopper': 'Hopper'}
    num_rows = len(variants)
    num_columns = len(metrics)
    fig, axes = plt.subplots(num_rows, num_columns)
    # cmap = plt.get_cmap('viridis')
    # colors = cmap(np.linspace(0, 1, num=5))[::-1]
    for i, dataset in enumerate(variants.keys()):
        data_dict = {}
        for k, variant in enumerate(variants[dataset]):
            data_dict[legends[dataset][k]] = load_data_from_csv(DATA_FOLDER, dataset, variant)

        for j, metric in enumerate(metrics.keys()):
            print(f'Plotting figure 4 for {dataset} w.r.t {metric}')
            ax = axes[i][j] if num_rows > 1 else axes[j]  # When num_rows = 1, ax is 1D array.

            plot_different_variants(ax, data_dict, metric)

            # ax.xaxis.set_ticks([0, int(2e5), int(4e5), int(6e5), int(8e5)])
            # ax.set_xticklabels([0, '200K', '400K', '600K', '800K'])
            # if dataset == 'carla2d':
            #     ax.set_yscale('log')
            # ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

            ax.grid(True, linestyle='dashed')

            if j + 1 == len(metrics):
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            else:
                ax.legend().set_visible(False)

            if j == 0:
                ax.set_ylabel(titles[dataset], fontweight='bold')
            if i + 1 == num_rows:
                ax.set_xlabel('Epoch', fontweight=None)
            if i == 0:
                ax.set_title(metrics[metric], fontweight='bold')

    plt.tight_layout()
    plt.tight_layout()
    plt.show()


def nrc1_def():
    datasets = {'reacher': reacher_wd15e3, 'swimmer': swimmer_wd1e2, 'hopper': hopper_wd1e3}
    metrics = {'NRC1_pca1': r'$\mathbf{H}_{PCA_1}$', 'NRC1_pca2': r'$\mathbf{H}_{PCA_2}$', 'NRC1_pca3': r'$\mathbf{H}_{PCA_3}$',
               'NRC1_pca4': r'$\mathbf{H}_{PCA_4}$', 'NRC1_pca5': r'$\mathbf{H}_{PCA_5}$'}
    num_columns = len(datasets)
    fig, axes = plt.subplots(1, num_columns)
    for i, dataset in enumerate(datasets.keys()):
        print(f'Plotting NRC1_pca for {dataset}.')
        ax = axes[i]
        data = load_data_from_csv(DATA_FOLDER, dataset, datasets[dataset])
        plot_different_metrics(ax, metrics, data)
        ax.grid(True, linestyle='dashed')
        if i != 0:
            ax.legend().set_visible(False)
        else:
            ax.legend(fontsize=LEGEND_FONTSIZE)
        if i == 0:
            ax.set_ylabel('NRC1', fontweight='bold')
        ax.set_xlabel('Epoch', fontweight=None)
        ax.set_title(dataset.capitalize(), fontweight='bold')
    plt.tight_layout()
    plt.tight_layout()
    plt.show()


def EVR():
    datasets = {'reacher': reacher_wd15e3, 'swimmer': swimmer_wd1e2, 'hopper': hopper_wd1e3}
    metrics = {'EVR1': 'PC1', 'EVR2': 'PC2', 'EVR3': 'PC3', 'EVR4': 'PC4', 'EVR5': 'PC5'}
    num_columns = len(datasets)
    fig, axes = plt.subplots(1, num_columns)
    for i, dataset in enumerate(datasets.keys()):
        print(f'Plotting EVR for {dataset}')
        ax = axes[i]
        data = load_data_from_csv(DATA_FOLDER, dataset, datasets[dataset])
        plot_different_metrics(ax, metrics, data)
        ax.grid(True, linestyle='dashed')
        if i != 0:
            ax.legend().set_visible(False)
        else:
            ax.legend(fontsize=LEGEND_FONTSIZE)
        if i == 0:
            ax.set_ylabel('Explained Variance Ratio', fontweight='bold')
        ax.set_xlabel('Epoch', fontweight=None)
        ax.set_title(dataset.capitalize(), fontweight='bold')
    plt.tight_layout()
    plt.tight_layout()
    plt.show()


figure2()
# figure4()
# nrc1_def()
# EVR()
