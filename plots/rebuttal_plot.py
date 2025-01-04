import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd
import csv

import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch

from log_alias import *

DATA_FOLDER_CASE1 = '../results/camera_ready/case1'
DATA_FOLDER_CASE2 = '../results/camera_ready/case2'
N_BOOT = 20
FONT_SIZE = 10
LEGEND_FONTSIZE = 8
MUJOCO_SKIP = 50
MUJOCO_SKIP_HOPPER = 10
MUJOCO_SKIP_CASE2 = 5
CARLA_SKIP = 1
SMOOTH = 0.01
FONT_FAMILY = 'monospace'
# TODO: Different experiments use different number of seeds. Should set it here.
# NUM_SEEDS_CARLA = 2
# NUM_SEEDS = 3


# def do_smooth(x, smooth):
#     y = np.ones(smooth)
#     z = np.ones((x.shape[1]))  # Length should be the number of columns
#     smoothed_matrix = np.zeros_like(x)  # Initialize a matrix to store the result
#
#     for i in range(x.shape[0]):  # Loop through each row
#         smoothed_matrix[i, :] = np.convolve(x[i, :], y, 'same') / np.convolve(z, y, 'same')
#
#     return smoothed_matrix

def do_smooth(matrix, sigma):
    smoothed_matrix = np.zeros_like(matrix)  # Initialize a matrix to store the result

    for i in range(matrix.shape[0]):  # Loop through each row
        smoothed_matrix[i, :] = gaussian_filter1d(matrix[i, :], sigma)

    return smoothed_matrix


def load_data_from_csv(data_folder, dataset, variant):
    data = {}
    root_path = os.path.join(data_folder, dataset, variant)
    for f in os.listdir(root_path):
        path = os.path.join(root_path, f)
        if not os.path.isfile(path):
            continue

        if dataset in ['reacher', 'swimmer', 'hopper']:
            if '.csv' in f and 'progress' in f:
                with open(path, 'r') as csv_file:
                    df = pd.read_csv(csv_file, delimiter=",", header=0)
                for col in df.columns:
                    if col == 'Step' and data_folder == DATA_FOLDER_CASE1:  # adjust eval_freq for mujoco
                        data[col] = np.concatenate([data.get(col, []), df[col].to_numpy() * 100])
                    elif col == 'Step' and data_folder == DATA_FOLDER_CASE2:  # adjust eval_freq for mujoco
                        data[col] = np.concatenate([data.get(col, []), df[col].to_numpy() * 10])
                    else:
                        data[col] = np.concatenate([data.get(col, []), df[col].to_numpy()])
        elif dataset in ['utkface', 'carla1d', 'carla2d']:
            if '.csv' in f and 'progress' in f:
                with open(path, 'r') as csv_file:
                    df = pd.read_csv(csv_file, delimiter=",", header=0)
                for col in df.columns:
                    if col == 'Epoch':
                        data['Step'] = np.concatenate([data.get('Step', []), df[col].to_numpy()])
                    else:
                        data[col] = np.concatenate([data.get(col, []), df[col].to_numpy()])
            elif dataset == 'carla2d' and f == 'gamma_vs_nrc3.csv':
                with open(path, 'r') as csv_file:
                    df = pd.read_csv(csv_file, delimiter=",", header=0)
                data['NRC3(gamma)'] = np.concatenate([df['rs0'].to_numpy(), df['rs1'].to_numpy()])
                lambda_min = 0.156
                num_datapoint = df['rs0'].to_numpy().shape[0]
                gamma = np.linspace(0, lambda_min, num_datapoint)
                data['gamma'] = np.concatenate([gamma, gamma])

    # post-process the loaded data dictionary
    if dataset in ['reacher', 'swimmer', 'hopper'] and data_folder == DATA_FOLDER_CASE1:
        for k, v in data.items():
            if k in ['NRC3(gamma)', 'gamma']:
                continue
            if dataset in ['reacher', 'swimmer']:
                data[k] = data[k].reshape(3, -1)[:, ::MUJOCO_SKIP].flatten()  # Too many points in plots.
            else:
                data[k] = data[k].reshape(3, -1)[:, ::MUJOCO_SKIP_HOPPER].flatten()

    if dataset in ['reacher', 'swimmer', 'hopper'] and data_folder == DATA_FOLDER_CASE2:
        for k, v in data.items():
            if k in ['NRC3(gamma)', 'gamma']:
                continue
            data[k] = data[k].reshape(3, -1)[:, ::MUJOCO_SKIP_CASE2].flatten()  # Too many points in plots.

    if dataset in ['carla1d', 'carla2d']:
        for k, v in data.items():
            if k in ['NRC3(gamma)', 'gamma']:
                continue
            smooth_data = do_smooth(data[k].reshape(2, -1), SMOOTH)
            data[k] = smooth_data.flatten()
            data[k] = data[k].reshape(2, -1)[:, ::CARLA_SKIP].flatten()  # Too many points in plots.

    return data


def add_headers(fig, *, row_headers=None, col_headers=None, row_pad=1, col_pad=5, rotate_row_headers=True,
                **text_kwargs):
    # Based on https://stackoverflow.com/a/25814386

    axes = fig.get_axes()
    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )


def plot_different_metrics(ax, y_metrics_dict, data, x_metric='Step',
                           reverse_color=False):  # metrics_dict, key=metric_name, value=legend
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, num=len(y_metrics_dict)))[::-1 if reverse_color else 1]
    x_to_plot = data[x_metric]
    for i, m in enumerate(y_metrics_dict.keys()):
        y_to_plot = data[m]
        sns.lineplot(ax=ax, x=x_to_plot, y=y_to_plot, n_boot=N_BOOT, color=colors[i], label=y_metrics_dict[m])


def plot_different_variants(ax, data_dict, y_metric, x_metrix='Step',
                            reverse_color=False):  # data_dict, key=legend, value=full_data
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, num=len(data_dict)))[::-1 if reverse_color else 1]
    for i, variant in enumerate(data_dict.keys()):
        x_to_plot = data_dict[variant][x_metrix]
        y_to_plot = data_dict[variant][y_metric]
        sns.lineplot(ax=ax, x=x_to_plot, y=y_to_plot, n_boot=N_BOOT, color=colors[i], label=variant)


def figure_main_result():
    data_plot = {}
    for dataset, variant in {'reacher': reacher, 'swimmer': swimmer, 'hopper': hopper, 'carla1d': carla1d,
                             'carla2d': carla2d, 'utkface': utkface}.items():
        data_plot[dataset] = load_data_from_csv(DATA_FOLDER_CASE1, dataset, variant)
        if dataset in ['reacher', 'swimmer', 'hopper']:
            if dataset == 'swimmer':
                cutoff = int(len(data_plot[dataset]['Step']) // 3 * 0.3)
            elif dataset == 'hopper':
                cutoff = int(len(data_plot[dataset]['Step']) // 3 * 0.5)
            elif dataset == 'reacher':
                cutoff = int(len(data_plot[dataset]['Step']) // 3 * 1)
            for k, v in data_plot[dataset].items():
                data_plot[dataset][k] = v.reshape(3, -1)[:, :cutoff].flatten()

    fig = plt.figure(figsize=(10, 8))

    gs = fig.add_gridspec(2, 1, hspace=0.3)  # spacing between the two groups

    gs0 = gs[0].subgridspec(2, 3, hspace=0.2)  # spacing within the groups
    gs1 = gs[1].subgridspec(2, 3, hspace=0.2)

    data_name = {'reacher': "Reacher", 'swimmer': 'Swimmer', 'hopper': "Hopper", 'carla2d': 'Carla 2D',
                 'carla1d': 'Carla 1D', 'utkface': 'UTKFace'}

    for i, dataset in enumerate(['reacher', 'swimmer', 'hopper', 'carla2d', 'carla1d', 'utkface']):
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
            # ax.set_ylabel('NRC1/NRC2', color='C0', fontfamily=FONT_FAMILY)

            ax2 = ax.twinx()
            sns.lineplot(ax=ax2, x=data_plot[dataset]['Step'], y=data_plot[dataset]['NRC3'], n_boot=N_BOOT, color='C1',
                         label='NRC3')
            print('NRC3 DONE')
            ax2.tick_params(axis='y', colors='C1')
            # ax2.set_ylabel('NRC3', color='C1', fontfamily=FONT_FAMILY)

            handles, labels = [], []
            for a in [ax, ax2]:
                for h, l in zip(*a.get_legend_handles_labels()):
                    handles.append(h)
                    labels.append(l)
            ax.legend(handles, labels, fontsize=LEGEND_FONTSIZE)
            ax2.legend().set_visible(False)
            ax.grid(True, linestyle='--')

        else:
            sns.lineplot(ax=ax, x=data_plot[dataset]['Step'], y=data_plot[dataset]['NRC1'], n_boot=N_BOOT, color='C0',
                         label='NRC1')
            print('NRC1 DONE')
            sns.lineplot(ax=ax, x=data_plot[dataset]['Step'], y=data_plot[dataset]['NRC2'], n_boot=N_BOOT, color='C0',
                         label='NRC2',
                         linestyle='--')
            print('NRC2 DONE')
            # ax.set_ylabel('NRC1/NRC2', fontfamily=FONT_FAMILY)
            ax.legend(fontsize=LEGEND_FONTSIZE)
            ax.grid(True, linestyle='--')

        if i % 3 == 0:
            ax.set_ylabel('NRC1-3', fontweight='bold', fontfamily=FONT_FAMILY)
        ax.set_title(data_name[dataset], fontweight='bold', fontfamily=FONT_FAMILY)

        if dataset == 'reacher':
            ax.xaxis.set_ticks([0, int(3.75e5), int(7.5e5), int(1.125e6), int(1.5e6)])
            ax.set_xticklabels([0, '0.375M', '0.75M', '1.125M', '1.5M'])
        if dataset == 'swimmer':
            ax.xaxis.set_ticks([0, int(0.75e5), int(1.5e5), int(2.25e5), int(3e5)])
            ax.set_xticklabels([0, '75K', '150K', '225K', '300K'])
        if dataset == 'hopper':
            ax.xaxis.set_ticks([0, int(2.5e4), int(5e4), int(7.5e4), int(1e5)])
            ax.set_xticklabels([0, '25K', '50K', '75K', '100K'])

        if dataset == 'hopper':
            ax2.set_ylim(ymin=0.1, ymax=0.18)

        if dataset in ['swimmer']:
            ax.set_yscale('log')
            ax2.set_yscale('log')

        # if dataset == 'carla1d':
        #     ax.set_ylim(ymax=0.3)
        #     # ax.set_yscale('log')
        if dataset == 'carla2d':
            ax.set_ylim(ymax=0.3)
            ax2.set_ylim(ymax=0.06)

        if dataset == 'utkface':
            ax.set_ylim(ymin=0)

        # ===================================================================================================================
        # ======================================== then plot Train and test MSE =============================================
        # ===================================================================================================================
        row = 1 if dataset in ['reacher', 'swimmer', 'hopper'] else 3
        if dataset in ['reacher', 'swimmer', 'hopper']:
            ax = fig.add_subplot(gs0[1, i % 3])
        else:
            ax = fig.add_subplot(gs1[1, i % 3])
        sns.lineplot(ax=ax, x=data_plot[dataset]['Step'], y=data_plot[dataset]['trainError'], n_boot=N_BOOT, color='C0',
                     label='Train MSE')
        print('Train MSE done')
        sns.lineplot(ax=ax, x=data_plot[dataset]['Step'], y=data_plot[dataset]['testError'], n_boot=N_BOOT, color='C0',
                     label='Test MSE', linestyle='--')
        print('Test MSE done')

        ax.tick_params(axis='y', colors='C0')
        # ax.set_ylabel('NRC1/NRC2', color='C0', fontfamily=FONT_FAMILY)

        ax2 = ax.twinx()
        sns.lineplot(ax=ax2, x=data_plot[dataset]['Step'], y=data_plot[dataset]['R_sq'], n_boot=N_BOOT, color='C1',
                     label=r'$R^2$')
        print('R_sq DONE')
        ax2.tick_params(axis='y', colors='C1')
        ax2.set_ylim(ymin=-0.1)
        # ax2.set_ylabel(r'$\mathbf{R}^2$', color='C1', fontfamily=FONT_FAMILY)

        handles, labels = [], []
        for a in [ax, ax2]:
            for h, l in zip(*a.get_legend_handles_labels()):
                handles.append(h)
                labels.append(l)
        legend_loc = {'reacher': 'right', 'swimmer': 'right', 'hopper': 'right', 'carla1d': 'right', 'carla2d': 'right',
                      'utkface': 'right'}
        bbox = {'carla1d': (1, 0.6), 'carla2d': (1, 0.6)}
        ax2.legend(handles, labels, fontsize=LEGEND_FONTSIZE, loc=legend_loc[dataset], bbox_to_anchor=bbox.get(dataset))
        ax.legend().set_visible(False)
        ax.grid(True, linestyle='--')
        # ax.set_ylabel('MSE', fontfamily=FONT_FAMILY)

        if i % 3 == 0:
            ax.set_ylabel('Model Performance', fontweight='bold', fontfamily=FONT_FAMILY)
        ax.set_xlabel('Epoch', fontfamily=FONT_FAMILY)

        if dataset == 'reacher':
            ax.xaxis.set_ticks([0, int(3.75e5), int(7.5e5), int(1.125e6), int(1.5e6)])
            ax.set_xticklabels([0, '0.375M', '0.75M', '1.125M', '1.5M'])
        if dataset == 'swimmer':
            ax.xaxis.set_ticks([0, int(0.75e5), int(1.5e5), int(2.25e5), int(3e5)])
            ax.set_xticklabels([0, '75K', '150K', '225K', '300K'])
        if dataset == 'hopper':
            ax.xaxis.set_ticks([0, int(2.5e4), int(5e4), int(7.5e4), int(1e5)])
            ax.set_xticklabels([0, '25K', '50K', '75K', '100K'])
        # if dataset in ['carla2d', 'carla1d']:
        #     ax.xaxis.set_ticks([0, 125, 250, 375, 500])
        #     ax.set_xticklabels([0, 125, 250, 375, 500])

        if dataset == 'reacher':
            ax.set_ylim(ymax=0.03, ymin=0)
            ax2.set_ylim(ymin=-0.1)
        if dataset == 'swimmer':
            ax.set_ylim(ymin=0)
            ax2.set_ylim(ymin=-0.1)

        # if dataset in ['carla1d']:
        #     # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        #     # ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        #     ax.set_yscale('log')

    for i in range(2, 4):
        for j in range(3):
            gs.update(bottom=0.1)

    positions_row_3_4 = [ax.get_position() for ax in fig.axes if ax.get_subplotspec().rowspan.start >= 2]

    # Move all subplots in rows 3 and 4 down
    for pos in positions_row_3_4:
        pos.y0 -= 0.2  # Move down by 0.2
        pos.y1 -= 0.2  # Move down by 0.2

    plt.show()
    # fig.savefig(f'./figures/camera_ready/Figure2.pdf')
    plt.close()


def EVR():
    datasets = {'reacher': reacher, 'swimmer': swimmer, 'hopper': hopper, 'carla2d': carla2d, 'carla1d': carla1d,
                'utkface': utkface}
    titles = {'reacher': r'Reacher($n=2$)', 'swimmer': r'Swimmer($n=2$)', 'hopper': r'Hopper($n=3$)',
              'carla2d': r'Carla 2D($n=2$)', 'carla1d': r'Carla 1D($n=1$)', 'utkface': r'UTKFace($n=1$)'}
    metrics = {'EVR1': 'PC1', 'EVR2': 'PC2', 'EVR3': 'PC3', 'EVR4': 'PC4', 'EVR5': 'PC5'}
    num_columns = len(datasets)
    fig, axes = plt.subplots(2, 3)
    for i, dataset in enumerate(datasets.keys()):
        print(f'Plotting EVR for {dataset}')
        row = 0 if dataset in ['reacher', 'swimmer', 'hopper'] else 1
        ax = axes[row][i % 3]
        data = load_data_from_csv(DATA_FOLDER_CASE1, dataset, datasets[dataset])
        plot_different_metrics(ax, metrics, data, reverse_color=True)

        if dataset == 'reacher':
            ax.xaxis.set_ticks([0, int(7.5e5), int(1.5e6)])
            ax.set_xticklabels([0, '0.75M', '1.5M'])
        if dataset == 'swimmer':
            ax.xaxis.set_ticks([0, int(5e5), int(1e6)])
            ax.set_xticklabels([0, '0.5M', '1M'])
        if dataset == 'hopper':
            ax.xaxis.set_ticks([0, int(1e5), int(2e5)])
            ax.set_xticklabels([0, '100K', '200K'])

        ax.grid(True, linestyle='dashed')
        if row == 0 and i == 2:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            ax.legend().set_visible(False)
        if i % 3 == 0:
            ax.set_ylabel('EVR', fontweight='bold', fontfamily=FONT_FAMILY, fontsize=FONT_SIZE)
        ax.set_xlabel('Epoch', fontweight=None, fontsize=FONT_SIZE)
        ax.set_title(titles[dataset], fontweight='bold', fontfamily=FONT_FAMILY, fontsize=FONT_SIZE)
    plt.tight_layout()
    plt.tight_layout()
    plt.show()
    # fig.savefig(f'./figures/test/Figure_EVR.pdf')


def figure_gamma():
    datasets = {'reacher': reacher, 'swimmer': swimmer, 'hopper': hopper, 'carla2d': carla2d}
    titles = {'reacher': 'Reacher', 'swimmer': 'Swimmer', 'hopper': 'Hopper', 'carla2d': 'Carla 2D'}
    num_columns = len(datasets)
    fig, axes = plt.subplots(1, num_columns)
    for i, dataset in enumerate(datasets.keys()):
        print(f'Plotting NRC3 v.s gamma for {dataset}')
        ax = axes[i]
        data = load_data_from_csv(DATA_FOLDER_CASE1, dataset, datasets[dataset])

        # x_to_plot = np.tile(np.linspace(0, 1, num=1000), 3)
        x_to_plot = data['gamma'] / max(data['gamma'])
        y_to_plot = data['NRC3(gamma)']
        if dataset in ['reacher', 'hopper', 'swimmer']:
            x_to_plot = x_to_plot[~np.isnan(x_to_plot)]
            y_to_plot = y_to_plot[~np.isnan(y_to_plot)]
        sns.lineplot(ax=ax, x=x_to_plot, y=y_to_plot, n_boot=N_BOOT, color='C0', label='NRC3')
        num_of_seeds = 3 if dataset in ['reacher', 'swimmer', 'hopper'] else 2
        minimum_idx = np.argmin(y_to_plot.reshape(num_of_seeds, -1).mean(axis=0))
        minimum = x_to_plot.reshape(num_of_seeds, -1)[0][minimum_idx]
        ax.axvline(x=minimum, color='C1', linestyle='--')

        if dataset in ['hopper', 'carla2d']:
            ax.xaxis.set_ticks([minimum.item(), 1])
            ax.set_xticklabels([round(minimum.item(), 3), 1])
        else:
            ax.xaxis.set_ticks([0, minimum.item(), 1])
            ax.set_xticklabels([0, round(minimum.item(), 3), 1])

        ax.grid(True, linestyle='dashed')

        if i == 0:
            ax.set_ylabel('Last Epoch NRC3', fontweight='bold', fontfamily=FONT_FAMILY, fontsize=FONT_SIZE)
        ax.set_xlabel(r'$\gamma/\lambda_{min}$')
        ax.set_title(titles[dataset], fontweight='bold', fontfamily=FONT_FAMILY, fontsize=FONT_SIZE)
    plt.tight_layout()
    plt.tight_layout()
    plt.show()
    # fig.savefig(f'./figures/test/Figure_gamma.pdf')


def figure4_rebuttal():
    variants = {'reacher': [reacher_wd0, reacher_wd5e6, reacher_wd5e5, reacher_wd5e4,
                            reacher_wd15e3],
                'swimmer': [swimmer_wd0, swimmer_wd5e5, swimmer_wd5e4, swimmer_wd5e3,
                            swimmer_wd1e2],
                'hopper': [hopper_wd0, hopper_wd5e6, hopper_wd5e5, hopper_wd5e4,
                           hopper_wd1e3],
                'carla2d': [carla2d_wd0, carla2d_wd5e4, carla2d_wd5e3, carla2d_wd1e2, carla2d_wd1e1],
                'carla1d': [carla1d_wd0, carla1d_wd5e4, carla1d_wd5e3, carla1d_wd1e2],
                'utkface': [utkface_wd0, utkface_wd5e4, utkface_wd5e3, utkface_wd1e2, utkface_wd1e1]
                }
    legends = {'reacher': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=5e-6$', r'$\lambda_{WD}=5e-5$', r'$\lambda_{WD}=5e-4$',
                           r'$\lambda_{WD}=1.5e-3$'],
               'swimmer': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=5e-5$', r'$\lambda_{WD}=5e-4$', r'$\lambda_{WD}=5e-3$',
                           r'$\lambda_{WD}=1e-2$'],
               'hopper': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=5e-6$', r'$\lambda_{WD}=5e-5$', r'$\lambda_{WD}=5e-4$',
                          r'$\lambda_{WD}=1e-3$'],
               'carla1d': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=5e-4$', r'$\lambda_{WD}=5e-3$', r'$\lambda_{WD}=1e-2$'],
               'carla2d': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=5e-4$', r'$\lambda_{WD}=5e-3$',
                           r'$\lambda_{WD}=1e-2$', r'$\lambda_{WD}=1e-1$'],
               'utkface': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=5e-4$', r'$\lambda_{WD}=5e-3$',
                           r'$\lambda_{WD}=1e-2$', r'$\lambda_{WD}=1e-1$']
               }
    metrics = {'trainError': 'Train MSE', 'testError': 'Test MSE', 'R_sq': r'\mathbf{R^2}', 'NRC1': 'NRC1',
               'NRC2': 'NRC2', 'NRC3': 'NRC3'}
    titles = {'reacher': 'Reacher', 'swimmer': 'Swimmer', 'hopper': 'Hopper', 'carla1d': 'Carla 1D',
              'carla2d': 'Carla 2D', 'utkface': 'UTKFace'}
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, num=9))[::-1]
    color_idx = {r'$\lambda_{WD}=0$': 0, r'$\lambda_{WD}=5e-6$': 1, r'$\lambda_{WD}=5e-5$': 2,
                 r'$\lambda_{WD}=5e-4$': 3,
                 r'$\lambda_{WD}=1e-3$': 4, r'$\lambda_{WD}=1.5e-3$': 5, r'$\lambda_{WD}=5e-3$': 6,
                 r'$\lambda_{WD}=1e-2$': 7,
                 r'$\lambda_{WD}=1e-1$': 8}

    num_rows = len(variants)
    num_columns = len(metrics)
    fig, axes = plt.subplots(num_rows, num_columns)
    for i, dataset in enumerate(variants.keys()):
        data_dict = {}
        for k, variant in enumerate(variants[dataset]):
            data_dict[legends[dataset][k]] = load_data_from_csv(DATA_FOLDER, dataset, variant)

        for j, metric in enumerate(metrics.keys()):
            if dataset in ['carla1d', 'utkface'] and metric == 'NRC3':
                fig.delaxes(axes[i][j])
                continue
            print(f'Plotting figure 4 for {dataset} w.r.t {metric}')
            ax = axes[i][j] if num_rows > 1 else axes[j]  # When num_rows = 1, ax is 1D array.

            for k, variant in enumerate(data_dict.keys()):
                x_to_plot = data_dict[variant]['Step']
                y_to_plot = data_dict[variant][metric]
                sns.lineplot(ax=ax, x=x_to_plot, y=y_to_plot, n_boot=N_BOOT, color=colors[color_idx[variant]],
                             label=None)

            if dataset == 'carla1d' and metric in ['trainError', 'testError']:
                ax.set_yscale('log')
            # if dataset == 'carla1d':
            #     ax.ticklabel_format(style='plain', axis='y')
            if dataset == 'reacher':
                ax.xaxis.set_ticks([0, int(6e5), int(1.2e6)])
                ax.set_xticklabels([0, '0.6M', '1.2M'])
            if dataset == 'swimmer':
                ax.xaxis.set_ticks([0, int(5e5), int(1e6)])
                ax.set_xticklabels([0, '0.5M', '1M'])
            if dataset == 'hopper':
                ax.xaxis.set_ticks([0, int(1e5), int(2e5)])
                ax.set_xticklabels([0, '100K', '200K'])

            ax.grid(True, linestyle='dashed')

            # if j + 1 == len(metrics):
            #     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=LEGEND_FONTSIZE)
            # elif dataset in ['carla1d', 'utkface'] and j == len(metrics) - 2:  # No NRC3 for 1D datasets, so plot legend beforehand
            #     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=LEGEND_FONTSIZE)
            # else:
            #     ax.legend().set_visible(False)
            # ax.legend().set_visible(False)

            if j == 0:
                ax.set_ylabel(titles[dataset], fontweight='bold')
            if i + 1 == num_rows:
                ax.set_xlabel('Epoch', fontweight=None)
            if i == 0:
                ax.set_title(metrics[metric], fontweight='bold')

    cbar = ColorbarBase(ax=fig.add_axes([0.88, 0.02, 0.03, 0.32]), cmap=cmap, norm=Normalize(vmin=0, vmax=1e-1))
    cbar.set_ticks(np.linspace(0, 0.1, num=9).tolist())
    cbar.set_ticklabels(['0', '5e-6', '5e-5', '5e-4', '1e-3', '1.5e-3', '5e-3', '1e-2', '1e-1'][::-1])
    cbar.set_label(r'$\lambda_{WD}$', rotation=90)

    plt.tight_layout()
    plt.tight_layout()
    plt.show()


def figure4_main():
    variants = {
        'swimmer': [swimmer_wd0, swimmer_wd5e5, swimmer_wd5e4, swimmer_wd5e3,
                    swimmer_wd1e2][::-1],
        'carla2d': [carla2d_wd0, carla2d_wd5e4, carla2d_wd5e3, carla2d_wd1e2, carla2d_wd1e1][::-1],
    }
    legends = {
        'swimmer': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=5e-5$', r'$\lambda_{WD}=5e-4$', r'$\lambda_{WD}=5e-3$',
                    r'$\lambda_{WD}=1e-2$'][::-1],
        'carla2d': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=5e-4$', r'$\lambda_{WD}=5e-3$',
                    r'$\lambda_{WD}=1e-2$', r'$\lambda_{WD}=1e-1$'][::-1],
    }
    metrics = {'trainError': 'Train MSE', 'testError': 'Test MSE', 'NRC1': 'NRC1',
               'NRC2': 'NRC2', 'NRC3': 'NRC3'}
    titles = {'reacher': 'Reacher', 'swimmer': 'Swimmer', 'hopper': 'Hopper', 'carla1d': 'Carla 1D',
              'carla2d': 'Carla 2D', 'utkface': 'UTKFace'}

    num_rows = len(variants)
    num_columns = len(metrics)
    fig, axes = plt.subplots(num_rows, num_columns)

    col_headers = ['Train MSE', 'Test MSE', 'NRC1', 'NRC2', 'NRC3']
    row_headers = ['Swimmer', 'Carla 2D']
    font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
    add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

    for i, dataset in enumerate(variants.keys()):
        data_dict = {}
        for k, variant in enumerate(variants[dataset]):
            data_dict[legends[dataset][k]] = load_data_from_csv(DATA_FOLDER_CASE1, dataset, variant)

        for j, metric in enumerate(metrics.keys()):
            if dataset in ['carla1d', 'utkface'] and metric == 'NRC3':
                fig.delaxes(axes[i][j])
                continue
            print(f'Plotting figure 4 for {dataset} w.r.t {metric}')
            ax = axes[i][j] if num_rows > 1 else axes[j]  # When num_rows = 1, ax is 1D array.

            plot_different_variants(ax, data_dict=data_dict, y_metric=metric)

            # if dataset == 'carla2d' and metric in ['trainError', 'testError']:
            #     # ax.set_yscale('log')
            #     ax.set_ylim(ymax=120)
            # if dataset == 'carla2d' and metric == 'NRC3':
            #     # ax.set_yscale('log')
            #     ax.set_ylim(ymin=-0.001, ymax=0.015)
            #     # ax.ticklabel_format(style='plain', axis='y')
            if dataset == 'swimmer':
                ax.set_yscale('log')

            if dataset == 'reacher':
                ax.xaxis.set_ticks([0, int(6e5), int(1.2e6)])
                ax.set_xticklabels([0, '0.6M', '1.2M'])
            if dataset == 'swimmer':
                ax.xaxis.set_ticks([0, int(5e5), int(1e6)])
                ax.set_xticklabels([0, '0.5M', '1M'])
            if dataset == 'hopper':
                ax.xaxis.set_ticks([0, int(1e5), int(2e5)])
                ax.set_xticklabels([0, '100K', '200K'])

            if dataset == 'carla2d':
                ax.set_xlim(xmax=72)

            ax.grid(True, linestyle='dashed')

            if j + 1 == len(metrics):
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=LEGEND_FONTSIZE)
            elif dataset in ['carla1d', 'utkface'] and j == len(
                    metrics) - 2:  # No NRC3 for 1D datasets, so plot legend beforehand
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=LEGEND_FONTSIZE)
            else:
                ax.legend().set_visible(False)

            # if j == 0:
            #     ax.set_ylabel(titles[dataset], fontweight='bold')
            ax.set_ylabel(metrics[metric])
            if i + 1 == num_rows:
                ax.set_xlabel('Epoch', fontweight=None)
            # if i == 0:
            #     ax.set_title(metrics[metric], fontweight='bold')

    # plt.tight_layout()
    # plt.tight_layout()
    plt.show()
    # fig.savefig(f'./figures/test/Figure4_main.pdf')


def figure5_main():
    variants = {
        'swimmer': [swimmer_ufm_wd0, swimmer_ufm_wd1e5, swimmer_ufm_wd1e4, swimmer_ufm_wd1e3,
                    swimmer_ufm_wd1e2][::-1],
        'carla2d': [carla2d_wd0, carla2d_wd5e4, carla2d_wd5e3, carla2d_wd1e2, carla2d_wd1e1][::-1],
    }
    legends = {
        'swimmer': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=1e-5$', r'$\lambda_{WD}=1e-4$', r'$\lambda_{WD}=1e-3$',
                    r'$\lambda_{WD}=1e-2$'][::-1],
        'carla2d': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=5e-4$', r'$\lambda_{WD}=5e-3$',
                    r'$\lambda_{WD}=1e-2$', r'$\lambda_{WD}=1e-1$'][::-1],
    }
    metrics = {'trainError': 'Train MSE', 'NRC1': 'NRC1', 'NRC2': 'NRC2', 'NRC3': 'NRC3'}
    titles = {'reacher': 'Reacher', 'swimmer': 'Swimmer', 'hopper': 'Hopper', 'carla1d': 'Carla 1D',
              'carla2d': 'Carla 2D', 'utkface': 'UTKFace'}

    num_rows = len(variants)
    num_columns = len(metrics)
    fig, axes = plt.subplots(num_rows, num_columns)

    col_headers = ['Train MSE', 'NRC1', 'NRC2', 'NRC3']
    row_headers = ['Swimmer', 'Carla 2D']
    font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
    add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

    for i, dataset in enumerate(variants.keys()):
        data_dict = {}
        for k, variant in enumerate(variants[dataset]):
            data_dict[legends[dataset][k]] = load_data_from_csv(DATA_FOLDER_CASE2, dataset, variant)

        for j, metric in enumerate(metrics.keys()):
            if dataset in ['carla1d', 'utkface'] and metric == 'NRC3':
                fig.delaxes(axes[i][j])
                continue
            print(f'Plotting figure 4 for {dataset} w.r.t {metric}')
            ax = axes[i][j] if num_rows > 1 else axes[j]  # When num_rows = 1, ax is 1D array.

            plot_different_variants(ax, data_dict=data_dict, y_metric=metric)

            if metric == 'trainError':
                # ax.set_yscale('log')
                if dataset == 'swimmer':
                    ax.set_ylim(ymin=0.033, ymax=0.045)
                # if dataset == 'carla2d':
                #     ax.set_yscale('log')
            # if dataset == 'carla2d':
            #     if metric == 'NRC1':
            #         ax.set_ylim(ymax=0.3, ymin=0.1)
            #     if metric == 'NRC2':
            #         ax.set_ylim(ymax=0.3, ymin=0.15)
            #     if metric == 'NRC3':
            #         # ax.set_ylim(ymax=0.15, ymin=-0.001)
            #         ax.set_yscale('log')

            if dataset in ['reacher', 'swimmer', 'hopper']:
                ax.xaxis.set_ticks([0, int(5e3), int(1e4)])
                ax.set_xticklabels([0, '5K', '10K'])

            ax.grid(True, linestyle='dashed')

            if j + 1 == len(metrics):
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=LEGEND_FONTSIZE)
            elif dataset in ['carla1d', 'utkface'] and j == len(
                    metrics) - 2:  # No NRC3 for 1D datasets, so plot legend beforehand
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=LEGEND_FONTSIZE)
            else:
                ax.legend().set_visible(False)

            # if j == 0:
            #     ax.set_ylabel(titles[dataset], fontweight='bold')
            ax.set_ylabel(metrics[metric])
            if i + 1 == num_rows:
                ax.set_xlabel('Epoch', fontweight=None)
            # if i == 0:
            #     ax.set_title(metrics[metric], fontweight='bold')

    # plt.tight_layout()
    # plt.tight_layout()
    plt.show()
    # fig.savefig(f'./figures/test/Figure5_main.pdf')


def figure5_appendix():
    variants = {'reacher': [reacher_ufm_wd0, reacher_ufm_wd1e5, reacher_ufm_wd1e4, reacher_ufm_wd1e3,
                            reacher_ufm_wd1e2][::-1],
                'swimmer': [swimmer_ufm_wd0, swimmer_ufm_wd1e5, swimmer_ufm_wd1e4, swimmer_ufm_wd1e3,
                            swimmer_ufm_wd1e2][::-1],
                'hopper': [hopper_ufm_wd0, hopper_ufm_wd1e5, hopper_ufm_wd1e4, hopper_ufm_wd1e3,
                           hopper_ufm_wd1e2][::-1],
                'carla2d': [carla2d_wd0, carla2d_wd5e4, carla2d_wd5e3, carla2d_wd1e2, carla2d_wd1e1][::-1],
                'carla1d': [carla1d_wd0, carla1d_wd1e4, carla1d_wd1e3, carla1d_wd1e2, carla1d_wd1e1][::-1],
                'utkface': [utkface_wd0, utkface_wd5e4, utkface_wd5e3, utkface_wd5e2, utkface_wd1e1][::-1],
                }
    legends = {'reacher': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=1e-5$', r'$\lambda_{WD}=1e-4$', r'$\lambda_{WD}=1e-3$',
                           r'$\lambda_{WD}=1e-2$'][::-1],
               'swimmer': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=1e-5$', r'$\lambda_{WD}=1e-4$', r'$\lambda_{WD}=1e-3$',
                           r'$\lambda_{WD}=1e-2$'][::-1],
               'hopper': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=1e-5$', r'$\lambda_{WD}=1e-4$', r'$\lambda_{WD}=1e-3$',
                          r'$\lambda_{WD}=1e-2$'][::-1],
               'carla2d': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=5e-4$', r'$\lambda_{WD}=5e-3$',
                           r'$\lambda_{WD}=1e-2$', r'$\lambda_{WD}=1e-1$'][::-1],
               'carla1d': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=1e-4$', r'$\lambda_{WD}=1e-3$',
                           r'$\lambda_{WD}=1e-2$', r'$\lambda_{WD}=1e-1$'][::-1],
               'utkface': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=5e-4$', r'$\lambda_{WD}=5e-3$',
                           r'$\lambda_{WD}=5e-2$', r'$\lambda_{WD}=1e-1$'][::-1],
               }
    metrics = {'trainError': 'Train MSE', 'NRC1': 'NRC1', 'NRC2': 'NRC2', 'NRC3': 'NRC3'}
    titles = {'reacher': 'Reacher', 'swimmer': 'Swimmer', 'hopper': 'Hopper', 'carla1d': 'Carla 1D',
              'carla2d': 'Carla 2D', 'utkface': 'UTKFace'}

    num_rows = len(variants)
    num_columns = len(metrics)
    fig, axes = plt.subplots(num_rows, num_columns)

    col_headers = ['Train MSE', 'NRC1', 'NRC2', 'NRC3']
    row_headers = ['Reacher', 'Swimmer', 'Hopper', 'Carla 2D', 'Carla 1D', 'UTKFace']
    font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
    add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

    for i, dataset in enumerate(variants.keys()):
        data_dict = {}
        for k, variant in enumerate(variants[dataset]):
            data_dict[legends[dataset][k]] = load_data_from_csv(DATA_FOLDER_CASE2, dataset, variant)

        for j, metric in enumerate(metrics.keys()):
            if dataset in ['carla1d', 'utkface'] and metric == 'NRC3':
                fig.delaxes(axes[i][j])
                continue
            print(f'Plotting figure 4 for {dataset} w.r.t {metric}')
            ax = axes[i][j] if num_rows > 1 else axes[j]  # When num_rows = 1, ax is 1D array.

            plot_different_variants(ax, data_dict=data_dict, y_metric=metric)

            if metric == 'trainError':
                # ax.set_yscale('log')
                if dataset == 'swimmer':
                    ax.set_ylim(ymin=0.033, ymax=0.045)
                # if dataset == 'carla2d':
                #     ax.set_yscale('log')
            # if dataset == 'carla2d':
            #     if metric == 'NRC1':
            #         ax.set_ylim(ymax=0.3, ymin=0.1)
            #     if metric == 'NRC2':
            #         ax.set_ylim(ymax=0.3, ymin=0.15)
            #     if metric == 'NRC3':
            #         # ax.set_ylim(ymax=0.15, ymin=-0.001)
            #         ax.set_yscale('log')

            if dataset in ['reacher', 'swimmer', 'hopper']:
                ax.xaxis.set_ticks([0, int(5e3), int(1e4)])
                ax.set_xticklabels([0, '5K', '10K'])

            ax.grid(True, linestyle='dashed')

            if j + 1 == len(metrics):
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=LEGEND_FONTSIZE)
            elif dataset in ['carla1d', 'utkface'] and j == len(
                    metrics) - 2:  # No NRC3 for 1D datasets, so plot legend beforehand
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=LEGEND_FONTSIZE)
            else:
                ax.legend().set_visible(False)

            # if j == 0:
            #     ax.set_ylabel(titles[dataset], fontweight='bold')
            ax.set_ylabel(metrics[metric])
            if i + 1 == num_rows:
                ax.set_xlabel('Epoch', fontweight=None)
            # if i == 0:
            #     ax.set_title(metrics[metric], fontweight='bold')

    # plt.tight_layout()
    # plt.tight_layout()
    plt.show()
    # fig.savefig(f'./figures/test/Figure5_appendix.pdf')


def nrc1_def():
    datasets = {'reacher': reacher_wd15e3, 'swimmer': swimmer_wd1e2, 'hopper': hopper_wd1e3}
    metrics = {'NRC1_pca1': r'$\mathbf{H}_{PCA_1}$', 'NRC1_pca2': r'$\mathbf{H}_{PCA_2}$',
               'NRC1_pca3': r'$\mathbf{H}_{PCA_3}$',
               'NRC1_pca4': r'$\mathbf{H}_{PCA_4}$', 'NRC1_pca5': r'$\mathbf{H}_{PCA_5}$'}
    num_columns = len(datasets)
    fig, axes = plt.subplots(1, num_columns)
    for i, dataset in enumerate(datasets.keys()):
        print(f'Plotting NRC1_pca for {dataset}.')
        ax = axes[i]
        data = load_data_from_csv(DATA_FOLDER_CASE1, dataset, datasets[dataset])
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


def figure4_appendix(part):
    if part == 0:
        variants = {'reacher': [reacher_wd0, reacher_wd5e6, reacher_wd5e5, reacher_wd5e4,
                                reacher_wd15e3][::-1],
                    'swimmer': [swimmer_wd0, swimmer_wd5e5, swimmer_wd5e4, swimmer_wd5e3,
                                swimmer_wd1e2][::-1],
                    'hopper': [hopper_wd0, hopper_wd5e5, hopper_wd5e4, hopper_wd5e3,
                               hopper_wd1e2][::-1],
                    }
        legends = {
            'reacher': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=5e-6$', r'$\lambda_{WD}=5e-5$', r'$\lambda_{WD}=5e-4$',
                        r'$\lambda_{WD}=1.5e-3$'][::-1],
            'swimmer': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=5e-5$', r'$\lambda_{WD}=5e-4$', r'$\lambda_{WD}=5e-3$',
                        r'$\lambda_{WD}=1e-2$'][::-1],
            'hopper': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=5e-5$', r'$\lambda_{WD}=5e-4$', r'$\lambda_{WD}=5e-3$',
                       r'$\lambda_{WD}=1e-2$'][::-1],
        }
        col_headers = ['Reacher', 'Swimmer', 'Hopper']
        row_headers = ['Train MSE', 'Test MSE', r'$\mathbf{R^2}$', 'NRC1', 'NRC2', 'NRC3']

    elif part == 1:
        variants = {
            'carla2d': [carla2d_wd0, carla2d_wd5e4, carla2d_wd5e3, carla2d_wd1e2, carla2d_wd1e1][::-1],
            'carla1d': [carla1d_wd0, carla1d_wd1e4, carla1d_wd1e3, carla1d_wd1e2, carla1d_wd1e1][::-1],
            'utkface': [utkface_wd0, utkface_wd5e4, utkface_wd5e2, utkface_wd1e2, utkface_wd1e1][::-1]
        }
        legends = {
            'carla2d': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=5e-4$', r'$\lambda_{WD}=5e-3$', r'$\lambda_{WD}=1e-2$',
                        r'$\lambda_{WD}=1e-1$', ][::-1],
            'carla1d': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=1e-4$', r'$\lambda_{WD}=1e-3$', r'$\lambda_{WD}=1e-2$',
                        r'$\lambda_{WD}=1e-1$', ][::-1],
            'utkface': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=5e-4$', r'$\lambda_{WD}=5e-2$',
                        r'$\lambda_{WD}=1e-2$', r'$\lambda_{WD}=1e-1$', ][::-1]
        }
        col_headers = ['Carla 2D', 'Carla 1D', 'UTKFace']
        row_headers = ['Train MSE', 'Test MSE', r'$\mathbf{R^2}$', 'NRC1', 'NRC2', 'NRC3']

    metrics = {'trainError': 'Train MSE', 'testError': 'Test MSE', 'R_sq': r'$\mathbf{R^2}$', 'NRC1': 'NRC1',
               'NRC2': 'NRC2', 'NRC3': 'NRC3'}
    titles = {'reacher': 'Reacher', 'swimmer': 'Swimmer', 'hopper': 'Hopper', 'carla1d': 'Carla 1D',
              'carla2d': 'Carla 2D', 'utkface': 'UTKFace'}

    num_columns = len(variants)
    num_rows = len(metrics)
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(8, 10))

    row_headers = ['Train MSE', 'Test MSE', r'$\mathbf{R^2}$', 'NRC1', 'NRC2', 'NRC3']
    font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
    add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

    for i, dataset in enumerate(variants.keys()):
        data_dict = {}
        for k, variant in enumerate(variants[dataset]):
            data_dict[legends[dataset][k]] = load_data_from_csv(DATA_FOLDER_CASE1, dataset, variant)

        for j, metric in enumerate(metrics.keys()):
            if dataset in ['carla1d', 'utkface'] and metric == 'NRC3':
                fig.delaxes(axes[j][i])
                continue
            print(f'Plotting figure 4 for {dataset} w.r.t {metric}')
            ax = axes[j][i] if num_rows > 1 else axes[j]  # When num_rows = 1, ax is 1D array.

            plot_different_variants(ax, data_dict=data_dict, y_metric=metric)


            if dataset in ['reacher', 'swimmer', 'hopper']:
                if (dataset, metric) in [('reacher', 'trainError'), ('swimmer', 'trainError'), ('hopper', 'trainError'),
                                         ('reacher', 'testError'), ('swimmer', 'NRC3'), ('hopper', 'NRC3')]:
                    ax.set_yscale('log')
                if metric == 'R_sq':
                    if dataset == 'reacher':
                        ax.set_ylim(ymin=0.4)
                    if dataset == 'swimmer':
                        ax.set_ylim(ymin=0.4)
                    if dataset == 'hopper':
                        ax.set_ylim(ymin=0.2)
                if (dataset, metric) in [('hopper', 'testError'), ('swimmer', 'testError'), ('swimmer', 'trainError')]:
                    ax.set_ylim(ymax=0.25)

            if dataset == 'reacher':
                ax.xaxis.set_ticks([0, int(7.5e5), int(1.5e6)])
                ax.set_xticklabels([0, '0.75M', '1.5M'])
            if dataset == 'swimmer':
                ax.xaxis.set_ticks([0, int(5e5), int(1e6)])
                ax.set_xticklabels([0, '0.5M', '1M'])
            if dataset == 'hopper':
                ax.xaxis.set_ticks([0, int(1e5), int(2e5)])
                ax.set_xticklabels([0, '100K', '200K'])

            if dataset == 'carla2d':
                ax.set_xlim(xmax=72)

            # if dataset == 'utkface':
            #     ax.set_yscale('log')

            ax.grid(True, linestyle='dashed')

            if j == num_rows - 1:
                # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=LEGEND_FONTSIZE)
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), fontsize=LEGEND_FONTSIZE, ncol=2)
            elif dataset in ['carla1d',
                             'utkface'] and j == num_rows - 2:  # No NRC3 for 1D datasets, so plot legend beforehand
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), fontsize=LEGEND_FONTSIZE, ncol=2)
            else:
                ax.legend().set_visible(False)

            # if j == 0:
            #     ax.set_ylabel(titles[dataset], fontweight='bold')
            # ax.set_ylabel(metrics[metric])
            if j == num_rows - 1:
                ax.set_xlabel('Epoch', fontweight=None)
            if dataset in ['carla1d', 'utkface'] and j == num_rows - 2:
                ax.set_xlabel('Epoch', fontweight=None)
            # if i == 0:
            #     ax.set_title(metrics[metric], fontweight='bold')

    # plt.tight_layout()
    # plt.tight_layout()
    plt.show()
    # fig.savefig(f'./figures/test/Figure4_full_{part}.pdf')

# figure_main_result()
# EVR()
figure_gamma()
# figure4_main()
# figure5_main()
# figure4_appendix(part=1)
# figure5_appendix()
