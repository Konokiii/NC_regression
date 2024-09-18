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
import torch

from log_alias import *

DATA_FOLDER = '../results/rebuttal'
N_BOOT = 20
LEGEND_FONTSIZE = 7.5
MUJOCO_SKIP = 50
FONT_FAMILY = 'monospace'


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
                    if col == 'Step' and dataset in ['reacher', 'hopper', 'swimmer']:  # adjust eval_freq for mujoco
                        data[col] = np.concatenate([data.get(col, []), df[col].to_numpy() * 100])
                    elif col == 'Epoch' and dataset in ['carla1d', 'carla2d', 'utkface']:  # unify x-axis name as Step for carla/utkface
                        data['Step'] = np.concatenate([data.get('Step', []), df[col].to_numpy()])
                    elif col in ['NRC1', 'NRC2'] and dataset in ['reacher', 'hopper', 'swimmer']:  # adjust NRC1/2 values due to constant multiplier
                        data[col] = np.concatenate([data.get(col, []), df[col].to_numpy() * 256])
                    else:
                        data[col] = np.concatenate([data.get(col, []), df[col].to_numpy()])
        elif '.csv' in f and 'nrc3_gamma' in f:
            with open(path, 'r') as csv_file:
                df = pd.read_csv(csv_file, delimiter=",", header=0)
                if dataset == 'carla2d':
                    data['NRC3_vs_gamma'] = np.concatenate([df[f'rs{s}'].to_numpy() for s in [0, 1, 2]])
                else:
                    data['NRC3_vs_gamma'] = np.concatenate([data.get('NRC3_vs_gamma', []), df['NRC3'].to_numpy()])

    if dataset in ['reacher', 'swimmer', 'hopper']:
        for k in data.keys():
            if k == 'NRC3_vs_gamma':
                continue
            data[k] = data[k].reshape(3, -1)[:, ::MUJOCO_SKIP].flatten()  # Too many points in plots.
            # if k == 'Step' and variant.endswith('wd0'):  # Fit 0 wd curves to figure 4.
            #     scale = {'reacher': 1.2/6, 'swimmer': 1/6, 'hopper': 2/5}
            #     data['Step'] = data['Step'] * scale[dataset]

    return data


def add_headers(fig,*,row_headers=None,col_headers=None,row_pad=1,col_pad=5,rotate_row_headers=True,**text_kwargs):
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


def plot_different_metrics(ax, y_metrics_dict, data, x_metric='Step', reverse_color=False):  # metrics_dict, key=metric_name, value=legend
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, num=len(y_metrics_dict)))[::-1 if reverse_color else 1]
    x_to_plot = data[x_metric]
    for i, m in enumerate(y_metrics_dict.keys()):
        y_to_plot = data[m]
        sns.lineplot(ax=ax, x=x_to_plot, y=y_to_plot, n_boot=N_BOOT, color=colors[i], label=y_metrics_dict[m])


def plot_different_variants(ax, data_dict, y_metric, x_metrix='Step', reverse_color=False):  # data_dict, key=legend, value=full_data
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, num=len(data_dict)))[::-1 if reverse_color else 1]
    for i, variant in enumerate(data_dict.keys()):
        x_to_plot = data_dict[variant][x_metrix]
        y_to_plot = data_dict[variant][y_metric]
        sns.lineplot(ax=ax, x=x_to_plot, y=y_to_plot, n_boot=N_BOOT, color=colors[i], label=variant)


def figure2():
    data_plot = {}
    for dataset, variant in {'reacher': reacher, 'swimmer': swimmer, 'hopper': hopper, 'carla1d': carla1d, 'carla2d': carla2d, 'utkface': utkface}.items():
        data_plot[dataset] = load_data_from_csv(DATA_FOLDER, dataset, variant)
        if dataset in ['reacher', 'swimmer']:
            for k, v in data_plot[dataset].items():
                data_plot[dataset][k] = v.reshape(3, -1)[:, :80].flatten()

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

            if dataset == 'reacher':
                ax.xaxis.set_ticks([0, int(1e5), int(2e5), int(3e5), int(4e5)])
                ax.set_xticklabels([0, '100K', '200K', '300K', '400K'])
            if dataset == 'swimmer':
                ax.xaxis.set_ticks([0, int(1e5), int(2e5), int(3e5), int(4e5)])
                ax.set_xticklabels([0, '100K', '200K', '300K', '400K'])
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
            # ax.set_ylabel('NRC1/NRC2', fontfamily=FONT_FAMILY)
            ax.legend(fontsize=LEGEND_FONTSIZE)
            ax.grid(True, linestyle='--')

        if dataset in ['swimmer']:
            ax.set_yscale('log')
            ax2.set_yscale('log')

        if i % 3 == 0:
            ax.set_ylabel('NRC1-3', fontweight='bold', fontfamily=FONT_FAMILY)
        ax.set_title(data_name[dataset], fontweight='bold', fontfamily=FONT_FAMILY)

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
        legend_loc = {'reacher': 'right', 'swimmer': 'right', 'hopper': 'right', 'carla1d': 'lower left', 'carla2d': 'lower left', 'utkface': 'right'}
        ax2.legend(handles, labels, fontsize=LEGEND_FONTSIZE, loc=legend_loc[dataset])
        ax.legend().set_visible(False)
        ax.grid(True, linestyle='--')

        # ax.set_ylabel('MSE', fontfamily=FONT_FAMILY)

        if dataset == 'reacher':
            ax.xaxis.set_ticks([0, int(1e5), int(2e5), int(3e5), int(4e5)])
            ax.set_xticklabels([0, '100K', '200K', '300K', '400K'])
        if dataset == 'swimmer':
            ax.xaxis.set_ticks([0, int(1e5), int(2e5), int(3e5), int(4e5)])
            ax.set_xticklabels([0, '100K', '200K', '300K', '400K'])
        if dataset == 'hopper':
            ax.xaxis.set_ticks([0, int(0.5e5), int(1e5), int(1.5e5), int(2e5)])
            ax.set_xticklabels([0, '50K', '100K', '150K', '200K'])

        if dataset in ['carla1d']:
            # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            # ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax.set_yscale('log')

        if i % 3 == 0:
            ax.set_ylabel('Model Performance', fontweight='bold', fontfamily=FONT_FAMILY)
        ax.set_xlabel('Epoch', fontfamily=FONT_FAMILY)

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


def figure3():
    datasets = {'reacher': reacher_wd15e3, 'swimmer': swimmer_wd1e2, 'hopper': hopper_wd1e3, 'carla2d': carla2d}
    titles = {'reacher': 'Reacher', 'swimmer': 'Swimmer', 'hopper': 'Hopper', 'carla2d': 'Carla 2D'}
    num_columns = len(datasets)
    fig, axes = plt.subplots(1, num_columns)
    for i, dataset in enumerate(datasets.keys()):
        print(f'Plotting NRC3 v.s gamma for {dataset}')
        ax = axes[i]
        data = load_data_from_csv(DATA_FOLDER, dataset, datasets[dataset])

        x_to_plot = np.tile(np.linspace(0, 1, num=1000), 3)
        y_to_plot = data['NRC3_vs_gamma']
        sns.lineplot(ax=ax, x=x_to_plot, y=y_to_plot, n_boot=N_BOOT, color='C0', label='NRC3')
        minimum_idx = np.argmin(y_to_plot.reshape(3, -1).mean(axis=0))
        minimum = np.linspace(0, 1, num=1000)[minimum_idx]
        ax.axvline(x=minimum, color='C1', linestyle='--')

        if dataset in ['hopper', 'carla2d']:
            ax.xaxis.set_ticks([minimum.item(), 1])
            ax.set_xticklabels([round(minimum.item(), 3), 1])
        else:
            ax.xaxis.set_ticks([0, minimum.item(), 1])
            ax.set_xticklabels([0, round(minimum.item(), 3), 1])

        ax.grid(True, linestyle='dashed')

        if i == 0:
            ax.set_ylabel('Last Epoch NRC3', fontweight='bold', fontfamily=FONT_FAMILY)
        ax.set_xlabel(r'$\gamma/\lambda_{min}$')
        ax.set_title(titles[dataset], fontweight='bold', fontfamily=FONT_FAMILY)
    plt.tight_layout()
    plt.tight_layout()
    plt.show()


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
    titles = {'reacher': 'Reacher', 'swimmer': 'Swimmer', 'hopper': 'Hopper', 'carla1d': 'Carla 1D', 'carla2d': 'Carla 2D', 'utkface': 'UTKFace'}
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, num=9))[::-1]
    color_idx = {r'$\lambda_{WD}=0$': 0, r'$\lambda_{WD}=5e-6$': 1, r'$\lambda_{WD}=5e-5$': 2, r'$\lambda_{WD}=5e-4$': 3,
                 r'$\lambda_{WD}=1e-3$': 4, r'$\lambda_{WD}=1.5e-3$': 5, r'$\lambda_{WD}=5e-3$': 6, r'$\lambda_{WD}=1e-2$': 7,
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
                sns.lineplot(ax=ax, x=x_to_plot, y=y_to_plot, n_boot=N_BOOT, color=colors[color_idx[variant]], label=None)

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


def figure4():
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
    titles = {'reacher': 'Reacher', 'swimmer': 'Swimmer', 'hopper': 'Hopper', 'carla1d': 'Carla 1D', 'carla2d': 'Carla 2D', 'utkface': 'UTKFace'}

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
            data_dict[legends[dataset][k]] = load_data_from_csv(DATA_FOLDER, dataset, variant)

        for j, metric in enumerate(metrics.keys()):
            if dataset in ['carla1d', 'utkface'] and metric == 'NRC3':
                fig.delaxes(axes[i][j])
                continue
            print(f'Plotting figure 4 for {dataset} w.r.t {metric}')
            ax = axes[i][j] if num_rows > 1 else axes[j]  # When num_rows = 1, ax is 1D array.

            plot_different_variants(ax, data_dict=data_dict, y_metric=metric)

            if dataset == 'carla1d' and metric in ['trainError', 'testError']:
                ax.set_yscale('log')
                ax.ticklabel_format(style='plain', axis='y')
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

            ax.grid(True, linestyle='dashed')

            if j + 1 == len(metrics):
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=LEGEND_FONTSIZE)
            elif dataset in ['carla1d', 'utkface'] and j == len(metrics) - 2:  # No NRC3 for 1D datasets, so plot legend beforehand
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

    plt.tight_layout()
    plt.tight_layout()
    plt.show()


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
    datasets = {'reacher': reacher_wd15e3, 'swimmer': swimmer_wd1e2, 'hopper': hopper_wd1e3, 'carla1d': carla1d, 'carla2d': carla2d, 'utkface': utkface}
    titles = {'reacher': 'Reacher', 'swimmer': 'Swimmer', 'hopper': 'Hopper', 'carla1d': 'Carla 1D',
              'carla2d': 'Carla 2D', 'utkface': 'UTKFace'}
    metrics = {'EVR1': 'PC1', 'EVR2': 'PC2', 'EVR3': 'PC3', 'EVR4': 'PC4', 'EVR5': 'PC5'}
    num_columns = len(datasets)
    fig, axes = plt.subplots(2, 3)
    for i, dataset in enumerate(datasets.keys()):
        print(f'Plotting EVR for {dataset}')
        row = 0 if dataset in ['reacher', 'swimmer', 'hopper'] else 1
        ax = axes[row][i % 3]
        data = load_data_from_csv(DATA_FOLDER, dataset, datasets[dataset])
        plot_different_metrics(ax, metrics, data, reverse_color=True)

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
        if row == 0 and i == 2:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            ax.legend().set_visible(False)
        if i % 3 == 0:
            ax.set_ylabel('Explained Variance Ratio', fontweight='bold', fontfamily=FONT_FAMILY)
        ax.set_xlabel('Epoch', fontweight=None)
        ax.set_title(titles[dataset], fontweight='bold', fontfamily=FONT_FAMILY)
    plt.tight_layout()
    plt.tight_layout()
    plt.show()


def figure4_appendix(part):
    if part == 0:
        variants = {'reacher': [reacher_wd0, reacher_wd5e6, reacher_wd5e5, reacher_wd5e4,
                                reacher_wd15e3][::-1],
                    'swimmer': [swimmer_wd0, swimmer_wd5e5, swimmer_wd5e4, swimmer_wd5e3,
                                swimmer_wd1e2][::-1],
                    'hopper': [hopper_wd0, hopper_wd5e6, hopper_wd5e5, hopper_wd5e4,
                               hopper_wd1e3][::-1],
                    }
        legends = {
            'reacher': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=5e-6$', r'$\lambda_{WD}=5e-5$', r'$\lambda_{WD}=5e-4$',
                        r'$\lambda_{WD}=1.5e-3$'][::-1],
            'swimmer': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=5e-5$', r'$\lambda_{WD}=5e-4$', r'$\lambda_{WD}=5e-3$',
                        r'$\lambda_{WD}=1e-2$'][::-1],
            'hopper': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=5e-6$', r'$\lambda_{WD}=5e-5$', r'$\lambda_{WD}=5e-4$',
                       r'$\lambda_{WD}=1e-3$'][::-1],
                    }
        col_headers = ['Reacher', 'Swimmer', 'Hopper']
        row_headers = ['Train MSE', 'Test MSE', r'$\mathbf{R^2}$', 'NRC1', 'NRC2', 'NRC3']

    elif part == 1:
        variants = {
                    'carla2d': [carla2d_wd0, carla2d_wd5e4, carla2d_wd5e3, carla2d_wd1e2, carla2d_wd1e1][::-1],
                    'carla1d': [carla1d_wd0, carla1d_wd5e4, carla1d_wd5e3, carla1d_wd1e2][::-1],
                    'utkface': [utkface_wd0, utkface_wd5e4, utkface_wd5e3, utkface_wd1e2, utkface_wd1e1][::-1]
                    }
        legends = {
                   'carla1d': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=5e-4$', r'$\lambda_{WD}=5e-3$', r'$\lambda_{WD}=1e-2$'][::-1],
                   'carla2d': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=5e-4$', r'$\lambda_{WD}=5e-3$',
                               r'$\lambda_{WD}=1e-2$', r'$\lambda_{WD}=1e-1$'][::-1],
                   'utkface': [r'$\lambda_{WD}=0$', r'$\lambda_{WD}=5e-4$', r'$\lambda_{WD}=5e-3$',
                               r'$\lambda_{WD}=1e-2$', r'$\lambda_{WD}=1e-1$'][::-1]
                   }
        col_headers = ['Carla 2D', 'Carla 1D', 'UTKFace']
        row_headers = ['Train MSE', 'Test MSE', r'$\mathbf{R^2}$', 'NRC1', 'NRC2', 'NRC3']

    metrics = {'trainError': 'Train MSE', 'testError': 'Test MSE', 'R_sq': r'$\mathbf{R^2}$', 'NRC1': 'NRC1',
               'NRC2': 'NRC2', 'NRC3': 'NRC3'}
    titles = {'reacher': 'Reacher', 'swimmer': 'Swimmer', 'hopper': 'Hopper', 'carla1d': 'Carla 1D', 'carla2d': 'Carla 2D', 'utkface': 'UTKFace'}

    num_columns = len(variants)
    num_rows = len(metrics)
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(8, 10))

    row_headers = ['Train MSE', 'Test MSE', r'$\mathbf{R^2}$', 'NRC1', 'NRC2', 'NRC3']
    font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
    add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

    for i, dataset in enumerate(variants.keys()):
        data_dict = {}
        for k, variant in enumerate(variants[dataset]):
            data_dict[legends[dataset][k]] = load_data_from_csv(DATA_FOLDER, dataset, variant)

        for j, metric in enumerate(metrics.keys()):
            if dataset in ['carla1d', 'utkface'] and metric == 'NRC3':
                fig.delaxes(axes[j][i])
                continue
            print(f'Plotting figure 4 for {dataset} w.r.t {metric}')
            ax = axes[j][i] if num_rows > 1 else axes[j]  # When num_rows = 1, ax is 1D array.

            plot_different_variants(ax, data_dict=data_dict, y_metric=metric)

            if dataset in ['carla1d', 'utkface'] and metric in ['trainError', 'testError']:
                ax.set_yscale('log')
                # ax.ticklabel_format(style='plain', axis='y')
            if dataset in ['reacher', 'swimmer', 'hopper']:
                if (dataset, metric) not in [('reacher', 'R_sq'), ('swimmer', 'R_sq'), ('hopper', 'testError'), ('hopper', 'R_sq')]:
                    ax.set_yscale('log')
                if metric == 'R_sq':
                    ax.set_ylim(ymin=0.4)
                if (dataset, metric) == ('hopper', 'testError'):
                    ax.set_ylim(ymax=0.15)

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

            if j == num_rows - 1:
                # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=LEGEND_FONTSIZE)
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), fontsize=LEGEND_FONTSIZE, ncol=2)
            elif dataset in ['carla1d', 'utkface'] and j == num_rows - 2:  # No NRC3 for 1D datasets, so plot legend beforehand
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
    fig.savefig(f'./figures/1st arxiv/Figure4_full_{part}.pdf')

figure2()
# figure3()
# figure4()
# EVR()
# figure4_appendix(part=1)


