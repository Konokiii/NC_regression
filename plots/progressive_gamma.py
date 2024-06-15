import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import torch

from log_alias import *

plt.rcParams["font.family"] = "monospace"
DEFAULT_LINESTYLES = tuple(['solid' for _ in range(8)])
DEFAULT_COLORS = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:purple')
DATA_FOLDER = '../results/wwt'
FIGURE_SAVE_PATH = './figures'


def load_wwt(env):
    file_path = os.path.join(DATA_FOLDER, env)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        return data['min_eigval'], data['Sigma_sqrt'], data['WWT']


def compute_all_gamma_and_nrc3(min_eigval, Sigma_sqrt, WWT, num_to_search=5000):
    feasible_gamma = np.linspace(0, min_eigval, num=num_to_search).reshape(-1, 1)
    dim = Sigma_sqrt.shape[0]
    Sigma_sqrt = Sigma_sqrt.flatten()
    feasible_A = Sigma_sqrt - feasible_gamma ** 0.5 * np.eye(dim).flatten()

    WWT_normalized = WWT / np.linalg.norm(WWT, axis=1, keepdims=True)
    feasible_A_normalized = feasible_A / np.linalg.norm(feasible_A, axis=1, keepdims=True)
    diff_mat = WWT_normalized[:, None, :] - feasible_A_normalized[None, :, :]
    feasible_NRC3 = np.linalg.norm(diff_mat, axis=2) ** 2

    all_gamma = feasible_gamma[np.argmin(feasible_NRC3, axis=1)].flatten()
    all_NRC3 = np.min(feasible_NRC3, axis=1)

    return all_gamma, all_NRC3


def compute_nrc3_wrt_gamma(min_eigval, Sigma_sqrt, WWT, gamma):
    if gamma <= 0 or gamma >= min_eigval:
        print("Warning: Gamma is not contained in the desired interval! Compute NRC3 anyway.")

    dim = Sigma_sqrt.shape[0]
    Sigma_sqrt = Sigma_sqrt.flatten()
    A = Sigma_sqrt - gamma ** 0.5 * np.eye(dim).flatten()

    WWT_normalized = WWT / np.linalg.norm(WWT, axis=1, keepdims=True)
    A_normalized = A / np.linalg.norm(A)
    diff_mat = WWT_normalized - A_normalized
    all_NRC3 = np.linalg.norm(diff_mat, axis=1) ** 2

    return all_NRC3


def plot_gamma_over_epoch(ax, variant, cutoff=None):
    min_eigval, Sigma_sqrt, WWT = load_wwt(variant)
    all_gamma, _ = compute_all_gamma_and_nrc3(min_eigval, Sigma_sqrt, WWT)

    cutoff = cutoff or 1.0
    cutoff = int(all_gamma.shape[0] * cutoff)

    x_to_plot = np.arange(all_gamma.shape[0]) * 100
    y_to_plot1 = all_gamma
    y_to_plot2 = [min_eigval] * x_to_plot.shape[0]

    sns.lineplot(ax=ax, x=x_to_plot[:cutoff], y=y_to_plot1[:cutoff], color=DEFAULT_COLORS[0], linestyle='solid')
    sns.lineplot(ax=ax, x=x_to_plot[:cutoff], y=y_to_plot2[:cutoff], color=DEFAULT_COLORS[1], linestyle='dashed')


def compare_nrc3(ax, variant, cutoff=None):
    min_eigval, Sigma_sqrt, WWT = load_wwt(variant)
    all_gamma, current_nrc3 = compute_all_gamma_and_nrc3(min_eigval, Sigma_sqrt, WWT)
    previous_nrc3 = compute_nrc3_wrt_gamma(min_eigval, Sigma_sqrt, WWT, all_gamma[-1])

    cutoff = cutoff or 1.0
    cutoff = int(all_gamma.shape[0] * cutoff)

    x_to_plot = np.arange(all_gamma.shape[0]) * 100
    y_to_plot1 = current_nrc3
    y_to_plot2 = previous_nrc3

    sns.lineplot(ax=ax, x=x_to_plot[:cutoff], y=y_to_plot1[:cutoff], label='now', color=DEFAULT_COLORS[0],
                 linestyle=DEFAULT_LINESTYLES[0])
    sns.lineplot(ax=ax, x=x_to_plot[:cutoff], y=y_to_plot2[:cutoff], label='previous', color=DEFAULT_COLORS[1],
                 linestyle=DEFAULT_LINESTYLES[1])


def plot_nrc3_wrt_gamma():
    pass


if __name__ == '__main__':
    envs = ['reacher', 'swimmer', 'hopper']
    variants = [reacher_wwt, swimmer_wwt, hopper_wwt]
    measures = ['gamma', 'NRC3']

    num_rows = len(measures)
    num_columns = len(envs)
    fig, axes = plt.subplots(num_rows, num_columns)

    for i, measure in enumerate(measures):
        for j, env in enumerate(envs):
            ax = axes[i][j] if num_rows > 1 else axes[j]
            if i == 0:
                ax.set_title(env.capitalize(), fontweight='bold')
            if j == 0:
                ax.set_ylabel(measure, fontweight='bold')
            if i == num_rows - 1:
                ax.set_xlabel('Epoch')
            ax.set_yscale('log')
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

            if measure == 'gamma':
                plot_gamma_over_epoch(ax, variants[j], cutoff=0.15 if env == 'swimmer' else None)
            elif measure == 'NRC3':
                compare_nrc3(ax, variants[j], cutoff=0.15 if env == 'swimmer' else None)

    plt.tight_layout()
    plt.tight_layout()
    plt.show()
    plt.close()

