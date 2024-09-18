import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random
import pickle
import torch
import io


# ====================== utility ======================
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


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

# ========= colar mapping =========
# Normalize x values to [0, 1] for colormap mapping
cmap1 = plt.cm.viridis
cmap2 = plt.cm.plasma
norm = plt.Normalize(1, 4)

# ============================ Plot ============================
mosaic = [
    ["A0", "A1", "A2", "A3", ],
    ["B0", "B1", "B2", "B3", ]
]
row_headers = ["Reacher", "Swimmer"]
col_headers = ["Train MSE", "NRC1", "NRC2", "NRC3"]

subplots_kwargs = dict(sharex=False, sharey=False, figsize=(10, 6))
fig, axes = plt.subplot_mosaic(mosaic, **subplots_kwargs)

font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)


for row in ['A','B']:
    if row == 'A':
        path = 'result/reacher/'
    elif row == 'B':
        path = 'result/swimmer/'

    nc1 = pd.read_csv(os.path.join(path, 'nc1.csv'))
    nc2 = pd.read_csv(os.path.join(path, 'nc2.csv'))
    nc3 = pd.read_csv(os.path.join(path, 'nc3.csv'))
    acc = pd.read_csv(os.path.join(path, 'train_mse.csv'))
    test_acc = pd.read_csv(os.path.join(path, 'val_mse.csv'))

    i = row + '0'
    # ==== Training Acc
    axes[i].plot(acc['Step'].values*2, acc['e-2'].values, label='$\lambda_H=\lambda_W=1e-2$', color=cmap1(norm(1)))
    axes[i].plot(acc['Step'].values*2,  acc['e-3'].values, label='$\lambda_H=\lambda_W=1e-3$', color=cmap1(norm(2)))
    axes[i].plot(acc['Step'].values*2, acc['e-4'].values, label='$\lambda_H=\lambda_W=1e-4$', color=cmap1(norm(3)))
    axes[i].plot(acc['Step'].values*2, acc['e-5'].values, label='$\lambda_H=\lambda_W=1e-5$', color=cmap1(norm(4)))

    # axes[i].plot(acc['Step'].values, test_acc['e-2'].values, label='$\lambda_H=\lambda_W=1e-2$', color=cmap1(norm(1)), linestyle='--')
    # axes[i].plot(acc['Step'].values, test_acc['e-3'].values, label='$\lambda_H=\lambda_W=1e-3$', color=cmap1(norm(2)), linestyle='--')
    # axes[i].plot(acc['Step'].values, test_acc['e-4'].values, label='$\lambda_H=\lambda_W=1e-4$', color=cmap1(norm(3)), linestyle='--')
    # axes[i].plot(acc['Step'].values, test_acc['e-5'].values, label='$\lambda_H=\lambda_W=1e-5$', color=cmap1(norm(4)), linestyle='--')

    # axes[i].plot(acc['Step'].values, 1 - acc['ls0.5'].values, label='$\delta=0.5$', color=cmap1(norm(7)))
    # axes[i].plot(np.arange(800), 1 - acc['ls0.95'].values, label='$\delta=0.95$', color=cmap1(norm(8)))
    axes[i].set_ylabel('Train MSE')
    axes[i].set_xlabel('Epoch')
    axes[i].grid(True, linestyle='--')
    # axes[i].set_yscale("log")

    i = row + '1'
    # ===== NC1
    axes[i].plot(nc1['Step'].values*2, nc1['e-2'].values, label='$\lambda_H=\lambda_W=1e-2$', color=cmap1(norm(1)))
    axes[i].plot(nc1['Step'].values*2, nc1['e-3'].values, label='$\lambda_H=\lambda_W=1e-3$', color=cmap1(norm(2)))
    axes[i].plot(nc1['Step'].values*2, nc1['e-4'].values, label='$\lambda_H=\lambda_W=1e-4$', color=cmap1(norm(3)))
    axes[i].plot(nc1['Step'].values*2, nc1['e-5'].values, label='$\lambda_H=\lambda_W=1e-5$', color=cmap1(norm(4)))
    axes[i].set_ylabel('NRC1')
    axes[i].set_yscale("log")
    axes[i].set_xlabel('Epoch')
    axes[i].grid(True, linestyle='--')

    # ===== NC2
    i = row + '2'
    axes[i].plot(nc3['Step'].values*2, nc3['e-2'].values, label='$\lambda_H=\lambda_W=1e-2$', color=cmap1(norm(1)))
    axes[i].plot(nc3['Step'].values*2, nc3['e-3'].values, label='$\lambda_H=\lambda_W=1e-3$', color=cmap1(norm(2)))
    axes[i].plot(nc3['Step'].values*2, nc3['e-4'].values, label='$\lambda_H=\lambda_W=1e-4$', color=cmap1(norm(3)))
    axes[i].plot(nc3['Step'].values*2, nc3['e-5'].values, label='$\lambda_H=\lambda_W=1e-5$', color=cmap1(norm(4)))
    # axes[i].plot(nc1['Step'].values - 1, nc3['ls0.5'].values, label='$\delta=0.5$', color=cmap1(norm(7)))
    # axes[i].plot(nc1['Step'].values - 1, nc3['ls0.95'].values, label='$\delta=0.95$', color=cmap1(norm(8)))
    axes[i].set_ylabel('NRC2')
    axes[i].set_xlabel('Epoch')
    axes[i].grid(True, linestyle='--')
    # axes[i].set_yscale("log")
    # axes[i].set_xlim(0, 800)
    # axes[i].set_xticks([0, 200, 400, 600, 800])



    i = row + '3'
    # ===== NC3
    axes[i].plot(nc2['Step'].values*2, nc2['e-2'].values, label='$\lambda_H=\lambda_W=1e-2$', color=cmap1(norm(1)))
    axes[i].plot(nc2['Step'].values*2, nc2['e-3'].values, label='$\lambda_H=\lambda_W=1e-3$', color=cmap1(norm(2)))
    axes[i].plot(nc2['Step'].values*2, nc2['e-4'].values, label='$\lambda_H=\lambda_W=1e-4$', color=cmap1(norm(3)))
    axes[i].plot(nc2['Step'].values*2, nc2['e-5'].values, label='$\lambda_H=\lambda_W=1e-5$', color=cmap1(norm(4)))
    #axes[i].plot(nc1['Step'].values - 1, nc2['ls0.5'].values, label='$\delta=0.5$', color=cmap1(norm(7)))
    #axes[i].plot(nc1['Step'].values - 1, nc2['ls0.95'].values, label='$\delta=0.95$', color=cmap1(norm(8)))

    axes[i].set_ylabel('NRC3')
    axes[i].set_xlabel('Epoch')
    axes[i].grid(True, linestyle='--')
    # axes[i].set_xlim(0, 800)
    # axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))





    # i = row + '4'
    # # ==== Testing Acc
    # axes[i].plot(acc['Step'].values, 1 - test_acc['ls0'].values, label='$\delta=0$', color=cmap1(norm(1)))
    # axes[i].plot(acc['Step'].values, 1 - test_acc['ls0.05'].values, label='$\delta=0.05$', color=cmap1(norm(2)))
    # axes[i].plot(acc['Step'].values, 1 - test_acc['ls0.1'].values, label='$\delta=0.1$', color=cmap1(norm(3)))
    # axes[i].plot(acc['Step'].values, 1 - test_acc['ls0.2'].values, label='$\delta=0.2$', color=cmap1(norm(4)))
    # axes[i].plot(acc['Step'].values, 1 - test_acc['ls0.3'].values, label='$\delta=0.3$', color=cmap1(norm(5)))
    # axes[i].plot(acc['Step'].values, 1 - test_acc['ls0.4'].values, label='$\delta=0.4$', color=cmap1(norm(6)))
    # #axes[i].plot(acc['Step'].values, 1 - test_acc['ls0.5'].values, label='$\delta=0.5$', color=cmap1(norm(7)))
    # #axes[i].plot(np.arange(800), 1 - test_acc['ls0.95'].values, label='$\delta=0.95$', color=cmap1(norm(8)))
    # axes[i].set_ylabel('Test Error Rate')
    # axes[i].set_xlabel('Epoch')
    # axes[i].grid(True, linestyle='--')
    # axes[i].set_yscale("log")

    # axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #fig.suptitle("Model Convergence with Different Smoothing Hyperparameter")