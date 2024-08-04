from plots.quick_plot_helper import *
from plots.log_alias import *
import matplotlib.pyplot as plt
import matplotlib.image as pltimg


DEFAULT_BASE_PATH = './results'
DEFAULT_SAVE_PATH = './figures/'
DEFAULT_ENVS = ('hopper',)
DEFAULT_LINESTYLES = tuple(['solid' for _ in range(8)])
DEFAULT_COLORS = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:purple')
DEFAULT_Y_VALUE = 'AverageTestEpRet'
DEFAULT_SMOOTH = 1

data_folder = './results'
save_folder = './figures/'

y_use_log = None

dataset_group = [['reacher', 'swimmer', 'hopper'],
                 ['carla2d', 'carla1d', 'age']]
variant = [[reacher_final, swimmer_final, hopper_final],
            [carla2d_final, carla1d_final, age_final]]
use_ratio = [((3/8, 1/40, 5/8), (0.25, 1/40, 0.25)), None]
# dataset_group = [['reacher', 'swimmer', 'hopper', 'carla2d', 'carla1d', 'age']]
# variant_group = [[reacher_final, swimmer_final, hopper_final, carla2d_final, carla1d_final, age_final]]
# use_ratio = [((3/8, 1/40, 5/8, 1, 1, 1), (0.25, 1/40, 0.25, 1, 1, 1))]
measure_group = [['NRC1', 'NRC2', 'NRC3_epoch'], ['trainError', 'testError']]
legend_group = [['NRC1', 'NRC2', 'NRC3'], ['Train MSE', 'Test MSE']]
xlabel = 'Epoch'
ylabel = ['NRC1-3', 'Accuracy']
# use_ratio = [None, None]


data_plot = {'reacher': {}, 'swimmer':{}, 'hopper':{}, 'carla2d':{}, 'carla1d':{}, 'age':{}}
for l, datasets in enumerate(dataset_group):
    for i, y_values in enumerate(measure_group):
        for j, dataset in enumerate(datasets):
            full_path = os.path.join(DEFAULT_BASE_PATH, dataset, variant[l][j])
            print("Loading data from:", full_path)

            # ax = axes[2 * l + i][j] if num_rows > 1 else axes[j]  # When num_rows = 1, ax is 1D array.
            for k, y_metric in enumerate(y_values):
                if dataset in ['carla1d', 'age'] and y_metric == 'NRC3_epoch':
                    continue

                print('1===============')
                if dataset in ['reacher', 'swimmer', 'hopper']:
                    csv_path = os.path.join(full_path, y_metric + '.csv')
                    with open(csv_path, 'r') as f:
                        df = pd.read_csv(f, delimiter=",", header=0)
                        if y_metric in ['NRC1', 'NRC2', 'trainError', 'testError']:
                            x_to_plot = df.iloc[:, 0].to_numpy().reshape(-1)
                            y_to_plot = df.iloc[:, 1].to_numpy().reshape(-1)
                        elif y_metric in ['NRC3_c']:
                            x_to_plot = df["c"].to_numpy().reshape(-1)
                            x_to_plot = x_to_plot / x_to_plot.max()
                            y_to_plot = df["NRC3"].to_numpy().reshape(-1)
                        elif y_metric in ['NRC3_epoch']:
                            x_to_plot = df["epoch"].to_numpy().reshape(-1)
                            y_to_plot = df["NRC3"].to_numpy().reshape(-1)
                    print('2===============')

                elif dataset in ['carla2d']:
                    print('3===============')
                    if y_metric in ['NRC1', 'NRC2', 'NRC3_epoch', 'trainError', 'testError']:
                        csv_path = os.path.join(full_path, 'Carla2D_100ep.csv')
                        with open(csv_path, 'r') as f:
                            df = pd.read_csv(f, delimiter=",", header=0)
                            x_to_plot = df['Epoch']
                            if y_metric in ['NRC1', 'NRC2']:
                                y_to_plot = df[y_metric]
                            elif y_metric == 'NRC3_epoch':
                                y_to_plot = df['NRC3']
                            elif y_metric == 'trainError':
                                y_to_plot = df['Train Loss']
                            elif y_metric == 'testError':
                                y_to_plot = df['Test Loss']
                    elif y_metric in ['NRC3_c']:
                        csv_path = os.path.join(full_path, 'Carla2D_gamma.csv')
                        with open(csv_path, 'r') as f:
                            df = pd.read_csv(f, delimiter=",", header=0)
                            x_to_plot = df['gamma']
                            x_to_plot = x_to_plot / x_to_plot.max()
                            y_to_plot = df["NRC3"]

                elif dataset in ['carla1d']:
                    if y_metric in ['NRC1', 'NRC2', 'NRC3_epoch', 'trainError', 'testError']:
                        csv_path = os.path.join(full_path, 'Carla1D_100ep.csv')
                        with open(csv_path, 'r') as f:
                            df = pd.read_csv(f, delimiter=",", header=0)
                            x_to_plot = df['Epoch']
                            if y_metric in ['NRC1', 'NRC2']:
                                y_to_plot = df[y_metric]
                            elif y_metric == 'NRC3_epoch':
                                y_to_plot = df['NRC3']
                            elif y_metric == 'trainError':
                                y_to_plot = df['Train Loss']
                            elif y_metric == 'testError':
                                y_to_plot = df['Test Loss']

                elif dataset in ['age']:
                    if y_metric in ['NRC1', 'NRC2', 'NRC3_epoch', 'trainError', 'testError']:
                        csv_path = os.path.join(full_path, 'Age1D.csv')
                        with open(csv_path, 'r') as f:
                            df = pd.read_csv(f, delimiter=",", header=0)
                            x_to_plot = df['Epoch']
                            if y_metric in ['NRC1', 'NRC2']:
                                y_to_plot = df[y_metric]
                            elif y_metric == 'NRC3_epoch':
                                y_to_plot = df['NRC3']
                            elif y_metric == 'trainError':
                                y_to_plot = df['Train Loss']
                            elif y_metric == 'testError':
                                y_to_plot = df['Test Loss']

                print('3===============')

                # Want to make x-axis length of mujoco datasets uniform with 1000

                # Mujoco exps forget to square the nrc3
                if dataset in ['reacher', 'hopper', 'swimmer'] and y_metric == 'NRC3_epoch':
                    y_to_plot = y_to_plot ** 2


                print('4===============')


                data_plot[dataset][y_metric] ={'x':x_to_plot, 'y':y_to_plot}

                # ==================== dataset have been extracted ====================

num_columns = len(dataset_group[0])
num_rows = len(measure_group) * len(dataset_group)

# data_plot['reacher']['NRC1']['x'] = data_plot['reacher']['NRC1']['x']/12
# data_plot['reacher']['NRC2']['x'] = data_plot['reacher']['NRC2']['x']/12
# data_plot['reacher']['NRC3_epoch']['x'] = data_plot['reacher']['NRC3_epoch']['x']/12
# data_plot['reacher']['trainError']['x'] = data_plot['reacher']['trainError']['x']/12
# data_plot['reacher']['testError']['x'] = data_plot['reacher']['testError']['x']/12

for dataset in ['reacher', 'hopper', 'swimmer']:
    for k in data_plot[dataset].keys():
        if k == 'NRC3_c':
            continue
        total_size = dataset[k]['x'].shape[0]
        num_points = 200
        gap = total_size // num_points
        dataset[k]['x'] = dataset[k]['x'][::gap]
        dataset[k]['y'] = dataset[k]['y'][::gap]

# ==================== =========================== ====================
# fig, axes = plt.subplots(num_rows, num_columns, )
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(10, 8))

gs = fig.add_gridspec(2, 1, hspace=0.27)  # spacing between the two groups

gs0 = gs[0].subgridspec(2, 3, hspace=0.2)  # spacing within the groups
gs1 = gs[1].subgridspec(2, 3, hspace=0.2)

data_name = {'reacher': "Reacher", 'swimmer': 'Swimmer', 'hopper':"Hopper", 'carla2d':'Carla 2D', 'carla1d':'Carla 1D', 'age':'UTKFace'}

for i, dataset in enumerate(['reacher', 'swimmer', 'hopper', 'carla2d', 'carla1d', 'age']):

    #==== first plot nrc
    row = 0 if dataset in ['reacher', 'swimmer', 'hopper'] else 2
    if dataset in ['reacher', 'swimmer', 'hopper']:
        ax = fig.add_subplot(gs0[0, i%3])
    else:
        ax = fig.add_subplot(gs1[0, i % 3])

    if dataset in ['reacher', 'swimmer', 'hopper', 'carla2d']:
        ax.plot(data_plot[dataset]['NRC1']['x'], data_plot[dataset]['NRC1']['y'], color='C0', label='NRC1')
        ax.plot(data_plot[dataset]['NRC2']['x'], data_plot[dataset]['NRC2']['y'], color='C0', label='NRC2', linestyle='--')
        ax.tick_params(axis='y', colors='C0')
        #ax.set_ylabel('NRC1/NRC2', color='C0')

        ax2 = ax.twinx()
        ax2.plot(data_plot[dataset]['NRC3_epoch']['x'], data_plot[dataset]['NRC3_epoch']['y'], color='C1', label='NRC3')
        ax2.tick_params(axis='y', colors='C1')
        # ax2.set_ylabel('NRC3', color='C1')

        handles, labels = [], []
        for a in [ax, ax2]:
            for h, l in zip(*a.get_legend_handles_labels()):
                handles.append(h)
                labels.append(l)
        ax.legend(handles, labels)
        ax.grid(True, linestyle='--')
        if dataset == 'reacher':
            ax.set_xticklabels(['0', '0.4M', '0.8M', '1.2M'])
        if dataset == 'swimmer':
            ax.set_xticklabels(['0', '200K', '400K', '600K', '800K'])
        if dataset == 'hopper':
            ax.set_xticklabels(['0', '40K', '80K', '120K'])

    else:
        ax.plot(data_plot[dataset]['NRC1']['x'], data_plot[dataset]['NRC1']['y'], color='C0', label='NRC1')
        ax.plot(data_plot[dataset]['NRC2']['x'], data_plot[dataset]['NRC2']['y'], color='C0', label='NRC2', linestyle='--')
        # ax.set_ylabel('NRC1/NRC2')
        ax.grid(True, linestyle='--')
        ax.legend()
    ax.set_title(data_name[dataset], fontweight='bold')

    # ==== then plot Train and test MSE
    row = 1 if dataset in ['reacher', 'swimmer', 'hopper'] else 3
    if dataset in ['reacher', 'swimmer', 'hopper']:
        ax = fig.add_subplot(gs0[1, i % 3])
    else:
        ax = fig.add_subplot(gs1[1, i % 3])
    ax.plot(data_plot[dataset]['trainError']['x'], data_plot[dataset]['trainError']['y'], color='C0', label='Train MSE')
    ax.plot(data_plot[dataset]['testError']['x'], data_plot[dataset]['testError']['y'], color='C1', label='Test MSE')
    ax.legend()
    # ax.set_ylabel('MSE')
    ax.grid(True, linestyle='--')
    if dataset == 'reacher':
        ax.set_xticklabels(['0', '0.4M', '0.8M', '1.2M'])
    if dataset == 'swimmer':
        ax.set_xticklabels(['0', '200K', '400K', '600K', '800K'])
    if dataset == 'hopper':
        ax.set_xticklabels(['0', '40K', '80K', '120K'])
    if dataset=='carla1d':
        # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        # ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        # ax.get_yaxis().set_major_formatter(plt.LogFormatter(10, labelOnlyBase=False))
        ax.set_yscale("log")
    ax.set_xlabel('Epoch')

# ==================== =========================== ====================
# fig, axes = plt.subplots(num_rows, num_columns, )
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(10, 8))

gs = fig.add_gridspec(2, 1, hspace=0.3)  # spacing between the two groups

gs0 = gs[0].subgridspec(2, 3, hspace=0.2)  # spacing within the groups
gs1 = gs[1].subgridspec(2, 3, hspace=0.2)

data_name = {'reacher': "Reacher", 'swimmer': 'Swimmer', 'hopper':"Hopper", 'carla2d':'Carla 2D', 'carla1d':'Carla 1D', 'age':'UTKFace'}

for i, dataset in enumerate(['reacher', 'swimmer', 'hopper', 'carla2d', 'carla1d', 'age']):

    #==== first plot nrc
    row = 0 if dataset in ['reacher', 'swimmer', 'hopper'] else 2
    if dataset in ['reacher', 'swimmer', 'hopper']:
        ax = fig.add_subplot(gs0[0, i%3])
    else:
        ax = fig.add_subplot(gs1[0, i % 3])

    if dataset in ['reacher', 'swimmer', 'hopper', 'carla2d']:
        ax.plot(data_plot[dataset]['NRC1']['x'], data_plot[dataset]['NRC1']['y'], color='C0', label='NRC1')
        ax.plot(data_plot[dataset]['NRC2']['x'], data_plot[dataset]['NRC2']['y'], color='C0', label='NRC2', linestyle='--')
        ax.tick_params(axis='y', colors='C0')
        ax.set_ylabel('NRC1/NRC2', color='C0')

        ax2 = ax.twinx()
        ax2.plot(data_plot[dataset]['NRC3_epoch']['x'], data_plot[dataset]['NRC3_epoch']['y'], color='C1', label='NRC3')
        ax2.tick_params(axis='y', colors='C1')
        ax2.set_ylabel('NRC3', color='C1')

        handles, labels = [], []
        for a in [ax, ax2]:
            for h, l in zip(*a.get_legend_handles_labels()):
                handles.append(h)
                labels.append(l)
        ax.legend(handles, labels)
        ax.grid(True, linestyle='--')

        if dataset == 'reacher':
            ax.set_xticklabels(['0', '0.4M', '0.8M', '1.2M'])
        if dataset == 'swimmer':
            ax.set_xticklabels(['0', '200K', '400K', '600K', '800K'])
        if dataset == 'hopper':
            ax.set_xticklabels(['0', '40K', '80K', '120K'])

    else:
        ax.plot(data_plot[dataset]['NRC1']['x'], data_plot[dataset]['NRC1']['y'], color='C0', label='NRC1')
        ax.plot(data_plot[dataset]['NRC2']['x'], data_plot[dataset]['NRC2']['y'], color='C0', label='NRC2', linestyle='--')
        ax.set_ylabel('NRC1/NRC2')
        ax.grid(True, linestyle='--')
    ax.set_title(data_name[dataset], fontweight='bold')

    # ==== then plot Train and test MSE
    row = 1 if dataset in ['reacher', 'swimmer', 'hopper'] else 3
    if dataset in ['reacher', 'swimmer', 'hopper']:
        ax = fig.add_subplot(gs0[1, i % 3])
    else:
        ax = fig.add_subplot(gs1[1, i % 3])
    ax.plot(data_plot[dataset]['trainError']['x'], data_plot[dataset]['trainError']['y'], color='C0', label='Train MSE')
    ax.plot(data_plot[dataset]['testError']['x'], data_plot[dataset]['testError']['y'], color='C1', label='Test MSE')
    ax.legend()
    ax.set_ylabel('MSE')
    ax.grid(True, linestyle='--')
    
    if dataset == 'reacher':
        ax.set_xticklabels(['0', '0.4M', '0.8M', '1.2M'])
    if dataset == 'swimmer':
        ax.set_xticklabels(['0', '200K', '400K', '600K', '800K'])
    if dataset == 'hopper':
        ax.set_xticklabels(['0', '40K', '80K', '120K'])

    if dataset=='carla1d':
        # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
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



# Create a figure
fig = plt.figure(figsize=(10, 8))

# Create a GridSpec with 4 rows and 3 columns
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(4, 3, hspace=0.4)

# Add subplots in the first section (first two rows)
for i in range(2):
    for j in range(3):
        ax = fig.add_subplot(gs[i, j])
        ax.set_title(f'Section 1, Subplot {i * 3 + j + 1}')

# Add subplots in the second section (last two rows)
for i in range(2, 4):
    for j in range(3):
        ax = fig.add_subplot(gs[i, j])
        ax.set_title(f'Section 2, Subplot {(i - 2) * 3 + j + 1}')

# Adjust the vertical spacing between the two sections
plt.subplots_adjust(hspace=0.6)

plt.show()


