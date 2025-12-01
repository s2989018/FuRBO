# Script to evaluate BBOB on COBYLA

##########
# Imports
import matplotlib
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import patches

def plot_convergence(crv, ax):
    
    cwd = os.path.join(os.getcwd(), 'Experiments', 'FuRBO')
    
    
    # Load data
    crv = crv[:-5] + '1' + crv[-4:]
    if crv in os.listdir(cwd):
        file_name = os.path.join(cwd, crv, '01_Y_mono.npy')
        y_F = np.load(file_name) - fmin[crv]
        y_f_max = np.amax(y_F)
        y_F[y_F==np.amax(y_F)] = y_f_max
        
    crv = crv[:-5] + '2' + crv[-4:]
    if crv in os.listdir(cwd):
        file_name = os.path.join(cwd, crv, '01_Y_mono.npy')
        tmp = np.load(file_name) - fmin[crv]
        if y_f_max < np.amax(tmp):
            y_f_max = np.amax(tmp)
            y_F[y_F==np.amax(y_F)] = y_f_max
        else:
            tmp[tmp==np.amax(tmp)] = y_f_max
        y_F = np.vstack([y_F, tmp])
        
    crv = crv[:-5] + '3' + crv[-4:]
    if crv in os.listdir(cwd):
        file_name = os.path.join(cwd, crv, '01_Y_mono.npy')
        tmp = np.load(file_name) - fmin[crv]
        if y_f_max < np.amax(tmp):
            y_f_max = np.amax(tmp)
            y_F[y_F==np.amax(y_F)] = y_f_max
        else:
            tmp[tmp==np.amax(tmp)] = y_f_max
        y_F = np.vstack([y_F, tmp])
        
        
    cwd = os.path.join(os.getcwd(), 'Experiments', 'novel_FuRBO')
    
    # Load data
    crv = crv[:-5] + '1' + crv[-4:]
    if crv in os.listdir(cwd):
        file_name = os.path.join(cwd, crv, '01_Y_mono.npy')
        y_S = np.load(file_name) - fmin[crv]
        y_s_max = np.amax(y_S)
        y_S[y_S==np.amax(y_S)] = y_s_max
        
    crv = crv[:-5] + '2' + crv[-4:]
    if crv in os.listdir(cwd):
        file_name = os.path.join(cwd, crv, '01_Y_mono.npy')
        tmp = np.load(file_name) - fmin[crv]
        if y_s_max < np.amax(tmp):
            y_s_max = np.amax(tmp)
            y_S[y_S==np.amax(y_S)] = y_s_max
        else:
            tmp[tmp==np.amax(tmp)] = y_s_max
        y_S = np.vstack([y_S, tmp])
        
    crv = crv[:-5] + '3' + crv[-4:]
    if crv in os.listdir(cwd):
        file_name = os.path.join(cwd, crv, '01_Y_mono.npy')
        tmp = np.load(file_name) - fmin[crv]
        if y_s_max < np.amax(tmp):
            y_s_max = np.amax(tmp)
            y_S[y_S==np.amax(y_S)] = y_s_max
        else:
            tmp[tmp==np.amax(tmp)] = y_s_max
        y_S = np.vstack([y_S, tmp])


    cwd = os.path.join(os.getcwd(), 'Experiments', 'botorch')

    # Load data
    crv_seed = crv[:-5] + '1' + crv[-4:]
    if crv_seed in os.listdir(cwd):
        file_name = os.path.join(cwd, crv_seed, '01_Y_mono.npy')
        y_B = np.load(file_name) - fmin[crv]
        y_b_max = np.amax(y_B)
        y_B[y_B == np.amax(y_B)] = y_b_max

    crv_seed = crv[:-5] + '2' + crv[-4:]
    if crv_seed in os.listdir(cwd):
        file_name = os.path.join(cwd, crv_seed, '01_Y_mono.npy')
        tmp = np.load(file_name) - fmin[crv]
        if y_b_max < np.amax(tmp):
            y_b_max = np.amax(tmp)
            y_B[y_B == np.amax(y_B)] = y_b_max
        else:
            tmp[tmp == np.amax(tmp)] = y_b_max
        y_B = np.vstack([y_B, tmp])

    crv_seed = crv[:-5] + '3' + crv[-4:]
    if crv_seed in os.listdir(cwd):
        file_name = os.path.join(cwd, crv_seed, '01_Y_mono.npy')
        tmp = np.load(file_name) - fmin[crv]
        if y_b_max < np.amax(tmp):
            y_b_max = np.amax(tmp)
            y_B[y_B == np.amax(y_B)] = y_b_max
        else:
            tmp[tmp == np.amax(tmp)] = y_b_max
        y_B = np.vstack([y_B, tmp])


    
    # Find worst feasible
    y_max = max(np.amax(y_F), np.amax(y_S), np.amax(y_B))
    y_F[y_F==np.amax(y_F)] = y_max
    y_S[y_S==np.amax(y_S)] = y_max
    y_B[y_B==np.amax(y_B)] = y_max

        
    # Elaborate FuRBO data
    mean = np.mean(y_F, axis = 0)
    lb = mean - np.std(y_F, axis = 0)/np.sqrt(y_F.shape[0])
    ub = mean + np.std(y_F, axis = 0)/np.sqrt(y_F.shape[0])
    x = np.linspace(1, len(mean), len(mean))
    
    # Plot convergence of FuRBO
    ax.plot(x, mean, color = 'darkorange', lw=2)
    ax.fill_between(x, lb, ub, alpha = 0.2, color='darkorange', lw=2)
    top = np.amax(ub) + 0.1 * np.amax(ub)
    middle = np.amax(mean)/2
        
    # Elaborate SCBO data
    mean = np.mean(y_S, axis = 0)
    lb = mean - np.std(y_S, axis = 0)/np.sqrt(y_S.shape[0])
    ub = mean + np.std(y_S, axis = 0)/np.sqrt(y_S.shape[0])
    x = np.linspace(1, len(mean), len(mean))
    
    # Plot convergence of SCBO
    ax.plot(x, mean, color = 'darkgreen', lw=2)
    ax.fill_between(x, lb, ub, alpha = 0.2, color='darkgreen', lw=2)
    if np.amax(ub) + 0.1 * np.amax(ub) > top:
        top = np.amax(ub) + 0.1 * np.amax(ub)
    if np.amax(mean)/2 > middle:
        middle = np.amax(mean)/2

    # Elaborate BoTorch data
    mean = np.mean(y_B, axis=0)
    lb = mean - np.std(y_B, axis=0)/np.sqrt(y_B.shape[0])
    ub = mean + np.std(y_B, axis=0)/np.sqrt(y_B.shape[0])
    x = np.linspace(1, len(mean), len(mean))

    # Plot convergence of BoTorch
    ax.plot(x, mean, color='darkblue', lw=2)
    ax.fill_between(x, lb, ub, alpha=0.2, color='darkblue', lw=2)
    if np.amax(ub) + 0.1 * np.amax(ub) > top:
        top = np.amax(ub) + 0.1 * np.amax(ub)
    if np.amax(mean)/2 > middle:
        middle = np.amax(mean)/2
        
    ax.set_ylim(bottom = 0,
                top = top)
    
    ax.set_yticks([top,
                   middle,
                   0])
    
    ax.set_yticklabels([f"{(top):.1E}",
                        f"{(middle):.1E}",
                        "0.0"], rotation=45)
        
    return

##########
# Initialize plot
matplotlib.use('Agg')
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

fig = plt.figure(figsize=(16,8))
gs = gridspec.GridSpec(nrows=3,
                       ncols=6,
                       wspace=.4,
                       hspace=.4,
                       height_ratios=[0.5, 8, 8],
                       figure = fig)


patchList = []

fmin = np.load('bbob-constrained-targets.npy', allow_pickle = True)
fmin = fmin.item(0)

##########
# Iterate through the curves and plot them

crv = "bbob-constrained_f002_i01_d02"
ax = plt.subplot(gs[1, 1])
plot_convergence(crv, ax)
ax.set_xticks([0, 50, 100, 150])
ax.set_xticklabels([])
ax.set_title("Constraints: 3\n"
             "Active: 2")



crv = "bbob-constrained_f004_i01_d02"
ax = plt.subplot(gs[1, 3])
plot_convergence(crv, ax)
ax.set_xticks([0, 50, 100, 150])
ax.set_xticklabels([])
ax.set_title("Constraints: 17\n"
             "Active: 11")


crv = "bbob-constrained_f006_i01_d02"
ax = plt.subplot(gs[1, 5])
plot_convergence(crv, ax)
ax.set_xticks([0, 50, 100, 150])
ax.set_xticklabels([])
ax.set_title("Constraints: 54\n"
             "Active: 36")

crv = "bbob-constrained_f050_i01_d02"
ax = plt.subplot(gs[2, 1])
plot_convergence(crv, ax)
ax.set_xticks([0, 50, 100, 150])
ax.set_xticklabels(['0', '50', '100', '150'], rotation=45)
ax.set_xlabel('Evaluations')


crv = "bbob-constrained_f052_i01_d02"
ax = plt.subplot(gs[2, 3])
plot_convergence(crv, ax)
ax.set_xticks([0, 50, 100, 150])
ax.set_xticklabels(['0', '50', '100', '150'], rotation=45)
ax.set_xlabel('Evaluations', loc = 'left')


crv = "bbob-constrained_f054_i01_d02"
ax = plt.subplot(gs[2, 5])
plot_convergence(crv, ax)
ax.set_xticks([0, 50, 100, 150])
ax.set_xticklabels(['0', '50', '100', '150'], rotation=45)
ax.set_xlabel('Evaluations')

# Add legend
ax = plt.subplot(gs[0, :])
patchList = [
    patches.Patch(color='darkorange', label='FuRBO'),
    patches.Patch(color='darkgreen', label='novel_FuRBO'),
    patches.Patch(color='darkblue', label='BoTorch')  
]
ax.legend(ncols=3, handles=patchList, loc='center')
ax.axis('off')  # hide axes for legend



# Save figure
fig.savefig(os.path.join(os.getcwd(), 'FuRBOtwoDim' + '.png'), dpi=600)
    
# Close figure
plt.close(fig)

