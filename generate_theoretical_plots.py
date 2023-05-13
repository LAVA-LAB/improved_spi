import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.special import betaincinv
from guarantee_functions import compute_Nwedge_beta
import matplotlib as mpl

plt.style.use("matplotlibrc")
mpl.rcParams['figure.figsize'] = '2.15, 1.25'
mpl.rcParams['legend.borderpad'] = 0.5

vmax = 1
gamma = 0.95
delta = 0.1
zeta = 0.1
num_actions = 4

num_plot_points = 100

num_states = np.logspace(start=2, stop=7, num=num_plot_points)
nwedges = np.zeros(num_plot_points)
nwedges_2s = np.zeros(num_plot_points)
nwedges_beta = np.zeros(num_plot_points)

for i, s in enumerate(num_states):
    nwedges[i] = 32*vmax**2 / (zeta**2 * (1-gamma)**2) * math.log(2 * s * num_actions * 2**s / delta)
    nwedges_2s[i] = 32*vmax**2 / (zeta**2 * (1-gamma)**2) * math.log(8*s**2 * num_actions**2 / delta)
    nwedges_beta[i] = compute_Nwedge_beta(zeta, delta, gamma, vmax, s, num_actions, 1_000_000, 0, 0)

# setting the axes at the centre
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_ylim(bottom=0, top=max(nwedges_2s))
ax.set_xscale('log')
ax.set_xlabel('$|S|$')

# plot the function
plt.plot(num_states, nwedges_2s, color='#2ca02c', label='$N_{\\wedge}^{2s}$')
plt.plot(num_states, nwedges_beta, color='#d62728', label='$N_{\\wedge}^{\\beta}$')
plt.legend(loc="lower right")

ax.figure.savefig("plots/nwedges_zoomed.pdf", bbox_inches='tight', pad_inches=0.02)

inf_filter = nwedges < float('inf')
ax.set_ylim(bottom=0, top=max(nwedges[inf_filter]))
plt.plot(num_states, nwedges, color='#ff7f0e', label='$N_{\\wedge}^{\mathit{SPIBB}}$')
plt.legend(loc="right")

ax.figure.savefig("plots/nwedges.pdf", bbox_inches='tight', pad_inches=0.02)
