import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic

def binned_stats(sim_dict, bin_size=25):

    tfin = sim_dict['simulation_params']['tfin']
    num_bins = tfin / bin_size

    bin_ranges = np.arange(0, tfin, bin_size) + bin_size / 2

    binned_mean, bin_edges, binnumber = binned_statistic(sim_dict['t_span_save'], sim_dict['phi_save'][:, 2], statistic='mean', bins=num_bins)
    binned_std, bin_edges, binnumber = binned_statistic(sim_dict['t_span_save'], sim_dict['phi_save'][:, 2], statistic='std', bins=num_bins)

    # 2. Create Figure and Axes
    fig, ax = plt.subplots(figsize=(6, 4))  # Create a figure and a single axes, set figure size

    # 3. Plot Data
    ax.plot(bin_ranges, binned_mean, label=f'Mean', color='skyblue', linewidth=2, linestyle='-')
    ax.plot(bin_ranges, binned_std, label=f'Standard Deviation', color='green', linewidth=2, linestyle='-')

    # 4. Customize Plot Elements
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(r'$\phi_b$', fontsize=12)

    ax.legend(fontsize=10, loc='upper right')  # Add a legend
    ax.grid(True, linestyle=':', alpha=0.7)  # Add a grid

    # Customize tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10, length=5, width=1)

    # Remove top and right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 5. Display the Plot
    plt.tight_layout()  # Adjust plot to prevent labels from overlapping
    plt.show()
