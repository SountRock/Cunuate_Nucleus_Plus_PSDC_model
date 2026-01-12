'''
Storage of all specific and non-time-resolved phase analyses and the plots generated from them
'''

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import json
from collections import defaultdict
import pickle
import numpy as np
import os
from scipy.stats import gaussian_kde
from scipy.signal import spectrogram
from scipy.ndimage import gaussian_filter
import pycwt as wavelet
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
from mne_connectivity import phase_slope_index, seed_target_indices
import re


#-------- Other analyses and graphs --------------------------------------
def get_spike_phases(spike_times, t_min, t_max, period):
    # Convert input spike times to NumPy array for vectorized operations
    spike_times = np.array(spike_times)

    # Select only spikes that fall within the specified time window [t_min, t_max]
    spikes_in_range = spike_times[(spike_times >= t_min) & (spike_times <= t_max)]

    # Compute the phase of each spike within the given period:
    # 1) Shift spike times so that t_min corresponds to phase 0
    # 2) Take modulo with respect to the period to fold time into cycles
    # 3) Normalize to the range [0, 2π)
    phases = 2 * np.pi * ((spikes_in_range - t_min) % period) / period

    # Return spike phases in radians
    return phases

'''
The function visualizes the phase times of neuronal spikes, 
grouping them into populations (neuron groups), on a polar diagram (circular visualization).
'''
def plot_phase_circles_groups(neuron_groups, period=None, base_radius=1.0, radius_step=0.3, dir='res', group_names=None):
    # Collect all spike times into a single array to determine the global time range
    all_spikes = []
    for group in neuron_groups:
        for neuron in group:
            all_spikes.extend(neuron)
    all_spikes = np.array(all_spikes)

    # Determine global minimum and maximum spike times
    t_min = np.min(all_spikes)
    t_max = np.max(all_spikes)

    # If no period is provided, use the full time span as the period
    if period is None:
        period = t_max - t_min

    num_groups = len(neuron_groups)

    # Generate default group names if not provided or mismatched in length
    if group_names is None or len(group_names) != num_groups:
        group_names = [f'Group {i+1}' for i in range(num_groups)]

    # Assign a distinct color to each group
    cmap = plt.colormaps['tab10']
    group_colors = [cmap(i) for i in range(num_groups)]

    # Create a polar plot
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    # Set zero phase at the top (north) and clockwise phase direction
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    current_radius = base_radius

    # Plot spike phases for each group and each neuron
    for g_idx, group in enumerate(neuron_groups):
        color = group_colors[g_idx]
        for neuron_spikes in group:
            # Convert spike times to phases in radians
            phases = get_spike_phases(neuron_spikes, t_min, t_max, period)

            # Place all spikes of this neuron on the same radial level
            r = np.full_like(phases, current_radius)

            # Scatter plot of spike phases on the polar axis
            ax.scatter(phases, r, color=color, alpha=0.7, s=50)

            # Increase radius for the next neuron
            current_radius += radius_step

        # Add an extra gap between different groups
        current_radius += radius_step * 0.7

    # Draw dashed circular guides for each neuron radius
    total_neurons = sum(len(group) for group in neuron_groups)
    radii = [base_radius + i * radius_step for i in range(total_neurons)]
    for rr in radii:
        ax.plot(
            np.linspace(0, 2 * np.pi, 360),
            [rr] * 360,
            color='gray',
            linestyle='--',
            linewidth=0.5,
            alpha=0.3
        )

    # Remove axis tick labels and ticks for a cleaner visualization
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])

    # Create legend entries for groups
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=group_names[i],
            markerfacecolor=group_colors[i],
            markersize=10
        )
        for i in range(num_groups)
    ]

    ax.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=min(num_groups, 4),
        frameon=False
    )
    ax.set_title('Phase Circles of Spike Times by Groups')

    os.makedirs(dir, exist_ok=True)
    plt.savefig(os.path.join(dir, 'Phase_plot.png'), bbox_inches='tight', dpi=300)
    plt.close()

'''
Compute and plot phase histograms for groups of neurons.
'''
def plot_phase_histogram_by_group(
    neuron_groups,
    period,
    bins=36,
    dir='res',
    filename='phase_hist_groups.png',
    group_names=None,
    group_colors=None,
    t_min_slice=None  
):
    # Obtain global minimum and maximum spike times
    all_spikes = [spike for group in neuron_groups for neuron in group for spike in neuron]
    t_min_global, t_max = np.min(all_spikes), np.max(all_spikes)

    # Use provided left slice or the global minimum
    t_min_used = t_min_global if t_min_slice is None else max(t_min_global, t_min_slice)

    n_groups = len(neuron_groups)
    if group_names is None:
        group_names = [f"Group {i+1}" for i in range(n_groups)]

    # Choose colors for groups (default colormap if not provided)
    if group_colors is None:
        cmap = plt.colormaps['tab10']
        group_colors = [cmap(i) for i in range(n_groups)]
    else:
        assert len(group_colors) >= n_groups, "Not enough colors provided in group_colors"

    # Collect histograms per group taking the time slice into account
    histograms = []
    for group in neuron_groups:
        phases = []
        for neuron in group:
            phases.extend(get_spike_phases(neuron, t_min_used, t_max, period))
        counts, _ = np.histogram(phases, bins=bins, range=(0, 2 * np.pi))
        histograms.append(np.array(counts))

    # Prepare bin centers for plotting on circular axis
    bin_edges = np.linspace(0, 2 * np.pi, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Build combined polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(7, 7))
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    title = f'Phase Histogram by Group\nTime slice: [{t_min_used:.1f}, {t_max:.1f}] ms'
    ax.set_title(title)

    for i in range(n_groups):
        ax.bar(bin_centers, histograms[i],
               width=2*np.pi/bins, bottom=0.0,
               color=group_colors[i], alpha=0.6, edgecolor='black', label=group_names[i])

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=n_groups)
    os.makedirs(dir, exist_ok=True)
    plt.savefig(os.path.join(dir, filename), dpi=300, bbox_inches='tight')
    plt.show()

    # Individual polar histograms per group
    for i in range(n_groups):
        fig_ind, ax_ind = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
        ax_ind.set_theta_zero_location('N')
        ax_ind.set_theta_direction(-1)
        ax_ind.set_title(f'{group_names[i]} Phase Histogram\nTime slice: [{t_min_used:.1f}, {t_max:.1f}] ms')

        ax_ind.bar(bin_centers, histograms[i],
                   width=2*np.pi/bins, bottom=0.0,
                   color=group_colors[i], alpha=0.8, edgecolor='black')

        plt.savefig(os.path.join(dir, f'{group_names[i]}_hist.png'), dpi=300, bbox_inches='tight')
        plt.close()


'''
Compute spike phases for neuron groups and save data needed for phase histograms into a JSON file
'''    
def save_phase_histogram_data_to_json(
    neuron_groups,
    period,
    bins,
    group_names,
    group_colors,
    t_min_slice,
    output_json_path
):
    # Determine global minimum and maximum spike times
    all_spikes = [s for g in neuron_groups for n in g for s in n]
    t_min_global = min(all_spikes)
    t_max = max(all_spikes)
    t_min_used = t_min_global if t_min_slice is None else max(t_min_slice, t_min_global)

    # Compute phases for each neuron in each group
    groups_phases = []
    for group in neuron_groups:
        group_phases = []
        for neuron in group:
            phases = get_spike_phases(neuron, t_min_used, t_max, period)
            group_phases.extend(phases)
        groups_phases.append(group_phases)

    # Prepare data dictionary for JSON
    data = {
        'phases': groups_phases,
        'period': period,
        'bins': bins,
        'group_names': group_names,
        'group_colors': group_colors,
        't_min': t_min_used,
        't_max': t_max
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=2)

'''
Load phase histogram data from a JSON file and plot polar histograms
'''
def plot_phase_histogram_from_json(json_path, save_path=None, save_individual=True):
    with open(json_path, 'r') as f:
        data = json.load(f)

    phases_groups = data['phases']
    bins = data['bins']
    group_names = data['group_names']
    group_colors = data['group_colors']
    t_min = data['t_min']
    t_max = data['t_max']

    # Compute bin edges and centers for polar plotting
    bin_edges = np.linspace(0, 2 * np.pi, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(7, 7))
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    ax.set_title(f'Phase Histogram\nTime slice: [{t_min:.1f}, {t_max:.1f}] ms')

    for i, phases in enumerate(phases_groups):
        # Compute histogram counts for each group
        counts, _ = np.histogram(phases, bins=bins, range=(0, 2 * np.pi))
        # Plot as polar bars
        ax.bar(
            bin_centers, counts,
            width=2*np.pi/bins,
            bottom=0.0,
            color=group_colors[i],
            alpha=0.6,
            edgecolor='black',
            label=group_names[i]
        )

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=len(group_names))

    # Save or show combined plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    # Individual polar plots per group 
    if save_individual:
        output_dir = os.path.dirname(save_path or json_path)
        for i, phases in enumerate(phases_groups):
            counts, _ = np.histogram(phases, bins=bins, range=(0, 2 * np.pi))

            fig_i, ax_i = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
            ax_i.set_theta_zero_location('N')
            ax_i.set_theta_direction(-1)
            ax_i.set_title(f'{group_names[i]} Phase Histogram\n[{t_min:.1f}, {t_max:.1f}] ms')

            ax_i.bar(
                bin_centers, counts,
                width=2*np.pi/bins,
                bottom=0.0,
                color=group_colors[i],
                alpha=0.8,
                edgecolor='black'
            )

            # Save each individual group plot
            file_path = os.path.join(output_dir, f'{group_names[i].replace(" ", "_")}_hist.png')
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()

'''
Analysis of phase histogram differences helps to understand how the signal is amplified in phase or how stable it is. 
There are two types of bars: blue for amplification and red for attenuation. 
This data is currently stored in json for better manual viewing.
Sometimes a simple analysis reflects reality better than a complex one
'''
def save_phase_difference_data_to_json(
    target_spike_trains,
    reference_spike_trains,
    period,
    bins,
    t_min_slice,
    output_json_path
):
    # Combine all spike times to determine overall time window
    all_spikes = [s for neuron in target_spike_trains + reference_spike_trains for s in neuron]
    t_min_global = min(all_spikes)
    t_max = max(all_spikes)
    t_min_used = max(t_min_global, t_min_slice) if t_min_slice else t_min_global

    # Helper function to compute phases for a list of spike trains
    def flatten_phases(spike_trains):
        all_phases = []
        for neuron in spike_trains:
            all_phases.extend(get_spike_phases(neuron, t_min_used, t_max, period))
        return all_phases

    # Compute phases for target and reference
    target_phases = flatten_phases(target_spike_trains)
    ref_phases = flatten_phases(reference_spike_trains)

    # Compute histograms
    target_counts, bin_edges = np.histogram(target_phases, bins=bins, range=(0, 2*np.pi))
    ref_counts, _ = np.histogram(ref_phases, bins=bins, range=(0, 2*np.pi))
    differences = target_counts - ref_counts
    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()

    # Prepare data dictionary
    data = {
        'target_counts': target_counts.tolist(),
        'reference_counts': ref_counts.tolist(),
        'differences': differences.tolist(),
        'bin_centers': bin_centers,
        'bins': bins,
        'period': period,
        't_min': t_min_used,
        't_max': t_max
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=2)

'''
Load phase difference data from JSON and plot a polar histogram showing differences 
between target and reference spike phase distributions
'''
def plot_phase_difference_histogram_from_json(json_path, save_path=None, vmax=None, vstep=None):
    with open(json_path, 'r') as f:
        data = json.load(f)

    diffs = np.array(data['differences'])
    bin_centers = np.array(data['bin_centers'])
    t_min = data['t_min']
    t_max = data['t_max']
    n_bins = len(bin_centers)
    bar_heights = np.abs(diffs)

    # Set colors based on difference sign
    bar_colors = ['tab:blue' if d > 0 else 'tab:red' if d < 0 else 'gray' for d in diffs]

    # Compute percentages of positive/negative bins
    n_blue = np.sum(diffs > 0)
    n_red = np.sum(diffs < 0)
    n_total = n_blue + n_red
    blue_percent = 100 * n_blue / n_total if n_total > 0 else 0
    red_percent = 100 * n_red / n_total if n_total > 0 else 0

    # Automatic radial limits if not provided
    if vmax is None:
        vmax = np.max(bar_heights) + 1
    if vstep is None:
        vstep = max(1, int(vmax // 5))

    # Create polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Title with main info, time range, and bin ratio
    title_main = 'Phase Difference Histogram\nTarget - Reference'
    title_time = f'Time: [{t_min:.1f}, {t_max:.1f}] ms'
    title_ratio = f'Blue: {blue_percent:.1f}% | Red: {red_percent:.1f}%'
    ax.set_title(f'{title_main}\n{title_time}\n{title_ratio}', fontsize=12)

    # Draw bars
    ax.bar(
        bin_centers, bar_heights,
        width=2 * np.pi / n_bins,
        color=bar_colors,
        edgecolor='black',
        alpha=0.8
    )

    # Configure radial ticks
    r_ticks = np.arange(vstep, vmax + vstep, vstep)
    ax.set_rticks(r_ticks)
    ax.set_rlabel_position(22.5)
    ax.tick_params(labelsize=9)

    # Legend
    legend_elements = [
        Patch(facecolor='tab:blue', label='Target > Ref'),
        Patch(facecolor='tab:red', label='Target < Ref'),
        Patch(facecolor='gray', label='Equal')
    ]
    ax.legend(handles=legend_elements, loc='lower center',
              bbox_to_anchor=(0.5, -0.15), ncol=3)

    # Save or show plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

#==KDE=================================
'''
Compute a circular (wrapped) Kernel Density Estimate (KDE) for spike phases
'''
def compute_circular_kde(phases, num_points=500, bw_method='scott'):
    # Wrap phases to the [0, 2π] interval
    phases = np.asarray(phases) % (2 * np.pi)
    # Compute Gaussian KDE
    kde = gaussian_kde(phases, bw_method=bw_method)
    # Evaluation points on the circle
    angles = np.linspace(0, 2 * np.pi, num_points)
    # Evaluate KDE and normalize to max=1
    density = kde(angles)
    return angles, density / np.max(density)

'''
Plot circular (polar) KDE curves for multiple groups of spike phases.
'''
def plot_circular_kde_groups(phase_groups, group_names=None, group_colors=None, save_path=None, period=100):
    num_groups = len(phase_groups)
    if group_names is None:
        group_names = [f"Group {i+1}" for i in range(num_groups)]
    if group_colors is None:
        cmap = plt.get_cmap('tab10')
        group_colors = [cmap(i) for i in range(num_groups)]

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(15, 15))
    ax.set_theta_zero_location('N') # 0 at top
    ax.set_theta_direction(-1)      # clockwise

    # Plot KDE curves for each group
    for i, phases in enumerate(phase_groups):
        angles, density = compute_circular_kde(phases)
        ax.plot(angles, density, label=group_names[i], color=group_colors[i], linewidth=2)

    # Radial rings
    ax.set_rticks([0.5, 1.0])
    ax.set_rlabel_position(135)
    ax.tick_params(labelsize=18)
    ax.set_ylim(0, 1.05)

    # Angular labels with corresponding times
    tick_degrees = np.arange(0, 360, 22.5)
    for deg in tick_degrees:
        angle_rad = np.deg2rad(deg)
        time_ms = (deg / 360) * period
        label = f"{time_ms:.1f} ms"
        ax.text(
            angle_rad,
            1.3,
            label,
            ha='center',
            va='center',
            fontsize=18
        )

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=num_groups, fontsize=18)
    plt.subplots_adjust(bottom=0.2)

    # Save or show figure
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

'''
Helper method
'''
def plot_kde_from_spike_groups(
    spike_time_groups,  
    t_min, t_max, period,
    group_names=None, group_colors=None,
    save_path=None
):
    phase_groups = []
    for group in spike_time_groups:
        all_spikes = []
        for neuron_spikes in group:
            all_spikes.extend(neuron_spikes)  # flatten
        phases = get_spike_phases(all_spikes, t_min, t_max, period)
        phase_groups.append(phases)

    plot_circular_kde_groups(
        phase_groups,
        group_names=group_names,
        group_colors=group_colors,
        save_path=save_path
    )
#==KDE=================================


'''
Compute and plot a spike spectrogram for a list of spike trains.
'''
def plot_spike_spectrogram2(
    spike_trains,
    bin_size=1.0,        # Bin size in ms for converting spikes into a signal
    window_size=100.0,   # Window size in ms for the spectrogram
    overlap=50.0,        # Window overlap in ms
    fs=100000.0,         # Sampling rate in Hz for spike discretization
    save_path=None,      # Path to save the figure
    z_clip=3.0,          # Clipping for Z-score normalization
    max_freq=None,       # Maximum frequency to display (Hz)
    on_filter=True,      # Apply Gaussian smoothing if True
    sigma=(1,1),         # Smoothing kernel for Gaussian filter
    cmap_colors=None     # Custom colormap (list of colors)
):

    # Determine the total duration of the signal in samples
    max_time = max((max(train) if len(train) > 0 else 0) for train in spike_trains)
    duration = max_time + bin_size
    num_bins = int(np.ceil(duration * fs / 1000))  # convert ms → samples

    # Convert window and overlap from ms to samples
    nperseg = int(window_size * fs / 1000)
    noverlap = int(overlap * fs / 1000)
    nperseg = max(64, min(nperseg, num_bins))  # ensure minimal window size
    noverlap = min(noverlap, nperseg - 1)      # ensure overlap < window size

    summed_spectrograms = None

    # Loop over spike trains
    for train in spike_trains:
        # Create a binary spike vector
        spike_vector = np.zeros(num_bins, dtype=np.float32)
        indices = (np.array(train) * fs / 1000).astype(int)
        indices = indices[(indices >= 0) & (indices < num_bins)]
        spike_vector[indices] = 1.0

        # Compute PSD spectrogram
        f, t, Sxx = spectrogram(
            spike_vector,
            fs=fs,
            window='hann',
            nperseg=nperseg,
            noverlap=noverlap,
            scaling='density',
            mode='psd'
        )

        # Convert to dB
        Sxx_dB = 10 * np.log10(Sxx + 1e-12)

        # Sum spectrograms across neurons
        summed_spectrograms = Sxx_dB if summed_spectrograms is None else summed_spectrograms + Sxx_dB

    # Limit maximum frequency
    if max_freq is not None:
        freq_mask = f <= max_freq
        f = f[freq_mask]
        summed_spectrograms = summed_spectrograms[freq_mask, :]

    # Remove duplicate frequency bins
    f_unique, idx_f = np.unique(f, return_index=True)
    if len(f_unique) != len(f):
        print(f"Warning: Duplicate frequencies removed: {len(f) - len(f_unique)}")
        f = f_unique
        summed_spectrograms = summed_spectrograms[idx_f, :]

    # Remove duplicate time bins
    t_unique, idx_t = np.unique(t, return_index=True)
    if len(t_unique) != len(t):
        print(f"Warning: Duplicate time points removed: {len(t) - len(t_unique)}")
        t = t_unique
        summed_spectrograms = summed_spectrograms[:, idx_t]

    # Optional Gaussian smoothing
    smoothed = gaussian_filter(summed_spectrograms, sigma=sigma) if on_filter else summed_spectrograms

    # Z-score normalization and clipping
    mean = np.mean(smoothed)
    std = np.std(smoothed)
    z_spectrogram = np.clip((smoothed - mean) / (std + 1e-8), -z_clip, z_clip)

    # Convert time to ms for plotting
    t_ms = t * 1000
    max_t = t_ms[-1] if len(t_ms) > 0 else duration
    xticks_pos = np.linspace(0, max_t, 30)
    xticks_labels = [f"{x:.0f}" for x in xticks_pos]

    plt.figure(figsize=(12, 6))

    # Plot spectrogram with custom colormap if provided
    if cmap_colors is not None:
        cmap = LinearSegmentedColormap.from_list('custom_cmap', cmap_colors)
        mesh = plt.pcolormesh(t_ms, f, z_spectrogram, shading='auto', cmap=cmap, vmin=-z_clip, vmax=z_clip)
        plt.colorbar(mesh, ticks=np.linspace(-z_clip, z_clip, 5), label='Z-score')
    else:
        mesh = plt.pcolormesh(t_ms, f, z_spectrogram, shading='auto', cmap='seismic', vmin=-z_clip, vmax=z_clip)
        plt.colorbar(mesh, label='Z-score')

    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'Spike Spectrogram (fs={fs:.0f} Hz)')
    plt.xticks(xticks_pos, xticks_labels, rotation=45, fontsize=8)

    # Optional contours for peak detection
    if z_spectrogram.shape[0] >= 2 and z_spectrogram.shape[1] >= 2:
        plt.contour(t_ms, f, z_spectrogram, levels=[1.5], colors='black', linewidths=1.2, alpha=0.8)

    plt.ylim([0, f[-1] if len(f) > 0 else fs / 2])
    plt.tight_layout()

    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

#==WAVELET=============================
'''
Convert a list of spike times (in milliseconds) into a discrete impulse signal.
A crutch for wavelet analysis, as it works with clean signals
'''
def peaks_to_impulse_signal(peaks_list_ms, n_samples, dt_ms, base=0.1):

    n_samples = int(n_samples)
    sig = np.full(n_samples, base, dtype=float)  

    if peaks_list_ms is None:
        return sig  # Return baseline signal if no peaks

    for peaks in peaks_list_ms:
        if peaks is None or len(peaks) == 0:
            continue  # Skip empty lists
        arr = np.asarray(peaks, dtype=float) 
        idx = np.floor(arr / dt_ms).astype(int)  # Convert ms -> sample indices
        idx = idx[(idx >= 0) & (idx < n_samples)]  # Keep indices within bounds
        sig[idx] = 1.0  # Set impulses at spike positions

    return sig

'''
Convert spike peak times into impulse signals, normalize them, and compute wavelet coherence (WCT).
It helps to see the time correlation between the input and output patterns and understand 
how long it lasts before the pattern undergoes significant changes
'''
def globalize_and_wct_plot_from_peaks(
        A_peaks_ms,
        B_peaks_ms,
        dt_ms,
        freq_range=(0.1, 3.0),
        dj=1/12,
        base=0.1,
        save_path='wct.png',
        eps_noise=1e-14):

    # Step 1: Shift spike times so the earliest spike starts at 0 ms
    all_peaks = []
    for L in (A_peaks_ms, B_peaks_ms):
        if L is None:
            continue
        for p in L:
            if p is not None and len(p) > 0:
                all_peaks.append(np.min(p))

    if len(all_peaks) > 0:
        global_min = float(np.min(all_peaks))
        if global_min > 0:
            def shift(peaks_list):
                out = []
                for p in peaks_list:
                    if p is None or len(p) == 0:
                        out.append(np.array([]))
                    else:
                        # Shift spike times to start from 0
                        out.append(np.maximum(np.asarray(p, dtype=float) - global_min, 0.0))
                return out
            A_peaks_ms = shift(A_peaks_ms)
            B_peaks_ms = shift(B_peaks_ms)

    # Step 2: Determine the total duration and number of samples
    all_peaks_max = []
    for L in (A_peaks_ms, B_peaks_ms):
        if L is None:
            continue
        for p in L:
            if p is not None and len(p) > 0:
                all_peaks_max.append(np.max(p))

    if len(all_peaks_max) > 0:
        maxT_ms = float(np.max(all_peaks_max)) + 1.0  # +1 ms buffer
        n_samples = int(np.ceil(maxT_ms / dt_ms))
    else:
        n_samples = 1

    # Step 3: Convert spike times to discrete impulse signals
    sigA = peaks_to_impulse_signal(A_peaks_ms, n_samples, dt_ms, base)
    sigB = peaks_to_impulse_signal(B_peaks_ms, n_samples, dt_ms, base)

    # Step 4: Add tiny noise to avoid degenerate signals (all zeros)
    if eps_noise is not None and eps_noise > 0:
        sigA = sigA + eps_noise * np.random.randn(n_samples)
        sigB = sigB + eps_noise * np.random.randn(n_samples)

    # Step 5: Normalize signals to zero mean and unit variance
    eps = 1e-12
    stdA = sigA.std()
    stdB = sigB.std()
    if stdA < eps:
        stdA = eps
    if stdB < eps:
        stdB = eps

    sigA = (sigA - sigA.mean()) / stdA
    sigB = (sigB - sigB.mean()) / stdB

    # Step 6: Compute Wavelet Coherence (WCT)
    dt_sec = dt_ms / 1000.0  # convert dt to seconds
    WCT, _, _, freqs, _ = wavelet.wct(
        sigA,
        sigB,
        dt_sec,
        dj=dj,
        sig=False,
        wavelet="morlet",
        normalize=True
    )

    # Replace NaN/inf for safe plotting
    WCT = np.nan_to_num(WCT, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure frequencies are ascending
    if freqs[0] > freqs[-1]:
        freqs = freqs[::-1]
        WCT = WCT[::-1]

    # Step 7: Plot WCT
    t_ms = np.arange(n_samples) * dt_ms
    fmin, fmax = freq_range
    fmask = (freqs >= fmin) & (freqs <= fmax)

    plt.figure(figsize=(9, 4))
    plt.pcolormesh(
        t_ms,
        freqs[fmask],
        WCT[fmask],
        shading="auto",
        cmap="viridis"
    )
    plt.colorbar(label="Wavelet Coherence")
    plt.xlabel("Time (ms)")
    plt.ylabel("Hz")
    plt.title("Wavelet Coherence")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return sigA, sigB, t_ms
#==WAVELET=============================

#==PSI=================================
#!!!!!! P.S. It's still a question, and it might be unnecessary or misleading, or it might not be necessary at all.
'''
 Convert a list of spike trains (in ms) into binned firing rates (Hz).
'''
def bin_spikes_all(spikes_ms, dt_ms):
    # Step 1: Determine the total duration (max spike time across all neurons + 1 ms buffer)
    max_ms = max(np.max(s) for s in spikes_ms if len(s)) + 1.0

    # Step 2: Determine number of bins
    n_bins = int(np.ceil(max_ms / dt_ms))

    # Step 3: Define bin edges
    edges = np.linspace(0, n_bins*dt_ms, n_bins+1)

    # Step 4: Initialize array for binned firing rates
    rates = np.zeros((len(spikes_ms), n_bins))

    # Step 5: Loop over neurons and bin their spikes
    for i, sp in enumerate(spikes_ms):
        if len(sp):
            counts, _ = np.histogram(sp, bins=edges) # count spikes in each bin
            rates[i] = counts / (dt_ms / 1000.0) # convert to Hz (spikes/sec)

    return rates

'''
Compute static and time-resolved Phase Slope Index (PSI) between two sets of spike trains
'''
def psi_analysis_top(A_spikes_ms, B_spikes_ms, 
                     A_spikes_label="A", B_spikes_label="B", 
                     dt_ms=10.0,
                     fmin=0.05, fmax=5.0,
                     window_ms=10000.0, step_ms=2000.0,
                     top_n=5,
                     save_path='psi.png'):
    # Sampling frequency for PSI computation (Hz)
    sfreq = 1000.0 / dt_ms 

    # Combine all spike trains for binning
    spikes_all = list(A_spikes_ms) + list(B_spikes_ms)
    
    # Convert spikes into firing rates per bin
    rates = bin_spikes_all(spikes_all, dt_ms)
    
    # Indices of all pairs A -> B
    indices = seed_target_indices(np.arange(len(A_spikes_ms)),
                                  np.arange(len(A_spikes_ms), len(A_spikes_ms)+len(B_spikes_ms)))
    
    # ==== static PSI =====
    # Add fake "epoch" dimension for compatibility
    data_epochs = rates[np.newaxis, :, :]
    conn = phase_slope_index(data_epochs, 
                             indices=indices, 
                             sfreq=sfreq,
                             fmin=fmin, 
                             fmax=fmax, 
                             mode="fourier")
    
    dense = conn.get_data(output="dense")
    if dense.ndim == 3:
        dense = dense.mean(axis=-1) # average over epochs if present
    
    # Extract A->B connections
    psi_static = dense[np.ix_(np.arange(len(A_spikes_ms)),
                              np.arange(len(A_spikes_ms), len(A_spikes_ms)+len(B_spikes_ms)))]

    # ==== Time-Resolved PSI =====
    win = int(window_ms / dt_ms) # samples per window
    step = int(step_ms / dt_ms)  # step in samples
    starts = np.arange(0, rates.shape[1]-win, step) # window start indices
    times = (starts + win/2) * dt_ms # center of window in ms
    psi_time = []

    # Sliding window PSI calculation
    for s in starts:
        seg = rates[:, s:s+win][np.newaxis, :, :]
        conn_seg = phase_slope_index(seg, 
                                     indices=indices, 
                                     sfreq=sfreq,
                                     fmin=fmin, 
                                     fmax=fmax, 
                                     mode="fourier")
        d = conn_seg.get_data(output="dense")
        if d.ndim == 3:
            d = d.mean(axis=-1)
        psi_time.append(d[np.ix_(np.arange(len(A_spikes_ms)),
                                 np.arange(len(A_spikes_ms), len(A_spikes_ms)+len(B_spikes_ms)))].ravel())
    psi_time = np.nan_to_num(np.array(psi_time).T)  # pairs × time

    # ==== Select top-n pairs =====
    mean_abs_psi = np.abs(psi_time).mean(axis=1) # mean over time
    top_idx = np.argsort(-mean_abs_psi)[:top_n] # top-N by magnitude
    labels_top = [f"A{i}→B{j}" for i in range(len(A_spikes_ms)) for j in range(len(B_spikes_ms))]
    labels_top = [labels_top[i] for i in top_idx]

    # ==== Ploting ====
    # Static PSI
    plt.figure(figsize=(6,5))
    plt.imshow(psi_static, cmap="RdBu_r", origin="lower")
    plt.title(f"Static PSI (all {A_spikes_label} → all {B_spikes_label})")
    plt.xlabel(B_spikes_label)
    plt.ylabel(A_spikes_label)
    plt.colorbar(label="PSI")
    plt.tight_layout()
    plt.savefig(f'{save_path}_Static_PSI.png', dpi=300)

    # Time-resolved PSI (top-N pairs)
    plt.figure(figsize=(12,5))
    plt.imshow(psi_time[top_idx], aspect="auto", origin="lower",
               cmap="RdBu_r", extent=[times[0], times[-1], 0.5, top_n+0.5])
    plt.yticks(np.arange(1, top_n+1), labels_top)
    plt.xlabel("time (ms)")
    plt.ylabel("pair channels")
    plt.title(f"Time-resolved PSI. Top {top_n} pairs by magnitude")
    plt.colorbar(label="PSI")
    plt.tight_layout()
    plt.savefig(f'{save_path}_Time_resolved_PSI.png', dpi=300)

    return psi_static, psi_time, times, top_idx
#==PSI=================================

'''
Simplified and fast ISI (inter-spike interval) analysis. 
    - spike_times_dict: dictionary {axon: [spike times]}
'''
def plot_ISI_analysis(plots_dir_name, spike_times_dict, total_time, title_label):
    """
    
    """

    all_ISI = []
    all_times = []

    # Create figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(18, 5))

    # 1. ISI vs Time
    for axon, spikes in spike_times_dict.items():
        if len(spikes) > 1:
            ISI = np.diff(spikes)
            times = spikes[1:]

            # Determine axon color
            color = "green" if axon < 65 else "red" if 65 <= axon < 83 else "blue"

            # Add points to the plot
            axs[0].scatter(times, ISI, color=color, s=5, alpha=0.7)
            
            all_ISI.extend(ISI)
            all_times.extend(times)

    if not all_ISI:
        print(f"No data available for {title_label}")
        return

    # Axis settings
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("ISI (ms)")
    axs[0].set_title(f"ISI vs Time - {title_label}")
    axs[0].set_xlim(0, total_time)
    axs[0].set_ylim(0, 100)
    axs[0].grid(True)

    # 2. ISI histogram
    axs[1].hist(all_ISI, bins=np.arange(0, 100, 2), color="black", alpha=0.7)
    axs[1].set_xlabel("ISI (ms)")
    axs[1].set_ylabel("Count")
    axs[1].set_title(f"ISI Histogram - {title_label}")

    plt.tight_layout()
    plt.savefig(f"{plots_dir_name}/ISI_analysis_{title_label}.png", dpi=300)
    print(f"ISI analysis plot saved: ISI_analysis_{title_label}.png")
#-------- Other analyses and graphs --------------------------------------

#-------- Rasters --------------------------------------------------------
'''
Plot a raster from a dictionary
'''
def plot_raster_dict2(
    spike_dict,
    title="Raster plot",
    size=(10, 12),
    dot_size=0.4,
    filename='raster.png',
    color=None,
    y_label='Neuron',
    label_fontsize=14,
    title_fontsize=16,
    yticks_fontsize=14,
    xlabel_fontsize=14,
    xticks_fontsize=14,
    step=500,            # Step of periods marked by black vertical lines (for better understanding of where each pattern phase is)
    y_prefix=None,       # Channel prefix in the dictionary
    spike_linewidth=2,   # Width of spike ticks
    y_color_ranges=None  # list of (y_min, y_max, color) or (y_min, y_max, color, alpha)
):
    plt.figure(figsize=size)

    keys_numeric = False
    try:
        keys_numeric = [float(k) for k in spike_dict.keys()]
        items = sorted(spike_dict.items(), key=lambda kv: float(kv[0]))
    except Exception:
        items = list(spike_dict.items())

    # safe maximum time
    max_t = max((max(times) for _, times in items if len(times) > 0), default=0)
    end_t = max_t if max_t > 0 else 2000.0

    # time vertical lines
    step = 500.0
    x = 0.0
    while x <= end_t:
        plt.axvline(x, color='black', linewidth=8.0, alpha=1.0, zorder=0)
        x += step

    keys_numeric = False
    try:
        keys_numeric = [float(k) for k in spike_dict.keys()]
        items = sorted(spike_dict.items(), key=lambda kv: float(kv[0]))
    except Exception:
        items = list(spike_dict.items())

    for i, (name, times) in enumerate(items):
        y_bottom = i - dot_size/2
        y_top = i + dot_size/2

        #select color for this trace: check if it falls into y_color_ranges, otherwise use the main color
        trace_color = None
        if y_color_ranges is not None:
            for rng in y_color_ranges:
                rstart, rend = rng[0], rng[1]
                rc = rng[2] if len(rng) >= 3 else None
                if keys_numeric:
                    try:
                        val = float(name)
                    except Exception:
                        val = None
                    if val is not None and rstart <= val <= rend:
                        trace_color = rc
                        break
                else:
                    idx1 = i + 1
                    if int(rstart) <= idx1 <= int(rend):
                        trace_color = rc
                        break

        if trace_color is None:
            trace_color = color if color is not None else 'C0'

        plt.vlines(times, y_bottom, y_top, color=trace_color, linewidth=spike_linewidth, zorder=2)

    plt.xlabel("Time (ms)", fontsize=xlabel_fontsize)

    n_items = len(items)
    step = max(1, n_items // 8)
    positions = list(range(0, n_items, step))
    if positions and positions[-1] != n_items - 1:
        positions.append(n_items - 1)

    if n_items > 20:
        ylabel_to_use = y_prefix if y_prefix is not None else y_label
        plt.ylabel(ylabel_to_use, fontsize=label_fontsize)
        plt.yticks(positions, [str(p + 1) for p in positions], fontsize=yticks_fontsize)
    else:
        plt.ylabel(y_label, fontsize=label_fontsize)
        plt.yticks(positions, [str(items[p][0]) for p in positions], fontsize=yticks_fontsize)

    plt.xticks(fontsize=xticks_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


'''
Load the pattern from the checkpoint of the ephaptic communication model
'''
def load_spike_data_from_checkpoint(checkpoint_file_name):
    with open(checkpoint_file_name, "rb") as f:
        checkpoint = pickle.load(f)

    total_time = checkpoint["t"] * checkpoint["dt"]

    spike_times_start = {}
    spike_times_middle = {}
    spike_times_end = {}

    N = checkpoint["N"]

    for i in range(N):
        spike_times_start[i + 1] = checkpoint["spike_times_start"][i]
        spike_times_middle[i + 1] = checkpoint["spike_times_middle"][i]
        spike_times_end[i + 1] = checkpoint["spike_times_end"][i]

    num_regular = checkpoint["num_regular"]
    num_irregular = checkpoint["num_irregular"]
    num_semiregular = checkpoint["num_semiregular"]
    
    regular_range = (1, num_regular)
    irregular_range = (num_regular + 1, num_regular + num_irregular)
    semiregular_range = (num_regular + num_irregular + 1, num_regular + num_irregular + num_semiregular)

    return (
        total_time, 
        regular_range, irregular_range, semiregular_range, 
        spike_times_start, spike_times_middle, spike_times_end
    )

'''
Save data for raster plot for further rendering
'''
def save_raster_data_to_json(sim, pop_groups, time_range=(0, 5000), 
                             json_filename='raster_data.json', all_pop_rates=None):
    # GID -> population
    gid2pop = {c['gid']: c['tags']['pop'] for c in sim.net.allCells}

    # Ordered GIDs by groups
    ordered_gids = []
    for group in pop_groups:
        gids = [gid for gid, pop in gid2pop.items() if pop.startswith(group)]
        ordered_gids.extend(sorted(gids))

    # For each GID — collect spike times filtered by the time range
    gid2spikes = defaultdict(list)
    for t, gid in zip(sim.simData['spkt'], sim.simData['spkid']):
        if time_range[0] <= t <= time_range[1]:
            gid2spikes[gid].append(t)

    # Group neurons by populations
    pop2neurons = defaultdict(list)
    for gid in ordered_gids:
        pop = gid2pop[gid]
        pop2neurons[pop].append(gid)

    # Assemble data for JSON
    data = {}
    for pop, gids in pop2neurons.items():
        data[pop] = {
            'gids': gids,
            'spikes': {gid: gid2spikes[gid] for gid in gids}
        }

    # Add popRates to JSON if provided
    data['popRates'] = all_pop_rates or {}

    with open(json_filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Raster data saved to {json_filename}")

'''
Split a raster dictionary into multiple sub-dictionaries based on numeric ranges in keys
'''
def split_raster_dict(raster,
                      prefix='ECell',
                      bins=((1, 20), (21, 60), (61, 100))):
    # Compile regex pattern to extract numeric part from keys
    pattern = re.compile(rf'^{re.escape(prefix)}(\d+)$')

    dicts = [dict() for _ in bins]
    others = {}  # Keys that don't match prefix or fall outside bins

    for k, v in raster.items():
        m = pattern.match(k)
        if not m:
            # Key does not match prefix pattern → put in others
            others[k] = v
            continue

        # Extract numeric part of the key
        num = int(m.group(1))
        placed = False

        # Place key-value pair into the corresponding bin dictionary
        for i, (a, b) in enumerate(bins):
            if a <= num <= b:
                dicts[i][k] = v
                placed = True
                break

        # If not placed in any bin, put in others
        if not placed:
            others[k] = v

    # Return tuple of bin dicts + others dict
    return tuple(dicts) + (others,)
#-------- Rasters --------------------------------------------------------

#-------- Utilities ------------------------------------------------------
def get_spike_times(sim, target, time_range=(0, 5000)):
    spkt = sim.simData['spkt']
    spkid = sim.simData['spkid']

    tmin, tmax = time_range
    times = {}

    for t, gid in zip(spkt, spkid):
        gid = int(gid)  
        if not (tmin <= t <= tmax):
            continue

        pop = sim.net.allCells[gid]['tags']['pop']

        if isinstance(target, str) and pop.startswith(target):
            times.setdefault(gid, []).append(t)
        elif isinstance(target, int) and gid == target:
            times.setdefault(gid, []).append(t)
    return times

#t_min, t_max in ms
def compute_population_spike_rate(raster_dict, t_min, t_max):
    duration_ms = t_max - t_min
    n_channels = len(raster_dict)

    total_spikes = sum(
        sum(t_min <= t <= t_max for t in spikes)
        for spikes in raster_dict.values()
    )

    return total_spikes / (n_channels * duration_ms)

def filterDictBySubstring(source, substring):
    filtered_dict = {key: value for key, value in source.items() if substring in key}
    return filtered_dict

def align_by_first_spike(channels, t_phase=0.0):
    result = []

    for v in channels:
        times = np.asarray(v, dtype=float)
        if times.size == 0:
            aligned = times.copy()
        else:
            shift = float(t_phase) - float(times[0])
            aligned = times + shift
        result.append(aligned)

    return result

def transformDictToMatrix(source):
    matrix = []
    for key, value in source.items():
        matrix.append(value)
    return matrix

def transformMatrixToDict(matrix):
    result_dict = {}
    for i, row in enumerate(matrix):
        result_dict[i] = row
    return result_dict

def filterAndRenumberDict(source, substring):
    filtered_dict = {key: value for key, value in source.items() if substring in key}
    renumbered_dict = {i + 1: value for i, (key, value) in enumerate(filtered_dict.items())}

    return renumbered_dict

#============================================================
def load_pop_spikes_from_json(json_filename):
    with open(json_filename, 'r') as f:
        data = json.load(f)

    pop_spikes = defaultdict(list)

    for pop, pop_data in data.items():
        if pop == 'popRates':
            continue  
        spikes_dict = pop_data.get('spikes', {})
        for gid, spikes in spikes_dict.items():
            pop_spikes[pop].extend(spikes)

    for pop in pop_spikes:
        pop_spikes[pop].sort()

    return dict(pop_spikes)

def load_pop_rates_from_json(json_filename):
    with open(json_filename, 'r') as f:
        data = json.load(f)
    
    # popRates may be missing, in which case we will return an empty dictionary
    pop_rates = data.get('popRates', {})
    
    return pop_rates
#-------- Utilities ------------------------------------------------------