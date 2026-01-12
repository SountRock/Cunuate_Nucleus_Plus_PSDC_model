"""
Analysis of the asynchrony of the input pattern.
Considers desynchrony both within the responses of a single axon and the "start" desynchrony
of responses between all axons.
"""

import json
import numpy as np
from collections import Counter

def Sp_Dev_Scores(spikes_1kHz_async, phases_ms):
    spikes = [np.sort(np.array(ch, dtype=float)) for ch in spikes_1kHz_async]

    scores = []
    for start_ms, end_ms in phases_ms:
        phase_len_ms = float(end_ms) - float(start_ms)
        if phase_len_ms <= 0:
            scores.append(float('nan')); continue

        # Collect spikes within the phase for each channel
        phase_spikes = [ch[(ch >= start_ms) & (ch < end_ms)] for ch in spikes]

        # Step 1: count spikes per period and compute ISI
        # as if those spikes were synchronous at the same frequency
        counts = [arr.size for arr in phase_spikes]
        mode = Counter(counts).most_common(1)[0][0]
        if mode == 0:
            nonz = [c for c in counts if c > 0]
            n_sync = int(round(np.median(nonz))) if nonz else 0
        else:
            n_sync = int(mode)

        if n_sync <= 1:
            scores.append(float('nan')); continue

        isi_s = (phase_len_ms / 1000.0) / n_sync # ISI in seconds
        freq_hz = 1.0 / isi_s

        per_channel_rmse = []
        first_times_ms = []

        for arr in phase_spikes:
            if arr.size == 0:
                continue

            # Step 2: align the synchronous signal by the first spike for each channel separately
            K = min(arr.size, n_sync)
            first_t_ms = float(arr[0])
            first_times_ms.append(first_t_ms)
            expected_ms = first_t_ms + np.arange(K) * (isi_s * 1000.0)
            observed_ms = arr[:K].astype(float)

            # Step 3: compute deviations of spikes from the synchronous (by index in sequence)
            # and obtain channel RMSE via root mean square deviation
            dev_s = (observed_ms - expected_ms) / 1000.0 # deviations in seconds
            rmse_s = float(np.sqrt(np.mean(dev_s ** 2))) if dev_s.size > 0 else 0.0
            per_channel_rmse.append(rmse_s)

        if len(per_channel_rmse) == 0 or len(first_times_ms) == 0:
            scores.append(float('nan')); continue

        # Step 4: take the mean (root mean square deviation across all channels)
        mean_rmse_s = float(np.mean(per_channel_rmse))

        # Step 5: compute max deviation of the first spike between channels (in seconds)
        max_first_dev_s = (max(first_times_ms) - min(first_times_ms)) / 1000.0

        # Step 6: mean_rmse_s / freq_synch + max_first_dev_s / freq_synch
        sp_dev = ((mean_rmse_s / freq_hz) + (max_first_dev_s / freq_hz)) * freq_hz
        scores.append(float(sp_dev))

    return scores

# Load pattern for preprocessing
with open('sequence_spike_times_final_pattern.json','rb') as f:
    spikes_1kHz_async = json.load(f)
phases_ms = [(0,500), (500,1000), (1000,1500), (1500, 2000), (2000, 2500)] # Periods in which asynchrony is measured
scores = Sp_Dev_Scores(spikes_1kHz_async, phases_ms)
print(scores)
