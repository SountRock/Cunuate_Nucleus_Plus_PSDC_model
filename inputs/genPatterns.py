'''
Generates multi-channel synchronous and asynchronous spike patterns (in ms)
and saves rasters to a folder. Includes a function for sequential generation
of a set of patterns (concatenation over time).
All times are in milliseconds (ms). Frequency is given in Hz, interval = 1000 / freq_hz (ms).
'''

from typing import List, Tuple, Union, Optional, Dict, Any
import os
import json
import numpy as np
import matplotlib.pyplot as plt

'''
Generates a regular (synchronous) sequence in milliseconds
'''
def gen_regular_spikes(freq_hz: float, duration_ms: float, phase_ms: float = 0.0) -> np.ndarray:
    if freq_hz <= 0:
        return np.array([], dtype=float)
    isi_ms = 1000.0 / freq_hz
    times = np.arange(phase_ms, duration_ms + 1e-9, isi_ms)
    times = times[times <= duration_ms]
    return times

'''
For segments of length T_dev_ms choose up to n_dev spikes and shift them randomly
in the range [-maxISIdev_ms, +maxISIdev_ms] (ms).
Returns corrected times, sorted and clipped to [0, duration_ms] (ms).
'''
def apply_segmented_deviations(base_times_ms: np.ndarray,
                               duration_ms: float,
                               maxISIdev_ms: float,
                               n_dev: int,
                               T_dev_ms: float,
                               seed: int = None,
                               min_gap_ms: float = 1.0) -> np.ndarray:
 
    rng = np.random.default_rng(seed)
    times = base_times_ms.copy()
    if len(times) == 0 or T_dev_ms <= 0 or (n_dev == 0) or maxISIdev_ms == 0:
        return times.copy()

    seg_start = 0.0
    while seg_start < duration_ms:
        seg_end = min(seg_start + T_dev_ms, duration_ms)
        idxs = np.where((times >= seg_start) & (times < seg_end))[0]
        if len(idxs) > 0:
            k = min(n_dev, len(idxs))
            chosen = rng.choice(idxs, size=k, replace=False)
            shifts = rng.uniform(-maxISIdev_ms, maxISIdev_ms, size=k)
            times[chosen] = times[chosen] + shifts
        seg_start += T_dev_ms

    times = np.clip(times, 0.0, duration_ms)
    times = np.sort(times)

    if min_gap_ms > 0 and len(times) > 1:
        corrected = [times[0]]
        for t in times[1:]:
            prev = corrected[-1]
            if t - prev < min_gap_ms:
                t = prev + min_gap_ms
                if t > duration_ms:
                    continue
            corrected.append(t)
        times = np.array(corrected)

    return times

'''
Saves a single multi-channel raster.
times_ms can be an np.ndarray (single channel) or a list of arrays (one array per channel).
'''
def plot_and_save_raster_ms(times_ms: Union[np.ndarray, List[np.ndarray]],
                            out_path: str,
                            title: str = "",
                            figsize: Tuple[float, float] = (10, 2),
                            dpi: int = 300):
    # Normalize input into a list
    if isinstance(times_ms, np.ndarray):
        series = [times_ms]
    else:
        series = times_ms

    n_channels = len(series)
    fig, ax = plt.subplots(figsize=figsize)
    if n_channels > 0:
        # eventplot accepts a list of lists/arrays
        # lineoffsets — Y positions (set from n-1..0 so channel 0 is at the top)
        lineoffsets = np.arange(n_channels)[::-1]
        ax.eventplot(series, lineoffsets=lineoffsets, linelengths=0.8)
        ax.set_yticks(lineoffsets)
        # channel labels 1..n top to bottom
        ax.set_yticklabels([str(i+1) for i in range(n_channels)][::-1])
    else:
        ax.set_yticks([])

    ax.set_xlabel("Time (ms)")
    ax.set_title(title)
    # overall X boundary
    max_t = 0.0
    for s in series:
        if len(s):
            max_t = max(max_t, float(np.max(s)))
    max_t = max(max_t, 1.0)
    x_range = max_t
    pad = x_range * 0.02
    ax.set_xlim(0 - pad, max_t + pad)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0.12)
    plt.close(fig)

'''
Saves a combined raster (ms): top = sync (multi-channel), bottom = async (multi-channel),
sharing common X-axis limits.
'''
def plot_and_save_combined_ms(sync_list: List[np.ndarray],
                              async_list: List[np.ndarray],
                              out_path: str,
                              titles: Tuple[str, str] = ("sync", "async"),
                              figsize: Tuple[float, float] = (10, 4),
                              dpi: int = 300):
    # determine the common maximum time
    max_t = 0.0
    for s in sync_list + async_list:
        if len(s):
            max_t = max(max_t, float(np.max(s)))
    max_t = max(max_t, 1.0)
    x_range = max_t
    pad = x_range * 0.02

    n_sync = len(sync_list)
    n_async = len(async_list)
    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
    # top: sync
    if n_sync > 0:
        offsets_sync = np.arange(n_sync)[::-1]
        axs[0].eventplot(sync_list, lineoffsets=offsets_sync, linelengths=0.8)
        axs[0].set_yticks(offsets_sync)
        axs[0].set_yticklabels([str(i+1) for i in range(n_sync)][::-1])
    else:
        axs[0].set_yticks([])
    axs[0].set_ylabel(titles[0])
    axs[0].grid(axis='x', linestyle='--', alpha=0.4)

    # bottom: async
    if n_async > 0:
        offsets_async = np.arange(n_async)[::-1]
        axs[1].eventplot(async_list, lineoffsets=offsets_async, linelengths=0.8)
        axs[1].set_yticks(offsets_async)
        axs[1].set_yticklabels([str(i+1) for i in range(n_async)][::-1])
    else:
        axs[1].set_yticks([])
    axs[1].set_ylabel(titles[1])
    axs[1].grid(axis='x', linestyle='--', alpha=0.4)

    axs[0].set_title("Synchronous (top) / Asynchronous (bottom)")
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_xlim(0 - pad, max_t + pad)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0.12)
    plt.close(fig)

'''
Generates multi-channel synchronous and asynchronous patterns and saves rasters:
    - sync.png  (all channels)
    - async.png (all channels)
    - combined.png (sync top, async bottom)
For async, a different seed per channel is used: seed + channel_index, so asynchrony differs between channels.
Also saves spike time matrices to JSON (lists of lists).
'''
def make_and_save_patterns_ms(freq_hz: float,
                              duration_ms: float,
                              maxISIdev_ms: float,
                              n_dev: int,
                              T_dev_ms: float,
                              n_channels: int = 10,
                              out_dir: str = "rasters",
                              seed: int = 0):
    
    os.makedirs(out_dir, exist_ok=True)

    # base synchronous sequence (same for all channels)
    base_sync = gen_regular_spikes(freq_hz=freq_hz, duration_ms=duration_ms)

    # list of arrays for sync (duplicate base sequence across channels)
    sync_list = [base_sync.copy() for _ in range(n_channels)]

    # build async_list: each channel gets its own deviated version
    async_list: List[np.ndarray] = []
    for ch in range(n_channels):
        ch_seed = None if seed is None else int(seed) + ch + 1
        async_times = apply_segmented_deviations(base_sync,
                                                 duration_ms=duration_ms,
                                                 maxISIdev_ms=maxISIdev_ms,
                                                 n_dev=n_dev,
                                                 T_dev_ms=T_dev_ms,
                                                 seed=ch_seed,
                                                 min_gap_ms=1.0)
        async_list.append(async_times)

    # Save rasters (local for this pattern)
    plot_and_save_raster_ms(sync_list, os.path.join(out_dir, "sync.png"),
                            title=f"Synchronous {freq_hz} Hz, dur={duration_ms} ms, channels={n_channels}",
                            figsize=(10, max(2.0, 0.3 * n_channels)))
    plot_and_save_raster_ms(async_list, os.path.join(out_dir, "async.png"),
                            title=f"Asynchronous (dev={maxISIdev_ms} ms, n={n_dev}, T_dev={T_dev_ms} ms), channels={n_channels}",
                            figsize=(10, max(2.0, 0.3 * n_channels)))
    plot_and_save_combined_ms(sync_list, async_list, os.path.join(out_dir, "combined.png"),
                              figsize=(10, max(4.0, 0.3 * max(n_channels, n_channels))))

    # --- Convert numpy arrays to native Python lists and save JSON ---
    sync_serializable = [arr.tolist() for arr in sync_list]
    async_serializable = [arr.tolist() for arr in async_list]

    sync_json_path = os.path.join(out_dir, "spike_times_sync.json")
    async_json_path = os.path.join(out_dir, "spike_times_async.json")
    with open(sync_json_path, 'w', encoding='utf-8') as f:
        json.dump(sync_serializable, f, ensure_ascii=False, indent=2)
    with open(async_json_path, 'w', encoding='utf-8') as f:
        json.dump(async_serializable, f, ensure_ascii=False, indent=2)

    print(f"Saved JSON: {sync_json_path}")
    print(f"Saved JSON: {async_json_path}")

    return sync_list, async_list

def _get_cfg_val(cfg: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in cfg:
            return cfg[k]
    # try upper/lower case variants
    for k in keys:
        ku = k.upper()
        kl = k.lower()
        if ku in cfg:
            return cfg[ku]
        if kl in cfg:
            return cfg[kl]
    return default

'''
Choose a small jitter amount to desynchronize sync between channels.
Returns a value in ms. We pick it so as not to break min_gap_ms=1.0:
limit jitter <= 0.5 ms and <= maxISIdev_ms/2 (if maxISIdev_ms is provided).
'''
def _jitter_amount_for_desync(freq_hz: float, maxISIdev_ms: float) -> float:
    if maxISIdev_ms is None or maxISIdev_ms <= 0:
        base = 0.2
    else:
        base = maxISIdev_ms / 2.0
    return min(0.5, max(0.05, base))

'''
Loads JSON (sync/async 2D arrays) and draws raster images.
'''
def plot_sequence_from_saved_json(json_path: str, out_dir: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sync = [np.array(ch, dtype=float) for ch in data.get("sync", [])]
    async_ = [np.array(ch, dtype=float) for ch in data.get("async", [])]

    plot_and_save_raster_ms(sync, os.path.join(out_dir, "sequence_sync_from_json.png"),
                            title="Sequence sync (from JSON)", figsize=(20, max(2.0, 0.25 * len(sync))))
    plot_and_save_raster_ms(async_, os.path.join(out_dir, "sequence_async_from_json.png"),
                            title="Sequence async (from JSON)", figsize=(20, max(2.0, 0.25 * len(async_))))
    plot_and_save_combined_ms(sync, async_, os.path.join(out_dir, "sequence_combined_from_json.png"),
                              figsize=(50, max(4.0, 0.25 * max(len(sync), len(async_)))))

'''
Generates a sequence of patterns specified by an array of config dicts.
Each dict may contain keys (variants):
  freq_hz / FREQ
  duration_ms / DURATION_MS
  maxISIdev_ms / MAX_ISI_DEV_MS
  n_dev / N_DEV
  T_dev_ms / T_DEV_MS
  n_channels / N_CHANNELS
  seed / SEED
'''
def make_and_save_sequence_from_configs(configs: List[Dict[str, Any]],
                                       out_dir: str = "rasters_sequence",
                                       gap_ms: float = 10.0):
    os.makedirs(out_dir, exist_ok=True)

    # Compute the maximum number of channels across configs (for alignment)
    max_n_channels = 0
    parsed_cfgs = []
    for cfg in configs:
        freq_hz = _get_cfg_val(cfg, ["freq_hz", "FREQ"], 1000.0)
        duration_ms = _get_cfg_val(cfg, ["duration_ms", "DURATION_MS"], 100.0)
        maxISIdev_ms = _get_cfg_val(cfg, ["maxISIdev_ms", "MAX_ISI_DEV_MS"], (1.0/freq_hz)*1000.0*0.8)
        n_dev = int(_get_cfg_val(cfg, ["n_dev", "N_DEV"], 0))
        T_dev_ms = _get_cfg_val(cfg, ["T_dev_ms", "T_DEV_MS", "T_dev_ms"], 25.0)
        n_channels = int(_get_cfg_val(cfg, ["n_channels", "N_CHANNELS"], 10))
        seed = _get_cfg_val(cfg, ["seed", "SEED"], 0)
        parsed = {
            "freq_hz": float(freq_hz),
            "duration_ms": float(duration_ms),
            "maxISIdev_ms": float(maxISIdev_ms),
            "n_dev": int(n_dev),
            "T_dev_ms": float(T_dev_ms),
            "n_channels": int(n_channels),
            "seed": None if seed is None else int(seed)
        }
        parsed_cfgs.append(parsed)
        if parsed["n_channels"] > max_n_channels:
            max_n_channels = parsed["n_channels"]

    # Initialize lists for the whole sequence: per channel
    sync_all_channels: List[List[float]] = [[] for _ in range(max_n_channels)]
    async_all_channels: List[List[float]] = [[] for _ in range(max_n_channels)]

    current_offset = 0.0

    for idx, p in enumerate(parsed_cfgs):
        freq_hz = p["freq_hz"]
        duration_ms = p["duration_ms"]
        maxISIdev_ms = p["maxISIdev_ms"]
        n_dev = p["n_dev"]
        T_dev_ms = p["T_dev_ms"]
        n_channels = p["n_channels"]
        seed = p["seed"]

        # generate base synchronous sequence for the pattern
        base_sync = gen_regular_spikes(freq_hz=freq_hz, duration_ms=duration_ms)

        # Desynchronize sync between channels (small per-channel constant jitter)
        # to avoid identical times between channels at boundaries.
        jitter_amt = _jitter_amount_for_desync(freq_hz, maxISIdev_ms)
        # create RNG for reproducibility; if seed is None RNG will be non-reproducible
        base_rng = np.random.default_rng(seed)
        # For each channel generate a small constant shift in [-jitter_amt, +jitter_amt]
        channel_shifts = [float(base_rng.uniform(-jitter_amt, jitter_amt)) for _ in range(n_channels)]

        # sync_list for the current pattern (per channel) — apply per-channel shift
        sync_list = []
        for ch in range(n_channels):
            shifted = base_sync.copy() + channel_shifts[ch]
            # Clip inside pattern [0, duration_ms]
            shifted = np.clip(shifted, 0.0, duration_ms)
            # sort and small correction of duplicates (in case of equality)
            shifted.sort()
            for i in range(1, shifted.size):
                if shifted[i] <= shifted[i-1] + 1e-9:
                    shifted[i] = shifted[i-1] + 1e-6
            sync_list.append(shifted)

        # async_list for the current pattern
        async_list = []
        for ch in range(n_channels):
            ch_seed = None if seed is None else int(seed) + ch + 1
            async_times = apply_segmented_deviations(base_sync,
                                                     duration_ms=duration_ms,
                                                     maxISIdev_ms=maxISIdev_ms,
                                                     n_dev=n_dev,
                                                     T_dev_ms=T_dev_ms,
                                                     seed=ch_seed,
                                                     min_gap_ms=1.0)
            # Note: async_times are generated from base_sync (without per-channel shift),
            # but later we will shift the whole pattern by current_offset.
            async_list.append(async_times)

        # shift the times of the current pattern by current_offset and add to the global lists
        for ch in range(max_n_channels):
            if ch < n_channels:
                # add shifted times (if any) to the corresponding channel
                s_times = (sync_list[ch] + current_offset) if len(sync_list[ch]) else np.array([], dtype=float)
                a_times = (async_list[ch] + current_offset) if len(async_list[ch]) else np.array([], dtype=float)
                sync_all_channels[ch].extend([float(x) for x in s_times.tolist()])
                async_all_channels[ch].extend([float(x) for x in a_times.tolist()])
            else:
                pass

        # Increase offset for the next pattern (add gap_ms)
        current_offset += duration_ms + gap_ms

    # Convert lists to numpy arrays and sort
    sync_arrays = []
    async_arrays = []
    for ch in range(max_n_channels):
        arr_s = np.array(sync_all_channels[ch], dtype=float)
        if arr_s.size:
            arr_s.sort()
        sync_arrays.append(arr_s)
        arr_a = np.array(async_all_channels[ch], dtype=float)
        if arr_a.size:
            arr_a.sort()
        async_arrays.append(arr_a)

    # --- Save JSON ONLY with merged per-channel arrays (sync & async) ---
    json_obj = {
        "sync": [arr.tolist() for arr in sync_arrays],
        "async": [arr[:-1].tolist() for arr in async_arrays]
    }
    json_path = os.path.join(out_dir, "sequence_spike_times.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, ensure_ascii=False, indent=2)

    print(f"Saved sequence JSON (merged per-channel arrays only): {json_path}")

    # 1) Now draw rasters **from the saved JSON**
    plot_sequence_from_saved_json(json_path, out_dir)

    return sync_arrays, async_arrays, json_obj

if __name__ == "__main__":
    # Sample config
    configs = [
        {
            "FREQ": 50.0,
            "DURATION_MS": 500.0,
            "MAX_ISI_DEV_MS": (1.0/50.0 * 1000.0) * 0.01,
            "N_DEV": 0,
            "T_DEV_MS": 10.0,
            "N_CHANNELS": 100,
            "SEED": 7
        },
        {
            "FREQ": 100.0,
            "DURATION_MS": 500.0,
            "MAX_ISI_DEV_MS": (1.0/100.0 * 1000.0) * 0.1,
            "N_DEV": 2,
            "T_DEV_MS": 10.0,
            "N_CHANNELS": 100,
            "SEED": 42
        },
        {
            "FREQ": 50.0,
            "DURATION_MS": 500.0,
            "MAX_ISI_DEV_MS": (1.0/50.0 * 1000.0) * 0.5,
            "N_DEV": 5,
            "T_DEV_MS": 10.0,
            "N_CHANNELS": 100,
            "SEED": 7
        },
        {
            "FREQ": 30.0,
            "DURATION_MS": 500.0,
            "MAX_ISI_DEV_MS": (1.0/30.0 * 1000.0) * 0.8,
            "N_DEV": 10,
            "T_DEV_MS": 10.0,
            "N_CHANNELS": 100,
            "SEED": 7
        },
        {
            "FREQ": 20.0,
            "DURATION_MS": 500.0,
            "MAX_ISI_DEV_MS": (1.0/20.0 * 1000.0) * 0.8,
            "N_DEV": 2,
            "T_DEV_MS": 2.0,
            "N_CHANNELS": 100,
            "SEED": 7
        }
    ]

    sync_seq, async_seq, seq_json = make_and_save_sequence_from_configs(configs,
                                                                         out_dir="rasters_sequence",
                                                                         gap_ms=10.0)

    print("Sequence generation completed")
