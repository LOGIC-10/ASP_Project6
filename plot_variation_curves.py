"""
Generate pitch/tempo accuracy curves and a pitch-tempo heatmap.

Outputs PNGs under plots/:
- plots/pitch_curve.png
- plots/tempo_curve.png
- plots/pitch_tempo_heatmap.png

Run (asp env):
    conda run -n asp python plot_variation_curves.py
"""
from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample

from shazam_system import (
    compute_fingerprints,
    compute_fingerprints_chroma,
    identify_song,
    identify_song_chroma,
    identify_song_multi_tempo,
    load_music_db,
    _iter_music_db,
)


def _sample_clip(sig: np.ndarray, fs: int, clip_sec: float, rng: np.random.Generator) -> np.ndarray:
    clip_len = int(fs * clip_sec)
    if len(sig) <= clip_len:
        return sig.copy()
    start = int(rng.integers(0, len(sig) - clip_len))
    return sig[start : start + clip_len]


def _eval_curve(
    factors: Sequence[float],
    transform: Callable[[np.ndarray, float], np.ndarray],
    identify_fn: Callable[[np.ndarray], Tuple[int, Dict]],
    flat_db,
    fs: int,
    rng: np.random.Generator,
    clip_sec: float,
    n_queries: int,
) -> List[float]:
    accs = []
    for factor in factors:
        hits = 0
        for _ in range(n_queries):
            song_idx = int(rng.integers(0, len(flat_db)))
            _, _, sig = flat_db[song_idx]
            clip = _sample_clip(sig, fs, clip_sec, rng)
            clip = transform(clip, factor)
            pred, _ = identify_fn(clip)
            if pred == song_idx:
                hits += 1
        accs.append(hits / float(n_queries))
    return accs


def _eval_heatmap(
    pitch_factors: Sequence[int],
    tempo_factors: Sequence[float],
    transform: Callable[[np.ndarray, int, float], np.ndarray],
    identify_fn: Callable[[np.ndarray], Tuple[int, Dict]],
    flat_db,
    fs: int,
    rng: np.random.Generator,
    clip_sec: float,
    n_queries: int,
) -> np.ndarray:
    H = np.zeros((len(tempo_factors), len(pitch_factors)))
    for i, tempo in enumerate(tempo_factors):
        for j, semi in enumerate(pitch_factors):
            hits = 0
            for _ in range(n_queries):
                song_idx = int(rng.integers(0, len(flat_db)))
                _, _, sig = flat_db[song_idx]
                clip = _sample_clip(sig, fs, clip_sec, rng)
                clip = transform(clip, semi, tempo)
                pred, _ = identify_fn(clip)
                if pred == song_idx:
                    hits += 1
            H[i, j] = hits / float(n_queries)
    return H


def main():
    fs = 16000
    clip_sec = 3.0
    rng = np.random.default_rng(42)
    n_queries_curve = 2   # per factor (runtime-friendly)
    n_queries_heat = 1    # per cell

    os.makedirs("plots", exist_ok=True)

    print("Loading musicDB...", flush=True)
    music_db = load_music_db("Project6_musicDB.mat")
    flat_db = list(_iter_music_db(music_db))

    print("Building fingerprints (baseline)...", flush=True)
    t0 = time.time()
    fp_base = compute_fingerprints(music_db)
    print(f"  done in {time.time() - t0:.2f} s")

    print("Building fingerprints (chroma)...", flush=True)
    t0 = time.time()
    fp_chroma = compute_fingerprints_chroma(music_db)
    print(f"  done in {time.time() - t0:.2f} s")

    # Pitch curve factors and evaluators
    pitch_vals = [-6, -4, -2, 0, 2, 4, 6]
    def _pitch_shift_resample(clip: np.ndarray, semitones: float) -> np.ndarray:
        factor = 2.0 ** (semitones / 12.0)
        new_len = max(1, int(round(len(clip) / factor)))
        return resample(clip, new_len)

    pitch_transform = _pitch_shift_resample
    identify_base = lambda clip: identify_song(clip, fp_base, fs=fs, return_info=True)
    identify_chroma = lambda clip: identify_song_chroma(clip, fp_chroma, fs=fs, return_info=True)

    print("Evaluating pitch curve...", flush=True)
    pitch_acc_base = _eval_curve(pitch_vals, pitch_transform, identify_base, flat_db, fs, rng, clip_sec, n_queries_curve)
    pitch_acc_chroma = _eval_curve(pitch_vals, pitch_transform, identify_chroma, flat_db, fs, rng, clip_sec, n_queries_curve)

    # Tempo curve factors and evaluators (system-level tempo robustness)
    tempo_vals = [0.80, 0.90, 1.00, 1.10, 1.20]
    def _time_stretch_resample(clip: np.ndarray, rate: float) -> np.ndarray:
        new_len = max(1, int(round(len(clip) / rate)))
        return resample(clip, new_len)

    tempo_transform = _time_stretch_resample
    identify_base_multi = lambda clip: identify_song_multi_tempo(
        clip,
        fp_base,
        fs=fs,
        tempo_factors=(0.9, 1.0, 1.1),
        identify_fn=identify_song,
        return_info=True,
    )
    identify_chroma_multi = lambda clip: identify_song_multi_tempo(
        clip,
        fp_chroma,
        fs=fs,
        tempo_factors=(0.9, 1.0, 1.1),
        identify_fn=identify_song_chroma,
        return_info=True,
    )

    print("Evaluating tempo curve...", flush=True)
    tempo_acc_base = _eval_curve(tempo_vals, tempo_transform, identify_base_multi, flat_db, fs, rng, clip_sec, n_queries_curve)
    tempo_acc_chroma = _eval_curve(tempo_vals, tempo_transform, identify_chroma_multi, flat_db, fs, rng, clip_sec, n_queries_curve)

    # Heatmap on reduced grid (chroma + multi-tempo)
    heat_pitch = [-4, 0, 4]
    heat_tempo = [0.9, 1.0, 1.1]
    heat_transform = lambda clip, semi, rate: _time_stretch_resample(_pitch_shift_resample(clip, semi), rate)
    identify_heat = identify_chroma

    print("Evaluating heatmap grid...", flush=True)
    heat_acc = _eval_heatmap(heat_pitch, heat_tempo, heat_transform, identify_heat, flat_db, fs, rng, clip_sec, n_queries_heat)

    # Plot pitch curve
    plt.figure(figsize=(7, 4))
    plt.plot(pitch_vals, pitch_acc_base, marker="o", label="Design A (baseline)")
    plt.plot(pitch_vals, pitch_acc_chroma, marker="o", label="Design B (chroma)")
    plt.xlabel("Pitch shift (semitones)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("Accuracy vs Pitch shift")
    plt.tight_layout()
    plt.savefig("plots/pitch_curve.png", dpi=200)

    # Plot tempo curve
    plt.figure(figsize=(7, 4))
    plt.plot(tempo_vals, tempo_acc_base, marker="o", label="Baseline + multi-tempo search")
    plt.plot(tempo_vals, tempo_acc_chroma, marker="o", label="Chroma + multi-tempo search")
    plt.xlabel("Tempo factor")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("Accuracy vs Tempo change")
    plt.tight_layout()
    plt.savefig("plots/tempo_curve.png", dpi=200)

    # Heatmap
    plt.figure(figsize=(6, 4.5))
    im = plt.imshow(heat_acc, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, label="Accuracy")
    plt.xticks(range(len(heat_pitch)), heat_pitch)
    plt.yticks(range(len(heat_tempo)), heat_tempo)
    plt.xlabel("Pitch shift (semitones)")
    plt.ylabel("Tempo factor")
    plt.title("Chroma + multi-tempo accuracy heatmap")
    plt.tight_layout()
    plt.savefig("plots/pitch_tempo_heatmap.png", dpi=200)

    print("Saved plots to plots/*.png")


if __name__ == "__main__":
    main()
