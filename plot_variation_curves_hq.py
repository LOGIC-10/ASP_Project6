"""
High-fidelity pitch/tempo sweep with caching and parallel execution.

Outputs:
- plots/pitch_curve_hq.png
- plots/tempo_curve_hq.png
- plots/pitch_tempo_heatmap_hq.png

Runs librosa pitch_shift/time_stretch (high quality). Uses cached fingerprints to avoid rebuild.
Exec in asp env:
    conda run -n asp python plot_variation_curves_hq.py
"""
from __future__ import annotations

import gzip
import os
import pickle
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from shazam_system import (
    compute_fingerprints,
    compute_fingerprints_chroma,
    identify_song,
    identify_song_chroma,
    identify_song_multi_tempo,
    load_music_db,
    _iter_music_db,
)

CACHE_DIR = Path("cache")
PLOTS_DIR = Path("plots")
FP_BASE_CACHE = CACHE_DIR / "fp_base.pkl.gz"
FP_CHROMA_CACHE = CACHE_DIR / "fp_chroma.pkl.gz"
MAT_PATH = "Project6_musicDB.mat"


# -------------------- caching -------------------- #
def _save_fp(fp_obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(fp_obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_fp(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def ensure_fingerprint_caches(mat_path: str) -> Tuple[Path, Path]:
    if not FP_BASE_CACHE.exists():
        print("Building baseline fingerprints (cache miss)...", flush=True)
        db = load_music_db(mat_path)
        fp = compute_fingerprints(db)
        _save_fp(fp, FP_BASE_CACHE)
    else:
        print("Using cached baseline fingerprints.", flush=True)
    if not FP_CHROMA_CACHE.exists():
        print("Building chroma fingerprints (cache miss)...", flush=True)
        db = load_music_db(mat_path)
        fp = compute_fingerprints_chroma(db)
        _save_fp(fp, FP_CHROMA_CACHE)
    else:
        print("Using cached chroma fingerprints.", flush=True)
    return FP_BASE_CACHE, FP_CHROMA_CACHE


# -------------------- worker helpers -------------------- #
def _sample_clip(sig: np.ndarray, fs: int, clip_sec: float, rng: np.random.Generator) -> np.ndarray:
    clip_len = int(fs * clip_sec)
    if len(sig) <= clip_len:
        return sig.copy()
    start = int(rng.integers(0, len(sig) - clip_len))
    return sig[start : start + clip_len]


# -------------------- main -------------------- #
def main():
    fs = 16000
    clip_sec = 3.0
    pitch_vals = [-4, -2, 0, 2, 4]  # moderate grid for runtime
    tempo_vals = [0.80, 0.90, 1.00, 1.10, 1.20]
    heat_pitch = [-4, 0, 4]
    heat_tempo = [0.9, 1.0, 1.1]
    n_queries_curve = 4
    n_queries_heat = 2
    seeds_base = 123

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_fingerprint_caches(MAT_PATH)

    # Load db and fingerprints once
    print("Loading musicDB and fingerprints into memory...", flush=True)
    flat_db = list(_iter_music_db(load_music_db(MAT_PATH)))
    fp_base = _load_fp(FP_BASE_CACHE)
    fp_chroma = _load_fp(FP_CHROMA_CACHE)

    rng = np.random.default_rng(seeds_base)

    def eval_pitch(semi: float) -> Tuple[float, float]:
        def pitch_shift(clip):
            return librosa.effects.pitch_shift(y=clip, sr=fs, n_steps=semi)

        hits_b = hits_c = 0
        for _ in range(n_queries_curve):
            idx = int(rng.integers(0, len(flat_db)))
            sig = flat_db[idx][2]
            clip = pitch_shift(_sample_clip(sig, fs, clip_sec, rng))
            pred_b, _ = identify_song(clip, fp_base, fs=fs, return_info=True)
            pred_c, _ = identify_song_chroma(clip, fp_chroma, fs=fs, return_info=True)
            hits_b += int(pred_b == idx)
            hits_c += int(pred_c == idx)
        return hits_b / n_queries_curve, hits_c / n_queries_curve

    def eval_tempo(rate: float) -> Tuple[float, float]:
        def stretch(clip):
            return librosa.effects.time_stretch(y=clip, rate=rate)

        hits_b = hits_c = 0
        for _ in range(n_queries_curve):
            idx = int(rng.integers(0, len(flat_db)))
            sig = flat_db[idx][2]
            clip = stretch(_sample_clip(sig, fs, clip_sec, rng))
            pred_b, _ = identify_song_multi_tempo(clip, fp_base, fs=fs, tempo_factors=(0.9, 1.0, 1.1), return_info=True)
            pred_c, _ = identify_song_multi_tempo(
                clip, fp_chroma, fs=fs, tempo_factors=(0.9, 1.0, 1.1), identify_fn=identify_song_chroma, return_info=True
            )
            hits_b += int(pred_b == idx)
            hits_c += int(pred_c == idx)
        return hits_b / n_queries_curve, hits_c / n_queries_curve

    print("Evaluating pitch curve (high fidelity, sequential)...", flush=True)
    pitch_acc_base = []
    pitch_acc_chroma = []
    for semi in pitch_vals:
        acc_b, acc_c = eval_pitch(semi)
        pitch_acc_base.append(acc_b)
        pitch_acc_chroma.append(acc_c)

    print("Evaluating tempo curve (high fidelity, sequential)...", flush=True)
    tempo_acc_base = []
    tempo_acc_chroma = []
    for rate in tempo_vals:
        acc_b, acc_c = eval_tempo(rate)
        tempo_acc_base.append(acc_b)
        tempo_acc_chroma.append(acc_c)

    print("Evaluating pitch-tempo heatmap (chroma)...", flush=True)
    heat_acc = np.zeros((len(heat_tempo), len(heat_pitch)))
    for i, t in enumerate(heat_tempo):
        for j, p in enumerate(heat_pitch):
            hits = 0
            for _ in range(n_queries_heat):
                idx = int(rng.integers(0, len(flat_db)))
                sig = flat_db[idx][2]
                clip = librosa.effects.pitch_shift(y=_sample_clip(sig, fs, clip_sec, rng), sr=fs, n_steps=p)
                clip = librosa.effects.time_stretch(y=clip, rate=t)
                pred, _ = identify_song_chroma(clip, fp_chroma, fs=fs, return_info=True)
                hits += int(pred == idx)
            heat_acc[i, j] = hits / n_queries_heat

    # Plot pitch curve
    plt.figure(figsize=(7, 4))
    plt.plot(pitch_vals, pitch_acc_base, marker="o", label="Baseline")
    plt.plot(pitch_vals, pitch_acc_chroma, marker="o", label="Chroma")
    plt.xlabel("Pitch shift (semitones)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("Accuracy vs Pitch shift (high fidelity)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pitch_curve_hq.png", dpi=200)

    # Plot tempo curve
    plt.figure(figsize=(7, 4))
    plt.plot(tempo_vals, tempo_acc_base, marker="o", label="Baseline + multi-tempo")
    plt.plot(tempo_vals, tempo_acc_chroma, marker="o", label="Chroma + multi-tempo")
    plt.xlabel("Tempo factor")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("Accuracy vs Tempo change (high fidelity)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "tempo_curve_hq.png", dpi=200)

    # Heatmap
    plt.figure(figsize=(6, 4.5))
    im = plt.imshow(heat_acc, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, label="Accuracy")
    plt.xticks(range(len(heat_pitch)), heat_pitch)
    plt.yticks(range(len(heat_tempo)), heat_tempo)
    plt.xlabel("Pitch shift (semitones)")
    plt.ylabel("Tempo factor")
    plt.title("Chroma accuracy heatmap (pitch x tempo)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pitch_tempo_heatmap_hq.png", dpi=200)

    print("Saved high-fidelity plots to plots/*.png")


if __name__ == "__main__":
    main()
