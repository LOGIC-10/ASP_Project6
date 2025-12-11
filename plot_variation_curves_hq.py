"""
High-fidelity pitch/tempo sweep with caching and parallel execution.

Outputs:
- plots/pitch_curve_hq.png
- plots/tempo_curve_hq.png
- plots/pitch_tempo_heatmap_hq.png

Runs librosa pitch_shift/time_stretch (high quality). Uses cached fingerprints to avoid rebuild.
Exec in asp env:
    conda run -n asp python plot_variation_curves_hq.py --from-cache cache/variation_results.json

If --from-cache is not provided, will run fresh sweeps (may be long). Prefer generating data with variation_runner.py.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

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


def _load_flat_db(mat_path: str):
    data = sio.loadmat(mat_path)
    db = data["musicDB"]
    return list(_iter_music_db(db))


def _eval_pitch_worker(
    semi: float,
    n_queries: int,
    fs: int,
    clip_sec: float,
    mat_path: str,
    fp_base_path: str,
    fp_chroma_path: str,
    seed: int,
) -> Tuple[float, float]:
    flat_db = _load_flat_db(mat_path)
    fp_base = _load_fp(Path(fp_base_path))
    fp_chroma = _load_fp(Path(fp_chroma_path))
    rng = np.random.default_rng(seed + int(semi * 10))
    hits_b = hits_c = 0
    for _ in range(n_queries):
        idx = int(rng.integers(0, len(flat_db)))
        sig = flat_db[idx][2]
        clip = librosa.effects.pitch_shift(y=_sample_clip(sig, fs, clip_sec, rng), sr=fs, n_steps=semi)
        pred_b, _ = identify_song(clip, fp_base, fs=fs, return_info=True)
        pred_c, _ = identify_song_chroma(clip, fp_chroma, fs=fs, return_info=True)
        hits_b += int(pred_b == idx)
        hits_c += int(pred_c == idx)
    return hits_b / n_queries, hits_c / n_queries


def _eval_tempo_worker(
    rate: float,
    n_queries: int,
    fs: int,
    clip_sec: float,
    mat_path: str,
    fp_base_path: str,
    fp_chroma_path: str,
    seed: int,
) -> Tuple[float, float]:
    flat_db = _load_flat_db(mat_path)
    fp_base = _load_fp(Path(fp_base_path))
    fp_chroma = _load_fp(Path(fp_chroma_path))
    rng = np.random.default_rng(seed + int(rate * 100))
    hits_b = hits_c = 0
    for _ in range(n_queries):
        idx = int(rng.integers(0, len(flat_db)))
        sig = flat_db[idx][2]
        clip = librosa.effects.time_stretch(y=_sample_clip(sig, fs, clip_sec, rng), rate=rate)
        pred_b, _ = identify_song_multi_tempo(clip, fp_base, fs=fs, tempo_factors=(0.9, 1.0, 1.1), return_info=True)
        pred_c, _ = identify_song_multi_tempo(
            clip, fp_chroma, fs=fs, tempo_factors=(0.9, 1.0, 1.1), identify_fn=identify_song_chroma, return_info=True
        )
        hits_b += int(pred_b == idx)
        hits_c += int(pred_c == idx)
    return hits_b / n_queries, hits_c / n_queries


def _eval_heat_worker(
    p: float,
    t: float,
    n_queries: int,
    fs: int,
    clip_sec: float,
    mat_path: str,
    fp_chroma_path: str,
    seed: int,
) -> float:
    flat_db = _load_flat_db(mat_path)
    fp_chroma = _load_fp(Path(fp_chroma_path))
    rng = np.random.default_rng(seed + int(p * 10) + int(t * 100))
    hits = 0
    for _ in range(n_queries):
        idx = int(rng.integers(0, len(flat_db)))
        sig = flat_db[idx][2]
        clip = librosa.effects.pitch_shift(y=_sample_clip(sig, fs, clip_sec, rng), sr=fs, n_steps=p)
        clip = librosa.effects.time_stretch(y=clip, rate=t)
        pred, _ = identify_song_chroma(clip, fp_chroma, fs=fs, return_info=True)
        hits += int(pred == idx)
    return hits / n_queries


# -------------------- main -------------------- #
def main():
    parser = argparse.ArgumentParser(description="High-fidelity pitch/tempo sweep and plotting.")
    parser.add_argument("--from-cache", type=str, default=None, help="Path to variation_results.json to plot without recompute")
    parser.add_argument("--workers", type=int, default=4, help="Process pool size when computing")
    parser.add_argument("--nq_curve", type=int, default=4, help="Queries per factor for pitch/tempo")
    parser.add_argument("--nq_heat", type=int, default=2, help="Queries per cell for heatmap")
    args = parser.parse_args()

    fs = 16000
    clip_sec = 3.0
    pitch_vals = [-4, -2, 0, 2, 4]  # moderate grid for runtime
    tempo_vals = [0.80, 0.90, 1.00, 1.10, 1.20]
    heat_pitch = [-4, 0, 4]
    heat_tempo = [0.9, 1.0, 1.1]
    n_queries_curve = args.nq_curve
    n_queries_heat = args.nq_heat
    max_workers = args.workers  # parallel processes for factor sweeps
    seeds_base = 123

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.from_cache:
        cache_path = Path(args.from_cache)
        if not cache_path.exists():
            raise FileNotFoundError(f"{cache_path} not found. Run variation_runner.py to generate it.")
        cached = json.loads(cache_path.read_text())
        pitch_vals = cached.get("pitch_vals", pitch_vals)
        tempo_vals = cached.get("tempo_vals", tempo_vals)
        heat_pitch = cached.get("heat_pitch", heat_pitch)
        heat_tempo = cached.get("heat_tempo", heat_tempo)
        pitch_acc_base = cached.get("pitch_base")
        pitch_acc_chroma = cached.get("pitch_chroma")
        tempo_acc_base = cached.get("tempo_base")
        tempo_acc_chroma = cached.get("tempo_chroma")
        heat_acc = np.array(cached.get("heat_acc")) if cached.get("heat_acc") is not None else None
        if pitch_acc_base is None or tempo_acc_base is None or heat_acc is None:
            raise ValueError("Cache file missing required fields. Re-run variation_runner.py.")
        print(f"Loaded results from {cache_path}")
    else:
        ensure_fingerprint_caches(MAT_PATH)

        print(f"Evaluating pitch curve (high fidelity, parallel, workers={max_workers})...", flush=True)
        pitch_acc_base = [0.0] * len(pitch_vals)
        pitch_acc_chroma = [0.0] * len(pitch_vals)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            for idx, semi in enumerate(pitch_vals):
                futures[ex.submit(
                    _eval_pitch_worker,
                    semi,
                    n_queries_curve,
                    fs,
                    clip_sec,
                    MAT_PATH,
                    str(FP_BASE_CACHE),
                    str(FP_CHROMA_CACHE),
                    seeds_base,
                )] = idx
            for fut in as_completed(futures):
                i = futures[fut]
                acc_b, acc_c = fut.result()
                pitch_acc_base[i] = acc_b
                pitch_acc_chroma[i] = acc_c

        print(f"Evaluating tempo curve (high fidelity, parallel, workers={max_workers})...", flush=True)
        tempo_acc_base = [0.0] * len(tempo_vals)
        tempo_acc_chroma = [0.0] * len(tempo_vals)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            for idx, rate in enumerate(tempo_vals):
                futures[ex.submit(
                    _eval_tempo_worker,
                    rate,
                    n_queries_curve,
                    fs,
                    clip_sec,
                    MAT_PATH,
                    str(FP_BASE_CACHE),
                    str(FP_CHROMA_CACHE),
                    seeds_base + 50,
                )] = idx
            for fut in as_completed(futures):
                i = futures[fut]
                acc_b, acc_c = fut.result()
                tempo_acc_base[i] = acc_b
                tempo_acc_chroma[i] = acc_c

        print(f"Evaluating pitch-tempo heatmap (chroma, parallel, workers={max_workers})...", flush=True)
        heat_acc = np.zeros((len(heat_tempo), len(heat_pitch)))
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            for i, t in enumerate(heat_tempo):
                for j, p in enumerate(heat_pitch):
                    futures[(i, j)] = ex.submit(
                        _eval_heat_worker,
                        p,
                        t,
                        n_queries_heat,
                        fs,
                        clip_sec,
                        MAT_PATH,
                        str(FP_CHROMA_CACHE),
                        seeds_base + 200 + i * 10 + j,
                    )
            for (i, j), fut in futures.items():
                heat_acc[i, j] = fut.result()

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
