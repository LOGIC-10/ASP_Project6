"""
Design comparison, ablation, and robustness evaluation.

Run in the asp conda env:
    conda run -n asp python compare_designs.py
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Tuple

import numpy as np

from shazam_system import (
    compute_fingerprints,
    compute_fingerprints_chroma,
    identify_song,
    identify_song_chroma,
    load_music_db,
    _iter_music_db,
)


def _sample_clip(sig: np.ndarray, fs: int, clip_sec: float, rng: np.random.Generator) -> np.ndarray:
    clip_len = int(fs * clip_sec)
    if len(sig) <= clip_len:
        return sig.copy()
    start = int(rng.integers(0, len(sig) - clip_len))
    return sig[start : start + clip_len]


def _add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    power = np.mean(x ** 2) + 1e-12
    noise_power = power / (10 ** (snr_db / 10.0))
    noise = rng.normal(scale=np.sqrt(noise_power), size=x.shape)
    return x + noise


def _pitch_shift(x: np.ndarray, fs: int, semitones: float) -> np.ndarray:
    import librosa

    return librosa.effects.pitch_shift(y=x, sr=fs, n_steps=semitones)


def _time_stretch(x: np.ndarray, rate: float) -> np.ndarray:
    from scipy.signal import resample

    new_len = max(1, int(round(len(x) / rate)))
    return resample(x, new_len)


def _flatten_db(music_db):
    return list(_iter_music_db(music_db))


def _evaluate(
    fingerprints,
    identify_fn: Callable[..., Tuple[int, Dict]],
    fs: int,
    flat_db,
    rng: np.random.Generator,
    clip_sec: float,
    transform: Callable[[np.ndarray], np.ndarray],
    n_queries: int,
) -> Tuple[int, List[float]]:
    hits = 0
    latencies: List[float] = []
    for _ in range(n_queries):
        song_idx = int(rng.integers(0, len(flat_db)))
        _, _, sig = flat_db[song_idx]
        clip = _sample_clip(sig, fs, clip_sec, rng)
        clip = transform(clip)
        t_start = time.perf_counter()
        pred, _ = identify_fn(clip, fingerprints, fs=fs, return_info=True)
        latencies.append(time.perf_counter() - t_start)
        if pred == song_idx:
            hits += 1
    return hits, latencies


def describe_fp(name: str, fp) -> None:
    total_hashes = sum(len(v) for v in fp.hash_index.values())
    approx_mem_mb = total_hashes * 8 / (1024 * 1024)  # two 32-bit ints
    avg_peaks = np.mean([s.get("num_peaks", 0) for s in fp.songs])
    avg_hashes = np.mean([s.get("num_hashes", 0) for s in fp.songs])
    avg_build = np.mean([s.get("build_time_sec", 0) for s in fp.songs])
    print(f"{name}: hashes={total_hashes:,} (~{approx_mem_mb:.2f} MB), avg peaks={avg_peaks:.1f}, avg hashes={avg_hashes:.1f}, avg build/song={avg_build*1000:.1f} ms")


def main():
    fs = 16000
    clip_sec = 3.0
    rng = np.random.default_rng(11)
    n_queries = 12

    print("Loading musicDB...", flush=True)
    music_db = load_music_db("Project6_musicDB.mat")
    flat_db = _flatten_db(music_db)

    # Design A: baseline constellation
    print("\n=== Build fingerprints: baseline (Design A) ===", flush=True)
    t0 = time.time()
    fp_base = compute_fingerprints(music_db)
    build_base = time.time() - t0
    print(f"Build time: {build_base:.2f} s", flush=True)
    describe_fp("Design A", fp_base)

    # Design A ablation: remove spectral whitening
    print("\n=== Build fingerprints: ablation (no-whiten) ===", flush=True)
    t0 = time.time()
    fp_ablate = compute_fingerprints(music_db, params={"whiten": False, "peak_threshold_rel": 24.0, "peak_percentile": 68.0})
    build_ablate = time.time() - t0
    print(f"Build time: {build_ablate:.2f} s", flush=True)
    describe_fp("Ablation", fp_ablate)

    # Design B: chroma-based
    print("\n=== Build fingerprints: chroma (Design B) ===", flush=True)
    t0 = time.time()
    fp_chroma = compute_fingerprints_chroma(music_db)
    build_chroma = time.time() - t0
    print(f"Build time: {build_chroma:.2f} s", flush=True)
    describe_fp("Design B", fp_chroma)

    scenarios = [
        ("clean", lambda x: x),
        ("noise_0dB", lambda x: _add_noise(x, 0.0, rng)),
        ("pitch+2", lambda x: _pitch_shift(x, fs, 2.0)),
        ("tempo0.9", lambda x: _time_stretch(x, 0.9)),
    ]

    def run_design(label: str, fp, identify_fn):
        print(f"\n--- {label} ---", flush=True)
        for name, transform in scenarios:
            hits, lat = _evaluate(fp, identify_fn, fs, flat_db, rng, clip_sec, transform, n_queries)
            acc = hits / float(n_queries)
            print(f"{name:10s} acc={acc:0.3f} ({hits}/{n_queries}), avg lookup={np.mean(lat)*1000:.1f} ms", flush=True)

    run_design("Design A (baseline)", fp_base, identify_song)
    run_design("Ablation (no-whiten)", fp_ablate, identify_song)
    run_design("Design B (chroma, pitch-robust)", fp_chroma, identify_song_chroma)


if __name__ == "__main__":
    main()
