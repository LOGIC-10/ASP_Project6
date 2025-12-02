"""
Self-check script for Project 6 fingerprinting.

Runs a small evaluation over random 3 s clips (clean and noisy) to report accuracy
and timing. Make sure the conda env `asp` is active when executing:
    conda run -n asp python evaluate_system.py
"""
from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np

from shazam_system import (
    compute_fingerprints,
    identify_song,
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


def _flatten_db(music_db) -> List[Tuple[str, str, np.ndarray]]:
    return list(_iter_music_db(music_db))


def main():
    rng = np.random.default_rng(7)
    fs = 16000
    clip_sec = 3.0
    n_queries = 24
    noisy_snr_db = 0.0  # evaluate robustness at 0 dB

    print("Loading musicDB...")
    music_db = load_music_db("Project6_musicDB.mat")
    flat_db = _flatten_db(music_db)

    print("Building fingerprints...")
    t0 = time.time()
    fingerprints = compute_fingerprints(music_db)
    build_time = time.time() - t0
    total_hashes = sum(len(v) for v in fingerprints.hash_index.values())
    approx_mem_mb = total_hashes * 2 * 4 / (1024 * 1024)  # two ints per entry, 4 bytes each
    print(f"Songs indexed: {len(fingerprints)}, total hashes: {total_hashes}, ~{approx_mem_mb:.2f} MB")
    print(f"Build time: {build_time:.2f} s")

    def run_trials(noisy: bool):
        hits = 0
        latencies = []
        for _ in range(n_queries):
            song_idx = int(rng.integers(0, len(flat_db)))
            title, genre, sig = flat_db[song_idx]
            clip = _sample_clip(sig, fs, clip_sec, rng)
            if noisy:
                clip = _add_noise(clip, noisy_snr_db, rng)
            t_start = time.perf_counter()
            pred, info = identify_song(clip, fingerprints, fs=fs, return_info=True)
            latencies.append(time.perf_counter() - t_start)
            if pred == song_idx:
                hits += 1
        return hits, latencies

    print(f"Running {n_queries} clean queries...")
    clean_hits, clean_lat = run_trials(noisy=False)
    print(f"Clean accuracy: {clean_hits}/{n_queries} = {clean_hits / n_queries:.3f}")
    print(f"Avg lookup time: {np.mean(clean_lat)*1000:.1f} ms")

    print(f"Running {n_queries} noisy queries at {noisy_snr_db} dB SNR...")
    noisy_hits, noisy_lat = run_trials(noisy=True)
    print(f"Noisy accuracy: {noisy_hits}/{n_queries} = {noisy_hits / n_queries:.3f}")
    print(f"Avg lookup time: {np.mean(noisy_lat)*1000:.1f} ms")


if __name__ == "__main__":
    main()

