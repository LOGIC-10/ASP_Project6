"""
Pitch/tempo variation sweep for Project 6.

Runs accuracy vs. pitch shift and tempo scaling for:
- Baseline constellation (Design A)
- Baseline + multi-tempo search (system-level tempo robustness)
- Chroma (pitch-robust fingerprint)

Execute in asp env:
    conda run -n asp python evaluate_variations.py
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Tuple

import numpy as np

import librosa

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
    name: str,
    factors: List[float],
    transform: Callable[[np.ndarray, float], np.ndarray],
    identify_fn: Callable[[np.ndarray], Tuple[int, Dict]],
    flat_db,
    fs: int,
    rng: np.random.Generator,
    clip_sec: float,
    n_queries: int,
) -> Dict[float, Tuple[float, float]]:
    results = {}
    for factor in factors:
        hits = 0
        lat = []
        for _ in range(n_queries):
            song_idx = int(rng.integers(0, len(flat_db)))
            _, _, sig = flat_db[song_idx]
            clip = _sample_clip(sig, fs, clip_sec, rng)
            clip = transform(clip, factor)
            t0 = time.perf_counter()
            pred, _ = identify_fn(clip)
            lat.append(time.perf_counter() - t0)
            if pred == song_idx:
                hits += 1
        acc = hits / float(n_queries)
        results[factor] = (acc, float(np.mean(lat) * 1000.0))
    print(f"\n{name}")
    for f, (acc, l_ms) in results.items():
        print(f"  factor={f:+.2f} acc={acc:.3f} avg_lookup={l_ms:.1f} ms")
    return results


def main():
    fs = 16000
    clip_sec = 3.0
    n_queries = 8  # per factor
    rng = np.random.default_rng(23)

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

    # Identify functions
    identify_base = lambda clip: identify_song(clip, fp_base, fs=fs, return_info=True)
    identify_base_multi = lambda clip: identify_song_multi_tempo(
        clip,
        fp_base,
        fs=fs,
        tempo_factors=(0.85, 0.9, 1.0, 1.1, 1.15),
        identify_fn=identify_song,
        return_info=True,
    )
    identify_chroma = lambda clip: identify_song_chroma(clip, fp_chroma, fs=fs, return_info=True)

    # Transforms
    pitch_factors = [-4.0, -2.0, 0.0, 2.0, 4.0]  # semitones
    tempo_factors = [0.80, 0.90, 1.00, 1.10, 1.20]  # time-stretch rates

    pitch_transform = lambda clip, semitones: librosa.effects.pitch_shift(y=clip, sr=fs, n_steps=semitones)
    tempo_transform = lambda clip, rate: librosa.effects.time_stretch(y=clip, rate=rate)

    print("\n=== Pitch sweep (semitones) ===", flush=True)
    pitch_base = _eval_curve("Base", pitch_factors, pitch_transform, identify_base, flat_db, fs, rng, clip_sec, n_queries)
    pitch_base_multi = _eval_curve(
        "Base + multi-tempo", pitch_factors, pitch_transform, identify_base_multi, flat_db, fs, rng, clip_sec, n_queries
    )
    pitch_chroma = _eval_curve("Chroma", pitch_factors, pitch_transform, identify_chroma, flat_db, fs, rng, clip_sec, n_queries)

    print("\n=== Tempo sweep (rate) ===", flush=True)
    tempo_base = _eval_curve("Base", tempo_factors, tempo_transform, identify_base, flat_db, fs, rng, clip_sec, n_queries)
    tempo_base_multi = _eval_curve(
        "Base + multi-tempo", tempo_factors, tempo_transform, identify_base_multi, flat_db, fs, rng, clip_sec, n_queries
    )
    tempo_chroma = _eval_curve("Chroma", tempo_factors, tempo_transform, identify_chroma, flat_db, fs, rng, clip_sec, n_queries)

    print("\nSummary (use in report):")
    print("Pitch sweep:", pitch_base, pitch_base_multi, pitch_chroma)
    print("Tempo sweep:", tempo_base, tempo_base_multi, tempo_chroma)


if __name__ == "__main__":
    main()

