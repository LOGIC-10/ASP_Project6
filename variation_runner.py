"""
Long-running, cache-friendly variation sweeps for pitch/tempo with result persistence.

Usage (asp env):
  # 1) Ensure fingerprint caches
  conda run -n asp python -c "from variation_runner import ensure_fingerprint_caches; ensure_fingerprint_caches('Project6_musicDB.mat')"

  # 2) Run pitch sweep (writes/updates cache/variation_results.json)
  conda run -n asp python variation_runner.py --mode pitch

  # 3) Run tempo sweep
  conda run -n asp python variation_runner.py --mode tempo

  # 4) Run heatmap
  conda run -n asp python variation_runner.py --mode heatmap

  # 5) Plot using cached results
  conda run -n asp python plot_variation_curves_hq.py --from-cache cache/variation_results.json

You can also run all modes in one go:
  conda run -n asp python variation_runner.py --mode all
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import librosa
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
FP_BASE_CACHE = CACHE_DIR / "fp_base.pkl.gz"
FP_CHROMA_CACHE = CACHE_DIR / "fp_chroma.pkl.gz"
RESULTS_JSON = CACHE_DIR / "variation_results.json"
MAT_PATH = "Project6_musicDB.mat"
FS = 16000
CLIP_SEC = 3.0


def _save_fp(fp_obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(fp_obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_fp(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def ensure_fingerprint_caches(mat_path: str = MAT_PATH):
    if not FP_BASE_CACHE.exists():
        print("Building baseline fingerprints (cache miss)...", flush=True)
        db = load_music_db(mat_path)
        _save_fp(compute_fingerprints(db), FP_BASE_CACHE)
    else:
        print("Using cached baseline fingerprints.")

    if not FP_CHROMA_CACHE.exists():
        print("Building chroma fingerprints (cache miss)...", flush=True)
        db = load_music_db(mat_path)
        _save_fp(compute_fingerprints_chroma(db), FP_CHROMA_CACHE)
    else:
        print("Using cached chroma fingerprints.")


def _load_flat_db(mat_path: str):
    data = sio.loadmat(mat_path)
    db = data["musicDB"]
    return list(_iter_music_db(db))


def _sample_clip(sig: np.ndarray, fs: int, clip_sec: float, rng: np.random.Generator) -> np.ndarray:
    L = int(fs * clip_sec)
    if len(sig) <= L:
        return sig.copy()
    start = int(rng.integers(0, len(sig) - L))
    return sig[start : start + L]


def _eval_pitch_worker(
    semi: float,
    n_queries: int,
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
        clip = librosa.effects.pitch_shift(y=_sample_clip(sig, FS, CLIP_SEC, rng), sr=FS, n_steps=semi)
        pred_b, _ = identify_song(clip, fp_base, fs=FS, return_info=True)
        pred_c, _ = identify_song_chroma(clip, fp_chroma, fs=FS, return_info=True)
        hits_b += int(pred_b == idx)
        hits_c += int(pred_c == idx)
    return hits_b / n_queries, hits_c / n_queries


def _eval_tempo_worker(
    rate: float,
    n_queries: int,
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
        clip = librosa.effects.time_stretch(y=_sample_clip(sig, FS, CLIP_SEC, rng), rate=rate)
        pred_b, _ = identify_song_multi_tempo(clip, fp_base, fs=FS, tempo_factors=(0.9, 1.0, 1.1), return_info=True)
        pred_c, _ = identify_song_multi_tempo(
            clip, fp_chroma, fs=FS, tempo_factors=(0.9, 1.0, 1.1), identify_fn=identify_song_chroma, return_info=True
        )
        hits_b += int(pred_b == idx)
        hits_c += int(pred_c == idx)
    return hits_b / n_queries, hits_c / n_queries


def _eval_heat_worker(
    semi: float,
    rate: float,
    n_queries: int,
    mat_path: str,
    fp_chroma_path: str,
    seed: int,
) -> float:
    flat_db = _load_flat_db(mat_path)
    fp_chroma = _load_fp(Path(fp_chroma_path))
    rng = np.random.default_rng(seed + int(semi * 10) + int(rate * 100))
    hits = 0
    for _ in range(n_queries):
        idx = int(rng.integers(0, len(flat_db)))
        sig = flat_db[idx][2]
        clip = librosa.effects.pitch_shift(y=_sample_clip(sig, FS, CLIP_SEC, rng), sr=FS, n_steps=semi)
        clip = librosa.effects.time_stretch(y=clip, rate=rate)
        pred, _ = identify_song_chroma(clip, fp_chroma, fs=FS, return_info=True)
        hits += int(pred == idx)
    return hits / n_queries


def save_results(data: Dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if RESULTS_JSON.exists():
        try:
            current = json.loads(RESULTS_JSON.read_text())
        except Exception:
            current = {}
    else:
        current = {}
    current.update(data)
    RESULTS_JSON.write_text(json.dumps(current, indent=2))
    print(f"Saved results to {RESULTS_JSON}")


def run_pitch(pitch_vals: Sequence[float], n_queries: int, workers: int, seed: int):
    print(f"Running pitch sweep: vals={pitch_vals}, n_queries={n_queries}, workers={workers}")
    results_b = [0.0] * len(pitch_vals)
    results_c = [0.0] * len(pitch_vals)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_eval_pitch_worker, semi, n_queries, MAT_PATH, str(FP_BASE_CACHE), str(FP_CHROMA_CACHE), seed): i
            for i, semi in enumerate(pitch_vals)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            acc_b, acc_c = fut.result()
            results_b[i] = acc_b
            results_c[i] = acc_c
    save_results(
        {
            "pitch_vals": list(pitch_vals),
            "pitch_base": results_b,
            "pitch_chroma": results_c,
            "pitch_nq": n_queries,
            "workers": workers,
        }
    )


def run_tempo(tempo_vals: Sequence[float], n_queries: int, workers: int, seed: int):
    print(f"Running tempo sweep: vals={tempo_vals}, n_queries={n_queries}, workers={workers}")
    results_b = [0.0] * len(tempo_vals)
    results_c = [0.0] * len(tempo_vals)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_eval_tempo_worker, rate, n_queries, MAT_PATH, str(FP_BASE_CACHE), str(FP_CHROMA_CACHE), seed): i
            for i, rate in enumerate(tempo_vals)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            acc_b, acc_c = fut.result()
            results_b[i] = acc_b
            results_c[i] = acc_c
    save_results(
        {
            "tempo_vals": list(tempo_vals),
            "tempo_base": results_b,
            "tempo_chroma": results_c,
            "tempo_nq": n_queries,
            "workers": workers,
        }
    )


def run_heatmap(pitch_vals: Sequence[float], tempo_vals: Sequence[float], n_queries: int, workers: int, seed: int):
    print(f"Running heatmap: pitch={pitch_vals}, tempo={tempo_vals}, n_queries={n_queries}, workers={workers}")
    H = np.zeros((len(tempo_vals), len(pitch_vals)))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {}
        for i, t in enumerate(tempo_vals):
            for j, p in enumerate(pitch_vals):
                futures[(i, j)] = ex.submit(
                    _eval_heat_worker, p, t, n_queries, MAT_PATH, str(FP_CHROMA_CACHE), seed + 100 + i * 10 + j
                )
        for (i, j), fut in futures.items():
            H[i, j] = fut.result()
    save_results(
        {
            "heat_pitch": list(pitch_vals),
            "heat_tempo": list(tempo_vals),
            "heat_acc": H.tolist(),
            "heat_nq": n_queries,
            "workers": workers,
        }
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run pitch/tempo sweeps with caching and parallelism.")
    parser.add_argument("--mode", choices=["pitch", "tempo", "heatmap", "all"], default="all")
    parser.add_argument("--workers", type=int, default=4, help="Process pool size")
    parser.add_argument("--nq_curve", type=int, default=6, help="Queries per factor for pitch/tempo curves")
    parser.add_argument("--nq_heat", type=int, default=3, help="Queries per cell for heatmap")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_fingerprint_caches(MAT_PATH)
    pitch_vals = [-4, -2, 0, 2, 4]
    tempo_vals = [0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20]
    heat_pitch = [-4, 0, 4]
    heat_tempo = [0.9, 1.0, 1.1]
    seed = 123

    if args.mode in ("pitch", "all"):
        run_pitch(pitch_vals, args.nq_curve, args.workers, seed)
    if args.mode in ("tempo", "all"):
        run_tempo(tempo_vals, args.nq_curve, args.workers, seed + 50)
    if args.mode in ("heatmap", "all"):
        run_heatmap(heat_pitch, heat_tempo, args.nq_heat, args.workers, seed + 100)


if __name__ == "__main__":
    main()

