"""
Shazam-style audio fingerprinting for Project 6.

Expose two main entry points:
- compute_fingerprints(music_db): builds an inverted index of hashes for all songs.
- identify_song(clip, fingerprints): returns the song index that best matches a 3 s clip.

The design follows a peak-constellation fingerprint with anchor/target hashing, which is
compact, fast to search, and reasonably robust to additive noise.
"""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter


# ----------------------------
# Data containers and helpers
# ----------------------------
@dataclass
class FingerprintDB:
    params: Dict[str, Any]
    songs: List[Dict[str, Any]]
    hash_index: Dict[int, List[Tuple[int, int]]]

    def __len__(self) -> int:
        return len(self.songs)


def _default_params() -> Dict[str, Any]:
    return {
        "fs": 16000,
        "n_fft": 2048,
        "hop_length": 512,
        "peak_neighborhood_freq": 16,
        "peak_neighborhood_time": 9,
        "peak_threshold_rel": 28.0,  # dB below the global max
        "peak_percentile": 72.0,  # dynamic floor to reject noise
        "min_freq_hz": 80.0,
        "max_freq_hz": 7500.0,
        "whiten": True,
        "fan_value": 8,  # how many target peaks to link per anchor
        "target_dt_min": 0.08,  # seconds
        "target_dt_max": 0.9,  # seconds
        "max_hashes_per_anchor": 15,
        "min_peaks_per_song": 30,
    }


def _preprocess_signal(x: np.ndarray) -> np.ndarray:
    """Normalize and lightly pre-emphasize to improve peak contrast."""
    x = np.asarray(x, dtype=float).flatten()
    if x.size == 0:
        return x
    x = x - np.mean(x)
    max_abs = np.max(np.abs(x))
    if max_abs > 0:
        x = x / max_abs
    # Emphasize transients/high-freq content for clearer peaks
    return signal.lfilter([1, -0.97], [1], x)


def _compute_spectrogram(
    x: np.ndarray, fs: int, n_fft: int, hop_length: int, *, whiten: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    f, t, Z = signal.stft(
        x,
        fs=fs,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary=None,
        padded=False,
    )
    mag = np.abs(Z)
    spec_db = 20 * np.log10(mag + 1e-12)
    if whiten:
        spec_db = spec_db - np.median(spec_db, axis=1, keepdims=True)
    return f, t, spec_db


def _pick_peaks(spec_db: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Return peaks as (time_bin, freq_bin) sorted by time."""
    neighborhood = (params["peak_neighborhood_freq"], params["peak_neighborhood_time"])
    local_max = maximum_filter(spec_db, size=neighborhood) == spec_db
    noise_floor = np.percentile(spec_db, params["peak_percentile"])
    amp_thresh = max(noise_floor + 5.0, spec_db.max() - params["peak_threshold_rel"])
    mask = local_max & (spec_db >= amp_thresh)
    if params.get("min_freq_hz") or params.get("max_freq_hz"):
        freqs = np.linspace(0, params["fs"] / 2.0, spec_db.shape[0])
        freq_mask = (freqs >= params["min_freq_hz"]) & (freqs <= params["max_freq_hz"])
        if freq_mask.any():
            mask = mask & freq_mask[:, None]
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.empty((0, 2), dtype=int)
    # Convert to (time, freq) and sort by time for efficient hashing
    peaks = np.stack([coords[:, 1], coords[:, 0]], axis=1)
    order = np.argsort(peaks[:, 0])
    return peaks[order]


def _pack_hash(f1: int, f2: int, dt: int) -> int:
    """Pack (f1, f2, dt) into a 32-bit int. Uses 11 bits each for f1/f2 and 10 bits for dt."""
    return ((f1 & 0x7FF) << 21) | ((f2 & 0x7FF) << 10) | (dt & 0x3FF)


def _generate_hashes(peaks: np.ndarray, params: Dict[str, Any], include_peak_index: bool = False) -> List[Tuple[int, int] | Tuple[int, int, int]]:
    """Produce list of (hash_int, anchor_time_bin[, anchor_peak_index])."""
    if peaks.shape[0] == 0:
        return []
    time_res = params["hop_length"] / params["fs"]
    dt_min = max(1, int(round(params["target_dt_min"] / time_res)))
    dt_max = max(dt_min + 1, int(round(params["target_dt_max"] / time_res)))
    hashes: List[Tuple[int, int] | Tuple[int, int, int]] = []
    n = len(peaks)
    for i in range(n):
        t1, f1 = int(peaks[i, 0]), int(peaks[i, 1])
        fan_added = 0
        j = i + 1
        while j < n:
            dt = int(peaks[j, 0]) - t1
            if dt > dt_max or fan_added >= params["max_hashes_per_anchor"]:
                break
            if dt >= dt_min:
                f2 = int(peaks[j, 1])
                if include_peak_index:
                    hashes.append((_pack_hash(f1, f2, dt), t1, i))
                else:
                    hashes.append((_pack_hash(f1, f2, dt), t1))
                fan_added += 1
            j += 1
    return hashes


def _iter_music_db(music_db: Any) -> Iterable[Tuple[str, str, np.ndarray]]:
    """
    Yield (title, genre, signal) triplets from either:
    - numpy structured array loaded from MATLAB (musicDB.mat)
    - list of dicts with keys 'title','genre','signal'
    """
    if isinstance(music_db, np.ndarray) and music_db.dtype.names:
        flat = music_db.reshape(-1)
        for entry in flat:
            title = str(entry["title"][0]) if entry["title"].size else ""
            genre = str(entry["genre"][0]) if entry["genre"].size else ""
            signal_arr = np.asarray(entry["signal"]).flatten()
            yield title, genre, signal_arr
    elif isinstance(music_db, Sequence):
        for entry in music_db:
            title = str(entry.get("title", ""))
            genre = str(entry.get("genre", ""))
            signal_arr = np.asarray(entry.get("signal", []))
            yield title, genre, signal_arr
    else:
        raise TypeError("Unsupported music_db format. Provide MATLAB struct array or list of dicts.")


# ----------------------------
# Public API
# ----------------------------
def compute_fingerprints(music_db: Any, params: Optional[Dict[str, Any]] = None) -> FingerprintDB:
    """
    Build fingerprints for every song in music_db.

    Args:
        music_db: MATLAB struct array from Project6_musicDB.mat or list of dicts.
        params: override defaults from _default_params().
    Returns:
        FingerprintDB containing inverted hash index and per-song metadata.
    """
    p = _default_params()
    if params:
        p.update(params)

    hash_index: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    songs: List[Dict[str, Any]] = []

    for song_idx, (title, genre, sig) in enumerate(_iter_music_db(music_db)):
        t0 = time.time()
        sig_proc = _preprocess_signal(sig)
        _, _, spec_db = _compute_spectrogram(
            sig_proc, p["fs"], p["n_fft"], p["hop_length"], whiten=p.get("whiten", True)
        )
        peaks = _pick_peaks(spec_db, p)
        hashes = _generate_hashes(peaks, p)
        for h, anchor_t in hashes:
            hash_index[h].append((song_idx, anchor_t))
        duration = len(sig) / float(p["fs"])
        songs.append(
            {
                "title": title,
                "genre": genre,
                "duration_sec": duration,
                "num_peaks": int(peaks.shape[0]),
                "num_hashes": int(len(hashes)),
                "build_time_sec": time.time() - t0,
            }
        )

        if peaks.shape[0] < p["min_peaks_per_song"]:
            # Drop threshold adaptively for sparse songs
            relaxed = dict(p)
            relaxed["peak_threshold_rel"] = max(10.0, p["peak_threshold_rel"] - 6.0)
            relaxed["peak_percentile"] = max(50.0, p["peak_percentile"] - 10.0)
            peaks_relaxed = _pick_peaks(spec_db, relaxed)
            hashes_relaxed = _generate_hashes(peaks_relaxed, relaxed)
            for h, anchor_t in hashes_relaxed:
                hash_index[h].append((song_idx, anchor_t))
            songs[-1]["num_peaks_relaxed"] = int(peaks_relaxed.shape[0])
            songs[-1]["num_hashes_relaxed"] = int(len(hashes_relaxed))

    return FingerprintDB(params=p, songs=songs, hash_index=dict(hash_index))


def identify_song(
    clip: np.ndarray,
    fingerprints: FingerprintDB | Dict[str, Any],
    fs: int = 16000,
    return_info: bool = False,
) -> Any:
    """
    Identify the song index for a 3-second clip using a voting scheme on hash time-offsets.

    Args:
        clip: mono waveform (expected 48000 samples at 16 kHz).
        fingerprints: FingerprintDB or compatible dict with keys params/hash_index/songs.
        fs: sampling rate of the clip (defaults to 16 kHz).
        return_info: when True, also returns debug metadata.
    Returns:
        song_id (int, 0-based). If return_info is True, returns (song_id, info_dict).
        Returns -1 if no match is found.
    """
    fp_params = fingerprints.params if hasattr(fingerprints, "params") else fingerprints["params"]
    hash_index = fingerprints.hash_index if hasattr(fingerprints, "hash_index") else fingerprints["hash_index"]
    songs = fingerprints.songs if hasattr(fingerprints, "songs") else fingerprints.get("songs", [])

    if fs != fp_params["fs"]:
        # Resample if needed
        num = int(round(len(clip) * fp_params["fs"] / float(fs)))
        clip = signal.resample(clip, num)

    sig_proc = _preprocess_signal(clip)
    freqs, _, spec_db = _compute_spectrogram(
        sig_proc,
        fp_params["fs"],
        fp_params["n_fft"],
        fp_params["hop_length"],
        whiten=fp_params.get("whiten", True),
    )
    peaks = _pick_peaks(spec_db, fp_params)
    hashes = _generate_hashes(peaks, fp_params, include_peak_index=True)
    query_duration = len(clip) / float(fp_params["fs"])

    votes = Counter()
    song_vote_totals = Counter()
    match_records = []
    for item in hashes:
        if len(item) == 3:
            h, t_query, peak_idx = item  # type: ignore[misc]
        else:
            h, t_query = item  # type: ignore[misc]
            peak_idx = None
        if h not in hash_index:
            continue
        for song_id, t_song in hash_index[h]:
            offset = t_song - t_query
            votes[(song_id, offset)] += 1
            song_vote_totals[song_id] += 1
            if peak_idx is not None:
                match_records.append((song_id, offset, peak_idx))

    if not votes:
        return (-1, {}) if return_info else -1

    (best_song, best_offset), best_votes = votes.most_common(1)[0]
    total_for_song = song_vote_totals[best_song]
    # Confidence: ratio of best offset votes to total votes for that song
    confidence = best_votes / float(total_for_song + 1e-9)

    info = {
        "best_offset": int(best_offset),
        "best_votes": int(best_votes),
        "total_votes_for_song": int(total_for_song),
        "confidence": float(confidence),
        "title": songs[best_song]["title"] if best_song < len(songs) else "",
        "genre": songs[best_song]["genre"] if best_song < len(songs) else "",
        "votes_by_song": {int(k): int(v) for k, v in song_vote_totals.items()},
        "query_duration_sec": float(query_duration),
        "query_num_peaks": int(peaks.shape[0]),
        "query_num_hashes": int(len(hashes)),
        "matched_hashes_total": int(len(match_records)),
    }

    top_offsets = votes.most_common(8)
    info["top_matches"] = [
        {
            "song_id": int(song_id),
            "offset": int(offset),
            "votes": int(count),
            "title": songs[song_id]["title"] if song_id < len(songs) else "",
            "is_best": bool(song_id == best_song and offset == best_offset),
        }
        for (song_id, offset), count in top_offsets
    ]

    # Collect matched query peaks to visualize contributing regions
    matched_peaks = []
    all_matched_peaks = []
    if match_records and peaks.shape[0] > 0:
        time_res = fp_params["hop_length"] / fp_params["fs"]
        freq_axis = np.linspace(0, fp_params["fs"] / 2.0, spec_db.shape[0])
        for song_id, offset, peak_idx in match_records:
            if 0 <= peak_idx < peaks.shape[0]:
                t_bin, f_bin = peaks[peak_idx]
                entry = {
                    "time_sec": float(t_bin * time_res),
                    "freq_hz": float(freq_axis[int(f_bin)] if f_bin < len(freq_axis) else 0.0),
                    "t_bin": int(t_bin),
                    "f_bin": int(f_bin),
                    "song_id": int(song_id),
                    "offset": int(offset),
                }
                entry["is_best"] = bool(song_id == best_song and offset == best_offset)
                all_matched_peaks.append(entry)
                if entry["is_best"]:
                    matched_peaks.append({k: entry[k] for k in ("time_sec", "freq_hz", "t_bin", "f_bin", "song_id", "offset")})
        info["matched_query_peaks"] = matched_peaks
        info["all_matched_peaks"] = all_matched_peaks
        best_count = len(matched_peaks)
        info["matched_peak_fraction"] = float(best_count / max(1, peaks.shape[0]))
        info["matched_peak_count"] = int(best_count)

    return (best_song, info) if return_info else best_song


# ----------------------------
# Convenience loader
# ----------------------------
def load_music_db(mat_path: str) -> np.ndarray:
    """Load the provided Project6_musicDB.mat file."""
    import scipy.io as sio

    data = sio.loadmat(mat_path)
    if "musicDB" not in data:
        raise KeyError(f"musicDB not found in {mat_path}")
    return data["musicDB"]


# ----------------------------
# Chroma-based (pitch-robust) fingerprint - variant B
# ----------------------------
def _default_chroma_params() -> Dict[str, Any]:
    p = _default_params()
    p.update(
        {
            "mode": "chroma",
            "n_fft": 2048,
            "hop_length": 512,
            "chroma_top_k": 2,
            "chroma_percentile": 78.0,
            "target_dt_min": 0.08,
            "target_dt_max": 0.8,
            "max_hashes_per_anchor": 10,
            "peak_neighborhood_time": 9,
            "whiten": False,
        }
    )
    return p


def _pack_chroma_hash(c1: int, c2: int, dt: int) -> int:
    """Pack (chroma1, chroma2, dt) into <=20 bits."""
    return ((c1 & 0xF) << 14) | ((c2 & 0xF) << 10) | (dt & 0x3FF)


def _compute_chroma(x: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    import librosa

    chroma = librosa.feature.chroma_stft(
        y=x,
        sr=params["fs"],
        n_fft=params["n_fft"],
        hop_length=params["hop_length"],
    )
    key_idx = int(np.argmax(np.mean(chroma, axis=1)))
    chroma_rot = np.roll(chroma, -key_idx, axis=0)
    chroma_db = 20 * np.log10(chroma_rot + 1e-6)
    return chroma_db, key_idx


def _pick_chroma_peaks(chroma_db: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    thresh = np.percentile(chroma_db, params["chroma_percentile"])
    peaks = []
    T = chroma_db.shape[1]
    top_k = params["chroma_top_k"]
    for t in range(T):
        frame = chroma_db[:, t]
        if frame.max() < thresh:
            continue
        idx = np.argpartition(frame, -top_k)[-top_k:]
        for c in idx:
            if frame[c] >= thresh:
                peaks.append((t, int(c)))
    if not peaks:
        return np.empty((0, 2), dtype=int)
    peaks = np.array(peaks, dtype=int)
    order = np.argsort(peaks[:, 0])
    return peaks[order]


def _generate_chroma_hashes(peaks: np.ndarray, params: Dict[str, Any], include_peak_index: bool = False):
    if peaks.shape[0] == 0:
        return []
    time_res = params["hop_length"] / params["fs"]
    dt_min = max(1, int(round(params["target_dt_min"] / time_res)))
    dt_max = max(dt_min + 1, int(round(params["target_dt_max"] / time_res)))
    hashes = []
    n = len(peaks)
    for i in range(n):
        t1, c1 = int(peaks[i, 0]), int(peaks[i, 1])
        fan_added = 0
        j = i + 1
        while j < n:
            dt = int(peaks[j, 0]) - t1
            if dt > dt_max or fan_added >= params["max_hashes_per_anchor"]:
                break
            if dt >= dt_min:
                c2 = int(peaks[j, 1])
                if include_peak_index:
                    hashes.append((_pack_chroma_hash(c1, c2, dt), t1, i))
                else:
                    hashes.append((_pack_chroma_hash(c1, c2, dt), t1))
                fan_added += 1
            j += 1
    return hashes


def compute_fingerprints_chroma(music_db: Any, params: Optional[Dict[str, Any]] = None) -> FingerprintDB:
    p = _default_chroma_params()
    if params:
        p.update(params)
    hash_index: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    songs: List[Dict[str, Any]] = []

    for song_idx, (title, genre, sig) in enumerate(_iter_music_db(music_db)):
        t0 = time.time()
        sig_proc = _preprocess_signal(sig)
        chroma_db, key_idx = _compute_chroma(sig_proc, p)
        peaks = _pick_chroma_peaks(chroma_db, p)
        hashes = _generate_chroma_hashes(peaks, p)
        for h, anchor_t in hashes:
            hash_index[h].append((song_idx, anchor_t))
        duration = len(sig) / float(p["fs"])
        songs.append(
            {
                "title": title,
                "genre": genre,
                "duration_sec": duration,
                "num_peaks": int(peaks.shape[0]),
                "num_hashes": int(len(hashes)),
                "key_shift": int(key_idx),
                "build_time_sec": time.time() - t0,
            }
        )
    p["mode"] = "chroma"
    return FingerprintDB(params=p, songs=songs, hash_index=dict(hash_index))


def identify_song_chroma(
    clip: np.ndarray,
    fingerprints: FingerprintDB | Dict[str, Any],
    fs: int = 16000,
    return_info: bool = False,
):
    fp_params = fingerprints.params if hasattr(fingerprints, "params") else fingerprints["params"]
    hash_index = fingerprints.hash_index if hasattr(fingerprints, "hash_index") else fingerprints["hash_index"]
    songs = fingerprints.songs if hasattr(fingerprints, "songs") else fingerprints.get("songs", [])

    if fs != fp_params["fs"]:
        num = int(round(len(clip) * fp_params["fs"] / float(fs)))
        clip = signal.resample(clip, num)

    sig_proc = _preprocess_signal(clip)
    chroma_db, key_idx = _compute_chroma(sig_proc, fp_params)
    peaks = _pick_chroma_peaks(chroma_db, fp_params)
    hashes = _generate_chroma_hashes(peaks, fp_params, include_peak_index=True)
    query_duration = len(clip) / float(fp_params["fs"])

    votes = Counter()
    song_vote_totals = Counter()
    match_records = []
    for item in hashes:
        if len(item) == 3:
            h, t_query, peak_idx = item  # type: ignore[misc]
        else:
            h, t_query = item  # type: ignore[misc]
            peak_idx = None
        if h not in hash_index:
            continue
        for song_id, t_song in hash_index[h]:
            offset = t_song - t_query
            votes[(song_id, offset)] += 1
            song_vote_totals[song_id] += 1
            if peak_idx is not None:
                match_records.append((song_id, offset, peak_idx))

    if not votes:
        return (-1, {}) if return_info else -1

    (best_song, best_offset), best_votes = votes.most_common(1)[0]
    total_for_song = song_vote_totals[best_song]
    confidence = best_votes / float(total_for_song + 1e-9)

    info = {
        "best_offset": int(best_offset),
        "best_votes": int(best_votes),
        "total_votes_for_song": int(total_for_song),
        "confidence": float(confidence),
        "title": songs[best_song]["title"] if best_song < len(songs) else "",
        "genre": songs[best_song]["genre"] if best_song < len(songs) else "",
        "votes_by_song": {int(k): int(v) for k, v in song_vote_totals.items()},
        "key_shift": int(key_idx),
        "query_duration_sec": float(query_duration),
        "query_num_peaks": int(peaks.shape[0]),
        "query_num_hashes": int(len(hashes)),
        "matched_hashes_total": int(len(match_records)),
    }

    top_offsets = votes.most_common(8)
    info["top_matches"] = [
        {
            "song_id": int(song_id),
            "offset": int(offset),
            "votes": int(count),
            "title": songs[song_id]["title"] if song_id < len(songs) else "",
            "is_best": bool(song_id == best_song and offset == best_offset),
        }
        for (song_id, offset), count in top_offsets
    ]
    matched_peaks = []
    all_matched_peaks = []
    if match_records and peaks.shape[0] > 0:
        time_res = fp_params["hop_length"] / fp_params["fs"]
        for song_id, offset, peak_idx in match_records:
            if 0 <= peak_idx < peaks.shape[0]:
                t_bin, c_bin = peaks[peak_idx]
                entry = {
                    "time_sec": float(t_bin * time_res),
                    "chroma_bin": int(c_bin),
                    "song_id": int(song_id),
                    "offset": int(offset),
                }
                entry["is_best"] = bool(song_id == best_song and offset == best_offset)
                all_matched_peaks.append(entry)
                if entry["is_best"]:
                    matched_peaks.append({k: entry[k] for k in ("time_sec", "chroma_bin", "song_id", "offset")})
        info["matched_query_peaks"] = matched_peaks
        info["all_matched_peaks"] = all_matched_peaks
        best_count = len(matched_peaks)
        info["matched_peak_fraction"] = float(best_count / max(1, peaks.shape[0]))
        info["matched_peak_count"] = int(best_count)

    return (best_song, info) if return_info else best_song


def identify_song_multi_tempo(
    clip: np.ndarray,
    fingerprints: FingerprintDB | Dict[str, Any],
    fs: int = 16000,
    tempo_factors: Sequence[float] = (0.9, 1.0, 1.1),
    identify_fn=identify_song,
    return_info: bool = False,
):
    """
    Tempo-robust helper: try multiple tempo factors and pick the highest-confidence match.
    Useful for speed-altered clips (bonus 2).
    """
    best = None
    best_conf = -1.0
    for factor in tempo_factors:
        if abs(factor - 1.0) < 1e-6:
            stretched = clip
        else:
            new_len = max(1, int(round(len(clip) / factor)))
            stretched = signal.resample(clip, new_len)
        song_id, info = identify_fn(stretched, fingerprints, fs=fs, return_info=True)
        conf = info.get("confidence", 0.0) if isinstance(info, dict) else 0.0
        if conf > best_conf:
            best_conf = conf
            best = (song_id, info, factor)
    if best is None:
        return (-1, {}) if return_info else -1
    song_id, info, factor = best
    if return_info:
        if isinstance(info, dict):
            info["tempo_factor"] = factor
        return song_id, info
    return song_id
