"""
Interactive bonus UI for the Project 6 Shazam-style fingerprint system.

Run inside the asp conda environment from this directory:
    conda run -n asp streamlit run ui_demo.py
"""
from __future__ import annotations

import io
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st
from scipy import signal as sp_signal

from shazam_system import (
    FingerprintDB,
    compute_fingerprints,
    compute_fingerprints_chroma,
    identify_song,
    identify_song_chroma,
    load_music_db,
    _compute_chroma,
    _compute_spectrogram,
    _iter_music_db,
    _pick_chroma_peaks,
    _pick_peaks,
    _preprocess_signal,
)


DEFAULT_DB_PATH = Path(__file__).with_name("Project6_musicDB.mat")
CACHE_DIR = Path.cwd() / "fingerprint_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _ensure_mono(x: np.ndarray) -> np.ndarray:
    """Collapse multi-channel audio to mono float array."""
    if x.ndim == 1:
        return x.astype(float)
    return np.mean(x, axis=1).astype(float)


@st.cache_resource(show_spinner=False)
def load_music_library(mat_path: str) -> List[Dict[str, Any]]:
    """Load MATLAB music DB and expose it as a Python-friendly song list."""
    raw = load_music_db(mat_path)
    songs: List[Dict[str, Any]] = []
    for idx, (title, genre, sig) in enumerate(_iter_music_db(raw)):
        signal_arr = np.asarray(sig, dtype=float)
        songs.append(
            {
                "id": idx,
                "title": title or f"Track {idx:02d}",
                "genre": genre or "Unknown",
                "signal": signal_arr,
                "duration_sec": signal_arr.size / 16000.0,
            }
        )
    return songs


@st.cache_resource(show_spinner=False)
def build_fingerprints_cached(mat_path: str, mode: str) -> FingerprintDB:
    """Build fingerprints while letting Streamlit cache deduplicate repeated work."""
    songs = load_music_library(mat_path)
    if mode == "Chroma (pitch-robust)":
        return compute_fingerprints_chroma(songs)
    return compute_fingerprints(songs)


def summarize_fingerprints(fp: FingerprintDB) -> Dict[str, Any]:
    """Compute aggregate stats shown in the fingerprint status block."""
    num_songs = len(fp.songs)
    total_hashes = sum(song.get("num_hashes", 0) for song in fp.songs)
    total_peaks = sum(song.get("num_peaks", 0) for song in fp.songs)
    total_time = sum(song.get("build_time_sec", 0.0) for song in fp.songs)
    hash_table_entries = sum(len(v) for v in fp.hash_index.values())
    return {
        "num_songs": num_songs,
        "total_hashes": int(total_hashes),
        "total_peaks": int(total_peaks),
        "avg_hashes_per_song": float(total_hashes / max(1, num_songs)),
        "avg_peaks_per_song": float(total_peaks / max(1, num_songs)),
        "total_build_time": float(total_time),
        "hash_index_size": int(hash_table_entries),
    }


def _default_cache_path(mode: str) -> Path:
    """Return the on-disk pickle location for the selected fingerprint mode."""
    if "Chroma" in mode:
        return CACHE_DIR / "fingerprints_chroma.pkl"
    return CACHE_DIR / "fingerprints_constellation.pkl"


def load_or_build_fingerprints(
    mat_path: str, mode: str, cache_path: Path, persist: bool
) -> Tuple[FingerprintDB, Dict[str, Any]]:
    """Load fingerprints from cache if present, otherwise build and optionally persist."""
    if cache_path.exists():
        with cache_path.open("rb") as fh:
            fp = pickle.load(fh)
        return fp, summarize_fingerprints(fp)

    fp = build_fingerprints_cached(mat_path, mode)
    if persist:
        with cache_path.open("wb") as fh:
            pickle.dump(fp, fh)
    return fp, summarize_fingerprints(fp)


def slice_signal(signal_arr: np.ndarray, sr: int, start_sec: float, duration: float) -> np.ndarray:
    """Extract a subclip (in seconds) from a waveform at sampling rate sr."""
    start = max(0.0, start_sec)
    end = min(signal_arr.size / sr, start + duration)
    idx0 = int(round(start * sr))
    idx1 = int(round(end * sr))
    idx1 = max(idx0 + 1, idx1)
    return signal_arr[idx0:idx1]


def prepare_clip(signal_arr: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    """Resample a signal to the fingerprint model sample rate."""
    if src_sr == target_sr:
        return signal_arr.astype(float)
    num = int(round(len(signal_arr) * target_sr / float(src_sr)))
    return sp_signal.resample(signal_arr, num).astype(float)


def analyze_query(clip: np.ndarray, params: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Recompute STFT/chroma + peaks for the clip so the UI can visualize them."""
    clip_proc = _preprocess_signal(clip)
    hop = params["hop_length"]
    fs = params["fs"]
    time_res = hop / fs
    if "Chroma" in mode:
        chroma_db, _ = _compute_chroma(clip_proc, params)
        peaks = _pick_chroma_peaks(chroma_db, params)
        times = np.arange(chroma_db.shape[1]) * time_res
        peak_records = [
            {"time_sec": float(t * time_res), "chroma_bin": int(c)} for t, c in peaks
        ]
        return {
            "spec_kind": "chroma",
            "spec": chroma_db,
            "times": times,
            "freqs": np.arange(chroma_db.shape[0]),
            "all_peaks": peak_records,
            "hop_length": hop,
            "fs": fs,
        }

    freqs, times, spec_db = _compute_spectrogram(
        clip_proc,
        fs,
        params["n_fft"],
        hop,
        whiten=params.get("whiten", True),
    )
    peaks = _pick_peaks(spec_db, params)
    freq_bins = np.array(freqs)
    peak_records = []
    for t_bin, f_bin in peaks:
        freq_val = float(freq_bins[int(f_bin)]) if f_bin < len(freq_bins) else 0.0
        peak_records.append(
            {
                "time_sec": float(t_bin * time_res),
                "freq_hz": freq_val,
            }
        )
    return {
        "spec_kind": "spectrogram",
        "spec": spec_db,
        "times": times,
        "freqs": freq_bins,
        "all_peaks": peak_records,
        "hop_length": hop,
        "fs": fs,
    }


def make_waveform_preview(signal_arr: np.ndarray, sr: int, selection: Tuple[float, float]) -> go.Figure:
    """Render a lightweight waveform with a shaded region showing the selected clip."""
    duration = signal_arr.size / sr
    step = max(1, int(len(signal_arr) / 5000))
    samples = signal_arr[::step]
    times = np.linspace(0, duration, samples.size)
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(x=times, y=samples, mode="lines", line=dict(color="#7f8c8d"), name="waveform")
    )
    fig.add_shape(
        type="rect",
        x0=selection[0],
        x1=selection[1],
        y0=min(samples) * 1.1,
        y1=max(samples) * 1.1,
        fillcolor="rgba(231, 76, 60, 0.15)",
        line=dict(color="rgba(231, 76, 60, 0.7)", width=2),
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=140,
        showlegend=False,
        xaxis_title="sec",
        yaxis_title="amp",
    )
    return fig


def _build_scatter_trace(points: Sequence[Dict[str, Any]], *, is_matched: bool, mode: str) -> go.Scattergl:
    """Create an overlay trace for the spectrogram plot highlighting peaks."""
    if not points:
        return go.Scattergl()
    if mode == "chroma":
        y_vals = [p.get("chroma_bin", 0) for p in points]
        hover_y = [f"chroma {int(y)}" for y in y_vals]
    else:
        y_vals = [p.get("freq_hz", 0.0) for p in points]
        hover_y = [f"{val:.0f} Hz" for val in y_vals]
    colors = "#f39c12" if is_matched else "rgba(255,255,255,0.5)"
    sizes = 10 if is_matched else 4
    name = "Matched peaks" if is_matched else "All peaks"
    hover_text = [f"t={p['time_sec']:.2f}s | {y}" for p, y in zip(points, hover_y)]
    return go.Scattergl(
        x=[p.get("time_sec", 0.0) for p in points],
        y=y_vals,
        mode="markers",
        marker=dict(color=colors, size=sizes, line=dict(width=0)),
        name=name,
        hovertext=hover_text,
        hoverinfo="text",
    )


def build_spectrogram_figure(
    analysis: Dict[str, Any],
    matched_peaks: Sequence[Dict[str, Any]],
    *,
    show_all_peaks: bool,
) -> go.Figure:
    """Combine spectrogram/chroma heatmap with peak overlays for the main view."""
    spec = analysis["spec"]
    times = analysis["times"]
    freqs = analysis["freqs"]
    y_axis = np.maximum(freqs, 1.0) if analysis["spec_kind"] == "spectrogram" else freqs
    fig = go.Figure()
    heatmap = go.Heatmap(
        x=times,
        y=y_axis,
        z=spec,
        colorscale="Magma",
        colorbar=dict(title="dB" if analysis["spec_kind"] == "spectrogram" else "dB"),
    )
    fig.add_trace(heatmap)
    if show_all_peaks and analysis["all_peaks"]:
        fig.add_trace(_build_scatter_trace(analysis["all_peaks"], is_matched=False, mode=analysis["spec_kind"]))
    if matched_peaks:
        fig.add_trace(_build_scatter_trace(matched_peaks, is_matched=True, mode=analysis["spec_kind"]))

    yaxis_title = "Frequency (Hz)" if analysis["spec_kind"] == "spectrogram" else "Chroma bin"
    fig.update_layout(
        height=410,
        margin=dict(l=20, r=20, t=40, b=30),
        xaxis_title="Time (s)",
        yaxis_title=yaxis_title,
    )
    if analysis["spec_kind"] == "spectrogram":
        fig.update_yaxes(type="log", minexponent=0)
    else:
        fig.update_yaxes(tickmode="array", tickvals=list(range(12)))
    return fig


def build_offset_chart(top_matches: Sequence[Dict[str, Any]], hop_length: int, fs: int) -> go.Figure:
    """Visualize votes for the strongest (song, offset) hypotheses."""
    fig = go.Figure()
    if not top_matches:
        fig.update_layout(height=180, xaxis_title="Offset (s)", yaxis_title="Votes")
        return fig
    offsets = [item["offset"] * hop_length / fs for item in top_matches]
    votes = [item["votes"] for item in top_matches]
    labels = [item.get("title", f"Song {item['song_id']}") for item in top_matches]
    colors = ["#e74c3c" if item.get("is_best") else "#7f8c8d" for item in top_matches]
    fig.add_bar(
        x=offsets,
        y=votes,
        marker_color=colors,
        text=labels,
        hovertemplate="Offset %{x:.2f}s<br>Votes %{y}<br>%{text}<extra></extra>",
    )
    fig.update_layout(height=220, xaxis_title="Offset (s)", yaxis_title="Votes", margin=dict(l=10, r=10, t=20, b=10))
    return fig


def format_song_display(song_meta: Dict[str, Any]) -> str:
    """Readable label for dropdowns when selecting DB songs."""
    return f"#{song_meta['id']:02d} · {song_meta['title']} ({song_meta['genre']})"


def _normalize_audio(sig: np.ndarray) -> np.ndarray:
    max_abs = np.max(np.abs(sig)) if sig.size else 1.0
    if max_abs <= 0:
        return sig.astype(np.float32)
    return (sig / max_abs).astype(np.float32)


def _to_audio_bytes(sig: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, _normalize_audio(sig), sr, format="WAV")
    buf.seek(0)
    return buf.read()


def apply_augmentations(
    signal_arr: np.ndarray,
    sr: int,
    *,
    add_noise: bool,
    noise_snr_db: float,
    noise_seed: int,
    pitch_shift: float,
    tempo_factor: float,
) -> Tuple[np.ndarray, List[str]]:
    """Apply optional noise/pitch/time augmentations for demo and debugging."""
    y = np.asarray(signal_arr, dtype=float)
    summary = []
    if add_noise:
        sig_rms = np.sqrt(np.mean(y**2) + 1e-12)
        target_snr = max(1e-3, 10 ** (noise_snr_db / 20.0))
        noise_rms = sig_rms / target_snr
        rng = np.random.default_rng(int(noise_seed))
        noise = rng.standard_normal(y.shape) * noise_rms
        y = y + noise
        summary.append(f"Noise SNR≈{noise_snr_db:.1f} dB")
    if abs(pitch_shift) > 1e-3:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)
        summary.append(f"Pitch shift {pitch_shift:+.1f} semitones")
    if abs(tempo_factor - 1.0) > 1e-3:
        tempo_factor = max(0.5, min(2.0, tempo_factor))
        y = librosa.effects.time_stretch(y, rate=tempo_factor)
        summary.append(f"Tempo ×{tempo_factor:.2f}")
    return y.astype(float), summary


def main() -> None:
    """Launch the full Streamlit workflow for the fingerprint visualization demo."""
    st.set_page_config(page_title="Project6 Fingerprint UI", layout="wide")
    st.title("Shazam-style Fingerprint Visualizer")
    st.caption("Load database → pick a clip → run identification → see which fingerprints drive the decision.")

    if "state" not in st.session_state:
        st.session_state.state = {}

    state = st.session_state.state
    fp_record = state.get("fp_record")
    if fp_record is None:
        default_mode = "Baseline (constellation)"
        mat_path = str(DEFAULT_DB_PATH)
        cache_path = _default_cache_path(default_mode)
        with st.spinner("Preparing default fingerprint DB..."):
            fp, summary = load_or_build_fingerprints(mat_path, default_mode, cache_path, persist=True)
            songs = load_music_library(mat_path)
        state["fp_record"] = {
            "fp": fp,
            "summary": summary,
            "mode": default_mode,
            "mat_path": mat_path,
            "songs": songs,
        }
        fp_record = state["fp_record"]
    if "last_result" not in state and fp_record.get("songs"):
        default_song = fp_record["songs"][0]
        clip = slice_signal(default_song["signal"], 16000, 0.0, 3.0)
        clip_prepared = prepare_clip(clip, 16000, fp_record["fp"].params["fs"])
        identify_fn = identify_song_chroma if fp_record["mode"].startswith("Chroma") else identify_song
        song_id, info = identify_fn(clip_prepared, fp_record["fp"], fs=fp_record["fp"].params["fs"], return_info=True)
        analysis = analyze_query(clip_prepared, fp_record["fp"].params, fp_record["mode"])
        state["last_result"] = {
            "song_id": song_id,
            "info": info,
            "analysis": analysis,
            "clip_label": f"musicDB · {default_song['title']}",
            "clip_sr": 16000,
            "clip_duration": clip_prepared.size / fp_record["fp"].params["fs"],
            "fp_params": fp_record["fp"].params,
            "augment_summary": [],
            "augment_cfg": {
                "add_noise": False,
                "noise_snr_db": 25.0,
                "noise_seed": 0,
                "pitch_shift": 0.0,
                "tempo_factor": 1.0,
            },
        }
        state["clip_defaults"] = {
            "source": "musicDB random",
            "song_idx": 0,
            "clip_len": 3.0,
            "start_time": 0.0,
        }

    controls_col, status_col = st.columns([1.35, 1.0], gap="large")

    with controls_col:
        st.subheader("1. Fingerprint database")
        mode_options = ["Baseline (constellation)", "Chroma (pitch-robust)"]
        stored_mode = fp_record.get("mode", mode_options[0])
        mode_index = mode_options.index(stored_mode) if stored_mode in mode_options else 0
        mode = st.selectbox("Fingerprint design", mode_options, index=mode_index)
        mat_path = st.text_input("Project6_musicDB.mat path", fp_record.get("mat_path", str(DEFAULT_DB_PATH)))
        cache_path_default = _default_cache_path(mode)
        cache_path_str = st.text_input("Fingerprint cache (.pkl)", str(cache_path_default))
        persist_cache = st.checkbox("Persist cache after build", value=True)
        need_build = fp_record is None or fp_record.get("mode") != mode or fp_record.get("mat_path") != mat_path
        if st.button("Load / build fingerprint DB", disabled=not Path(mat_path).exists()):
            with st.spinner("Loading fingerprint DB..."):
                fp, summary = load_or_build_fingerprints(mat_path, mode, Path(cache_path_str), persist_cache)
                songs = load_music_library(mat_path)
            st.session_state.state["fp_record"] = {
                "fp": fp,
                "summary": summary,
                "mode": mode,
                "mat_path": mat_path,
                "songs": songs,
            }
            need_build = False
            st.success(f"Fingerprint DB ready: {summary['num_songs']} songs")
            fp_record = st.session_state.state.get("fp_record")

        if not fp_record or need_build:
            st.info("Load the fingerprint DB to unlock the rest of the UI.")

        st.divider()
        st.subheader("2. Choose query clip")
        clip_defaults = state.setdefault(
            "clip_defaults",
            {"source": "musicDB random", "song_idx": 0, "clip_len": 3.0, "start_time": 0.0},
        )
        source_options = ["musicDB random", "Upload local audio"]
        source_index = source_options.index(clip_defaults.get("source", source_options[0]))
        clip_source = st.radio("Clip source", source_options, index=source_index)
        clip_signal: Optional[np.ndarray] = None
        clip_sr = 16000
        clip_label = ""
        selection_range = (0.0, 3.0)
        current_song_idx = clip_defaults.get("song_idx", 0)

        if clip_source == "musicDB random":
            if not fp_record or not fp_record.get("songs"):
                st.warning("Load the music DB before sampling clips.")
            else:
                songs = fp_record["songs"]
                song_idx = st.selectbox(
                    "Select song",
                    options=list(range(len(songs))),
                    format_func=lambda idx: format_song_display(songs[idx]),
                    index=min(clip_defaults.get("song_idx", 0), len(songs) - 1),
                )
                current_song_idx = song_idx
                song_meta = songs[song_idx]
                duration = song_meta["signal"].size / clip_sr
                clip_len_max = max(1.0, min(8.0, duration))
                clip_len_value = min(clip_defaults.get("clip_len", 3.0), clip_len_max)
                clip_len = st.slider("Clip length (s)", 1.0, clip_len_max, clip_len_value, step=0.5)
                max_start = max(0.0, duration - clip_len)
                start_key = "musicdb_start"
                if start_key not in st.session_state:
                    st.session_state[start_key] = 0.0
                if st.button("Pick random 3 s"):
                    st.session_state[start_key] = float(random.uniform(0.0, max_start))
                start_val = min(float(st.session_state[start_key]), max_start)
                st.session_state[start_key] = start_val
                start_default = min(clip_defaults.get("start_time", start_val), max_start)
                start_time = st.slider("Start time (s)", 0.0, float(max_start), start_default, key=start_key)
                clip_signal = slice_signal(song_meta["signal"], clip_sr, start_time, clip_len)
                clip_label = f"musicDB · {song_meta['title']}"
                selection_range = (start_time, start_time + clip_len)
                st.plotly_chart(
                    make_waveform_preview(song_meta["signal"], clip_sr, selection_range), width="stretch"
                )
                st.caption(f"Clip length {clip_len:.2f}s · starts at {start_time:.2f}s / {duration:.1f}s total")
        else:
            uploaded = st.file_uploader("Upload audio (.wav/.mp3/.flac)", type=["wav", "mp3", "flac", "ogg", "m4a"])
            if uploaded:
                data, sr = sf.read(io.BytesIO(uploaded.read()))
                data = _ensure_mono(np.asarray(data))
                duration = data.size / sr
                clip_len_max = max(1.0, min(12.0, duration))
                clip_len_value = min(clip_defaults.get("clip_len", 3.0), clip_len_max)
                clip_len = st.slider("Clip length (s)", 1.0, clip_len_max, clip_len_value, step=0.5)
                max_start = max(0.0, duration - clip_len)
                start_default = min(clip_defaults.get("start_time", 0.0), max_start)
                start_time = st.slider("Start time (s)", 0.0, float(max_start), start_default, key="upload_start")
                clip_signal = slice_signal(data, sr, start_time, clip_len)
                clip_sr = sr
                clip_label = f"Upload · {uploaded.name}"
                selection_range = (start_time, start_time + clip_len)
                st.plotly_chart(make_waveform_preview(data, sr, selection_range), width="stretch")
                st.caption(f"Original sample rate {sr} Hz · clip {clip_len:.2f}s")

        augment_cfg = {
            "add_noise": False,
            "noise_snr_db": 25.0,
            "noise_seed": 0,
            "pitch_shift": 0.0,
            "tempo_factor": 1.0,
        }
        processed_clip = clip_signal
        augment_summary: List[str] = []
        if clip_signal is not None:
            st.markdown("**Clip preview**")
            st.audio(_to_audio_bytes(clip_signal, clip_sr), sample_rate=clip_sr)
            with st.expander("Optional perturbations", expanded=False):
                augment_cfg["add_noise"] = st.checkbox("Add Gaussian noise", value=False)
                augment_cfg["noise_snr_db"] = st.slider(
                    "Target SNR (dB, lower = noisier)", 5.0, 60.0, 25.0
                )
                augment_cfg["noise_seed"] = int(st.number_input("Noise RNG seed", value=0, step=1))
                augment_cfg["pitch_shift"] = st.slider("Pitch shift (semitones)", -6.0, 6.0, 0.0, step=0.5)
                augment_cfg["tempo_factor"] = st.slider("Tempo factor (>1 = faster)", 0.75, 1.25, 1.0, step=0.01)
                st.caption("Use these knobs to simulate noisy, pitch-shifted, or tempo-warped queries for debugging/demos.")
            processed_clip, augment_summary = apply_augmentations(
                clip_signal,
                clip_sr,
                add_noise=augment_cfg["add_noise"],
                noise_snr_db=augment_cfg["noise_snr_db"],
                noise_seed=augment_cfg["noise_seed"],
                pitch_shift=augment_cfg["pitch_shift"],
                tempo_factor=augment_cfg["tempo_factor"],
            )
            if augment_summary:
                st.markdown("**Preview after augmentations**")
                st.audio(_to_audio_bytes(processed_clip, clip_sr), sample_rate=clip_sr)
                st.caption("; ".join(augment_summary))

        run_ready = processed_clip is not None and fp_record and not need_build
        st.divider()
        run_btn = st.button("Identify & visualize", width="stretch", disabled=not run_ready)

    with status_col:
        st.subheader("Dashboard")
        if fp_record and not need_build:
            stats = fp_record["summary"]
            st.metric("Songs", stats["num_songs"])
            st.metric("Total hashes", f"{stats['total_hashes']:,}")
            st.metric("Total peaks", f"{stats['total_peaks']:,}")
            st.caption(
                f"Hash entries {stats['hash_index_size']:,} · avg {stats['avg_hashes_per_song']:.0f} hashes/song · build time ~{stats['total_build_time']:.1f}s"
            )
        else:
            st.info("Fingerprint DB stats unavailable until you load/build.")
        st.divider()
        if state.get("last_result"):
            info_small = state["last_result"]["info"]
            st.subheader("Latest match")
            st.metric("Song", info_small.get("title", "N/A"))
            st.metric("Confidence", f"{info_small.get('confidence', 0.0):.2f}")
            st.metric("Votes", f"{info_small.get('best_votes',0)} / {info_small.get('total_votes_for_song',0)}")
        else:
            st.info("Run identification to populate summary metrics.")

    if run_btn and run_ready and processed_clip is not None and fp_record:
        fp = fp_record["fp"]
        params = fp.params
        target_sr = params["fs"]
        clip_prepared = prepare_clip(processed_clip, clip_sr, target_sr)
        identify_fn = identify_song_chroma if "Chroma" in mode else identify_song
        with st.spinner("Running identify_song ..."):
            song_id, info = identify_fn(clip_prepared, fp, fs=target_sr, return_info=True)
        analysis = analyze_query(clip_prepared, params, mode)
        st.session_state.state["last_result"] = {
            "song_id": song_id,
            "info": info,
            "analysis": analysis,
            "clip_label": clip_label,
            "clip_sr": clip_sr,
            "clip_duration": clip_prepared.size / target_sr,
            "fp_params": params,
            "augment_summary": augment_summary,
            "augment_cfg": augment_cfg,
        }
        state["clip_defaults"] = {
            "source": clip_source,
            "song_idx": current_song_idx,
            "clip_len": float(clip_len) if 'clip_len' in locals() else clip_defaults.get("clip_len", 3.0),
            "start_time": float(start_time) if 'start_time' in locals() else clip_defaults.get("start_time", 0.0),
        }

    result = st.session_state.state.get("last_result")
    viz_col, results_col = st.columns([1.8, 1.2], gap="large")
    if not result:
        viz_col.info("Select a clip and press identify to view the spectrogram.")
        results_col.info("Match results will appear after running the query.")
        return

    info = result["info"]
    analysis = result["analysis"]
    fp_params = result["fp_params"]
    hop = analysis.get("hop_length", fp_params["hop_length"])
    fs = analysis.get("fs", fp_params["fs"])

    with viz_col:
        st.subheader("3. Fingerprint visualization")
        all_matches = info.get("all_matched_peaks", [])
        match_options = ["Best offset", "All matched"]
        offset_labels = {}
        for entry in info.get("top_matches", []):
            label = f"{entry.get('title','Song')} @ {entry['offset']} bin"
            option = f"{entry['song_id']}::{entry['offset']}"
            offset_labels[option] = label + (" ★" if entry.get("is_best") else "")
            match_options.append(option)
        match_choice = st.selectbox("Match cluster filter", match_options, help="Show only the best offset, all matches, or a specific (song, offset) pair")
        if match_choice == "Best offset":
            matched_points = info.get("matched_query_peaks", [])
        elif match_choice == "All matched":
            matched_points = all_matches
        else:
            song_id_str, offset_str = match_choice.split("::")
            target_song = int(song_id_str)
            target_offset = int(offset_str)
            matched_points = [
                p for p in all_matches if p.get("song_id") == target_song and p.get("offset") == target_offset
            ]
            st.caption(offset_labels.get(match_choice, ""))

        show_all_peaks = st.checkbox("Overlay all detected peaks", value=True)
        spec_fig = build_spectrogram_figure(analysis, matched_points, show_all_peaks=show_all_peaks)
        st.plotly_chart(spec_fig, width="stretch")

        if matched_points:
            peak_df = pd.DataFrame(matched_points)
            st.dataframe(peak_df[[c for c in peak_df.columns if c in {"time_sec", "freq_hz", "chroma_bin", "song_id", "offset"}]], height=200)
        else:
            st.info("No matched peaks for the current filter.")

    with results_col:
        st.subheader("4. Identification result")
        song_id = result["song_id"]
        if song_id >= 0:
            st.success(f"Match: {info.get('title', 'Unknown')} ({info.get('genre','')})")
        else:
            st.error("No song matched")
        confidence = float(info.get("confidence", 0.0))
        st.progress(min(max(confidence, 0.0), 1.0), text=f"Confidence {confidence:.2f}")
        st.write(f"Best offset: {info.get('best_offset', 0)} bins (~{info.get('best_offset',0)*hop/fs:.2f}s)")
        st.write(f"Votes: {info.get('best_votes',0)} / {info.get('total_votes_for_song',0)}")

        st.markdown("**Offset vote distribution**")
        st.plotly_chart(build_offset_chart(info.get("top_matches", []), hop, fs), width="stretch")

        st.markdown("**Fingerprint density stats**")
        num_peaks = info.get("query_num_peaks", 0)
        matched_count = info.get("matched_peak_count", 0)
        matched_fraction = info.get("matched_peak_fraction", 0.0)
        st.write(
            f"Query peaks: {num_peaks} · Hashes: {info.get('query_num_hashes', 0)} · Matched hashes: {info.get('matched_hashes_total', 0)}"
        )
        st.write(f"Matched peaks: {matched_count} ({matched_fraction*100:.1f}% of detected peaks)")

        st.markdown("**Clip metadata**")
        st.write(f"Source: {result.get('clip_label','')} · Duration {result.get('clip_duration',0):.2f}s · Original sample rate {result.get('clip_sr', fs)} Hz")
        aug_summary = result.get("augment_summary", [])
        if aug_summary:
            st.write("Query perturbations: " + " | ".join(aug_summary))
        else:
            st.write("Query perturbations: none (raw clip)")

        if info.get("votes_by_song"):
            votes_df = pd.DataFrame(
                [
                    {
                        "song_id": k,
                        "votes": v,
                    }
                    for k, v in info["votes_by_song"].items()
                ]
            ).sort_values("votes", ascending=False)
            st.markdown("**Votes per song**")
            st.dataframe(votes_df, height=200)


if __name__ == "__main__":
    main()
