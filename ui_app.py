"""
Streamlit interface for Project 6 fingerprinting.

Run (from this directory) in the asp env:
    conda run -n asp streamlit run ui_app.py
"""
from __future__ import annotations

import io
from typing import List

import numpy as np
import soundfile as sf
import streamlit as st

import librosa
import librosa.display  # type: ignore
import matplotlib.pyplot as plt

from shazam_system import (
    compute_fingerprints,
    compute_fingerprints_chroma,
    identify_song,
    identify_song_chroma,
    load_music_db,
)


@st.cache_resource
def _load_fingerprints(mode: str):
    music_db = load_music_db("Project6_musicDB.mat")
    if mode == "Chroma (pitch-robust)":
        return compute_fingerprints_chroma(music_db), mode
    return compute_fingerprints(music_db), mode


def _ensure_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1)


def _plot_spectrogram(x: np.ndarray, sr: int, matched_peaks: List[dict] | None):
    fig, ax = plt.subplots(figsize=(9, 4))
    S = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis="time", y_axis="log", ax=ax, cmap="magma")
    ax.set_title("Query spectrogram (log freq)")
    if matched_peaks:
        times = [p["time_sec"] for p in matched_peaks if "freq_hz" in p]
        freqs = [p["freq_hz"] for p in matched_peaks if "freq_hz" in p]
        if times and freqs:
            ax.scatter(times, freqs, c="cyan", s=18, marker="o", edgecolor="k", linewidth=0.4, alpha=0.8, label="matched peaks")
            ax.legend(loc="upper right")
    plt.tight_layout()
    return fig


def _plot_chroma(x: np.ndarray, sr: int, matched_peaks: List[dict] | None):
    fig, ax = plt.subplots(figsize=(9, 3))
    chroma = librosa.feature.chroma_stft(y=x, sr=sr, n_fft=2048, hop_length=512)
    librosa.display.specshow(chroma, y_axis="chroma", x_axis="time", hop_length=512, cmap="magma", ax=ax)
    ax.set_title("Query chroma")
    if matched_peaks:
        times = [p["time_sec"] for p in matched_peaks if "chroma_bin" in p]
        chroma_bins = [p["chroma_bin"] for p in matched_peaks if "chroma_bin" in p]
        if times and chroma_bins:
            ax.scatter(times, chroma_bins, c="cyan", s=18, marker="o", edgecolor="k", linewidth=0.4, alpha=0.8, label="matched peaks")
            ax.legend(loc="upper right")
    plt.tight_layout()
    return fig


def main():
    st.title("Project 6 Shazam Fingerprinter")
    st.write("Load a ~3 s audio clip, run identification, and inspect highlighted fingerprints on the spectrogram.")

    mode = st.sidebar.selectbox("Fingerprint design", ["Baseline (constellation)", "Chroma (pitch-robust)"])
    st.sidebar.write("Baseline: spectral peak hashes. Chroma: key-normalized chroma hashes, more pitch robust.")

    with st.spinner("Building fingerprint DB (≈7 s on first run)..."):
        fingerprints, fp_mode = _load_fingerprints(mode)
    st.success("Fingerprint DB ready")

    uploaded = st.file_uploader("Upload audio file (wav/flac/mp3)", type=["wav", "flac", "mp3"])
    if uploaded:
        data, sr = sf.read(io.BytesIO(uploaded.read()))
        data = _ensure_mono(data.astype(float))
        pred_fn = identify_song_chroma if fp_mode.startswith("Chroma") else identify_song
        song_id, info = pred_fn(data, fingerprints, fs=sr, return_info=True)
        st.subheader("Match results")
        if song_id >= 0:
            st.write(f"Song #{song_id} | Title: {info.get('title','')} | Genre: {info.get('genre','')}")
            st.write(f"Confidence: {info.get('confidence',0):.3f} | Votes: {info.get('best_votes',0)} / {info.get('total_votes_for_song',0)}")
        else:
            st.write("No match found")

        matched_peaks = info.get("matched_query_peaks") if isinstance(info, dict) else None
        if fp_mode.startswith("Chroma"):
            fig = _plot_chroma(data, sr, matched_peaks)
        else:
            fig = _plot_spectrogram(data, sr, matched_peaks)
        st.pyplot(fig)

        st.write("Highlighted points mark fingerprints that drove the winning match.")


if __name__ == "__main__":
    main()
