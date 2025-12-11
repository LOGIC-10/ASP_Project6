# UI Bonus Interface Test Log

This document lists the scripted checks used to verify that the Streamlit bonus interface can automatically load data, run identification, and visualize fingerprint metadata. Each test shows the exact command and the raw console output captured on the `asp` conda environment.

---

## Test 1 – Default clip identification pipeline
Ensures the default 3 s clip from the first song can load through the entire pipeline (`load_music_library` → `load_or_build_fingerprints` → `identify_song` → `analyze_query`).

Command:
```bash
conda run --no-capture-output -n asp python - <<'PY'
from ui_demo import load_music_library, load_or_build_fingerprints, slice_signal, analyze_query
from shazam_system import identify_song
from pathlib import Path

mat = 'Project6_musicDB.mat'
songs = load_music_library(mat)
fp, summary = load_or_build_fingerprints(mat, 'Baseline (constellation)', Path('fingerprint_cache/test_constellation.pkl'), persist=True)
clip = slice_signal(songs[0]['signal'], 16000, 0.0, 3.0)
song_id, info = identify_song(clip, fp, fs=16000, return_info=True)
print('songs', len(songs))
print('hash_entries', summary['hash_index_size'])
print('match_id', song_id, 'title', info.get('title'), 'confidence', round(info.get('confidence',0),3))
print('matched_peaks', len(info.get('matched_query_peaks', [])))
print('top_matches', info.get('top_matches', [])[:2])
analysis = analyze_query(clip, fp.params, 'Baseline (constellation)')
print('analysis_spec_shape', analysis['spec'].shape)
print('analysis_peak_count', len(analysis['all_peaks']))
PY
```

Output:
```
songs 150
hash_entries 2548502
match_id 0 title Eastsouthern-Louisiana. confidence 0.971
matched_peaks 1672
top_matches [{'song_id': 0, 'offset': 0, 'votes': 1672, 'title': 'Eastsouthern-Louisiana.', 'is_best': True}, {'song_id': 0, 'offset': -6, 'votes': 10, 'title': 'Eastsouthern-Louisiana.', 'is_best': False}]
analysis_spec_shape (1025, 90)
analysis_peak_count 368
```

---

## Test 2 – Augmentation controls feeding identify_song
Validates that the noise/pitch/tempo toggles produce a usable clip and that `identify_song` still returns metadata for visualization.

Command:
```bash
conda run --no-capture-output -n asp python - <<'PY'
from ui_demo import apply_augmentations, load_music_library, load_or_build_fingerprints, slice_signal, prepare_clip
from shazam_system import identify_song
from pathlib import Path

mat = 'Project6_musicDB.mat'
songs = load_music_library(mat)
clip = slice_signal(songs[0]['signal'], 16000, 0.0, 3.0)
aug_clip, summary = apply_augmentations(
    clip,
    16000,
    add_noise=True,
    noise_snr_db=12.0,
    noise_seed=123,
    pitch_shift=1.5,
    tempo_factor=0.95,
)
fp, _ = load_or_build_fingerprints(mat, 'Baseline (constellation)', Path('fingerprint_cache/test_constellation.pkl'), persist=True)
prep = prepare_clip(aug_clip, 16000, fp.params['fs'])
song_id, info = identify_song(prep, fp, fs=fp.params['fs'], return_info=True)
print('augment_summary', summary)
print('matched_song', song_id, info.get('title'))
print('confidence', round(info.get('confidence', 0), 3))
print('matched_peaks', len(info.get('matched_query_peaks', [])))
PY
```

Output:
```
augment_summary ['Noise SNR≈12.0 dB', 'Pitch shift +1.5 semitones', 'Tempo ×0.95']
matched_song 13 The_Oscar_Jordan_Band-Mister_Bad_Luck.
confidence 0.147
matched_peaks 5
```

---

## Test 3 – Visualization helpers
Checks that the spectrogram builder and offset chart render without errors for representative data.

Command:
```bash
conda run --no-capture-output -n asp python - <<'PY'
from ui_demo import build_spectrogram_figure, build_offset_chart, analyze_query, load_music_library, load_or_build_fingerprints, slice_signal
from pathlib import Path

mat='Project6_musicDB.mat'
songs=load_music_library(mat)
clip=slice_signal(songs[0]['signal'],16000,0.0,3.0)
fp,_=load_or_build_fingerprints(mat,'Baseline (constellation)',Path('fingerprint_cache/test_constellation.pkl'),persist=True)
analysis=analyze_query(clip,fp.params,'Baseline (constellation)')
fig=build_spectrogram_figure(analysis, [], show_all_peaks=True)
print('spectrogram_traces', len(fig.data))
chart=build_offset_chart([
    {'song_id':0,'offset':0,'votes':120,'title':'SongA','is_best':True},
    {'song_id':1,'offset':5,'votes':40,'title':'SongB','is_best':False}
], fp.params['hop_length'], fp.params['fs'])
print('offset_bars', len(chart.data[0]['x']) if chart.data else 0)
PY
```

Output:
```
spectrogram_traces 2
offset_bars 2
```

---

These automated backend checks ensure that the interface can load default content, respond to user perturbations, and render all required visual elements. The Streamlit page itself uses the same code paths, so these scripts serve as regression tests for the bonus UI.
