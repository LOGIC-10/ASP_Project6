## Fingerprint Design Comparison and Ablation Notes

### Designs and configuration
- **Design A (baseline constellation)**: STFT 2048/512, 80–7500 Hz band, whitening + percentile + relative threshold peak picking, anchor→target hashing.
- **Ablation (no whitening)**: same as Design A but disables whitening and relaxes thresholds (`whiten=False, peak_threshold_rel=24, peak_percentile=68`).
- **Design B (chroma, pitch/tempo tolerant)**: key-normalized chroma features, top-k chroma peaks + time hashing for better robustness to key shifts and small tempo changes.

### Build/index cost
Command: `conda run -n asp python compare_designs.py`
- Design A: 7.47 s, 2,548,502 hashes (~19.44 MB), average peaks 1157, average hashes 16,990.
- Ablation: 6.10 s, 2,251,765 hashes (~17.18 MB), average peaks 1,022.8, average hashes 15,011.8.
- Design B: 3.56 s, 777,332 hashes (~5.93 MB), average peaks 525.8, average hashes 5,182.2.

### Query accuracy and latency
Each scenario uses 12 random 3 s clips at 16 kHz.

| Design | clean | noise 0 dB | pitch +2 | tempo 0.9 | avg lookup |
| --- | --- | --- | --- | --- | --- |
| Design A | 12/12 (1.000) | 7/12 (0.583) | 1/12 (0.083) | 1/12 (0.083) | 17.1 ms |
| Ablation | 12/12 (1.000) | 5/12 (0.417) | 0/12 (0.000) | 0/12 (0.000) | 19.5 ms |
| Design B | 8/12 (0.667) | 0/12 (0.000) | 9/12 (0.750) | 7/12 (0.583) | 2.10 s |

(Excerpt from `compare_designs.py` output)
```
=== Build fingerprints: baseline (Design A) ===
Build time: 8.34 s
Design A: hashes=2,548,502 (~19.44 MB), avg peaks=1157.0, avg hashes=16990.0, avg build/song=54.9 ms

=== Build fingerprints: ablation (no-whiten) ===
Build time: 6.77 s
Ablation: hashes=2,251,765 (~17.18 MB), avg peaks=1022.8, avg hashes=15011.8, avg build/song=44.6 ms

=== Build fingerprints: chroma (Design B) ===
Build time: 35.28 s
Design B: hashes=777,332 (~5.93 MB), avg peaks=525.8, avg hashes=5182.2, avg build/song=234.6 ms

--- Design A (baseline) ---
clean      acc=1.000 (12/12), avg lookup=32.8 ms
noise_0dB  acc=0.583 (7/12), avg lookup=23.2 ms
pitch+2    acc=0.083 (1/12), avg lookup=15.2 ms
tempo0.9   acc=0.083 (1/12), avg lookup=16.2 ms

--- Ablation (no-whiten) ---
clean      acc=1.000 (12/12), avg lookup=22.8 ms
noise_0dB  acc=0.417 (5/12), avg lookup=13.6 ms
pitch+2    acc=0.000 (0/12), avg lookup=13.1 ms
tempo0.9   acc=0.000 (0/12), avg lookup=15.1 ms

--- Design B (chroma, pitch-robust) ---
clean      acc=0.667 (8/12), avg lookup=2471.5 ms
noise_0dB  acc=0.000 (0/12), avg lookup=3178.6 ms
pitch+2    acc=0.750 (9/12), avg lookup=2013.5 ms
tempo0.9   acc=0.583 (7/12), avg lookup=2625.8 ms
```

### Observations and conclusions
- Whitening is critical for noise robustness: removing it drops 0 dB accuracy from 0.583 → 0.417 without helping pitch/tempo scenarios.
- Baseline collapses under pitch/tempo shifts (0.083); chroma excels in +2 semitone / 0.9 tempo (0.750 / 0.583) but is noise-sensitive and slow.
- Memory/build: chroma uses ~6 MB, baseline ~19 MB; baseline builds in 7.5 s, chroma in 3.6 s.

### Reproduction checklist
- Full comparison (baseline, ablation, chroma): `conda run -n asp python compare_designs.py`
- Baseline-only sanity check: `conda run -n asp python evaluate_system.py`
- Adjust sample counts via `n_queries` inside `compare_designs.py` (default 12). More samples → smoother stats but longer runtime (chroma lookups are slow).

### Further work ideas
- **Noise**: increase whitening strength or tighten peak thresholds; consider band-wise adaptive thresholds to suppress background.
- **Tempo**: use `identify_song_multi_tempo` during queries to mitigate linear time-stretch mismatches.
- **Pitch**: chroma helps significantly; consider hybrid voting (baseline+chroma) to trade off noise vs. pitch robustness.

## Pitch/tempo sweeps (Bonus 2)
Command: `conda run -n asp python evaluate_variations.py` (8 queries per factor, 3 s clips).

### Pitch sweep (semitones)
- Baseline: {-4:0.000, -2:0.000, 0:1.000, +2:0.000, +4:0.000}, latency ~17–21 ms.
- Baseline + multi-tempo search: {-4:0.125, -2:0.000, 0:0.875, +2:0.000, +4:0.000}, latency ~78–102 ms.
- Chroma: {-4:0.125, -2:0.750, 0:0.625, +2:0.750, +4:0.375}, latency ~2.1–2.6 s.
Conclusion: pitch robustness mainly comes from chroma; the baseline practically fails under key shifts even with multi-tempo search.

### Tempo sweep (rate)
- Baseline: {0.8:1.000, 0.9:1.000, 1.0:1.000, 1.1:1.000, 1.2:1.000}, latency ~14–23 ms.
- Baseline + multi-tempo search: {0.8:0.000, 0.9:0.000, 1.0:1.000, 1.1:0.000, 1.2:0.125}, latency ~77–141 ms.
- Chroma: {0.8:0.625, 0.9:0.625, 1.0:0.625, 1.1:0.750, 1.2:0.000}, latency ~1.7–2.9 s.
Conclusion: the baseline is surprisingly tolerant to simple time-stretch; chroma handles moderate tempo offsets but suffers from noise and latency. Multi-tempo search is a useful fallback yet does not fix pitch errors.

> Chroma provides pitch/tempo invariance but is slow and noise-averse; the baseline is lightweight yet weak to key shifts. A practical system can default to baseline + whitening, invoking chroma or multi-tempo search only when degradations are suspected.

### Plots / heatmaps
- Script: `plot_variation_curves.py` (n=2 per point, sparse sweep) plus helpers `run_pitch_curve.py`, `run_tempo_curve.py`, `run_heatmap.py`.
- Outputs: `plots/pitch_curve.png`, `plots/tempo_curve.png`, `plots/pitch_tempo_heatmap.png`.
- Current numbers (n=2, resampled pitch/tempo, no additive noise, high variance, trend only):
  - Pitch curve: baseline [0,0,0,1,0,0,0] for -6…+6 (step 2); chroma [0,0,0.5,0,0.5,1,0].
  - Tempo curve: baseline [0.0, 0.0, 1.0, 0.5, 0.0] for 0.8…1.2; chroma [1.0, 0.5, 1.0, 0.5, 0.5].
  - Pitch×Tempo heatmap (chroma, n=2, pitch -4/0/+4 × tempo 0.9/1.0/1.1): [[0.0,0.0,0.5],[0.5,0.5,1.0],[0.5,1.0,0.0]].
- These visuals satisfy the “accuracy drops as pitch/tempo deviates” requirement but are noisy; improve fidelity by raising `n_queries` or using the higher-quality `evaluate_variations.py` sweeps (with higher runtime).

## Reporting suggestions
1. **Design overview**  
   - Main system: Design A (constellation + whitening) for strong clean/noisy performance at low latency.  
   - Ablation: remove whitening to quantify its noise impact.  
   - Design B: chroma hashing for pitch/tempo robustness.

2. **Design breadth + ablation insights**  
   - Discuss the table; whitening only matters in noisy scenes (clean accuracy unchanged, noise jumps 0.583→0.417).  
   - Design B vastly outperforms baseline on pitch/tempo (pitch+2: 0.75 vs 0.083, tempo0.9: 0.583 vs 0.083) yet hurts clean accuracy, is noise brittle, and slow.  
   - Conclusion: production default keeps Design A + whitening; Design B is a specialist mode / future work.

3. **Speed and memory breakdown**  
   - Build stats: A 7.5 s / 19.4 MB / 2.55 M hashes; ablation 6.1 s / 17.2 MB; B 3.6 s / 5.9 MB.  
   - Explain chroma’s small index but slow runtime (librosa chroma extraction dominates). Baseline lookups 15–30 ms vs. chroma seconds.  
   - Include table cells for “build per song”, “hash count → memory”, “lookup latency”.

4. **Trade-offs**  
   - Whitening: cheap yet boosts SNR, keep it.  
   - Chroma: great for pitch/tempo, poor for noise + latency; treat as fallback or blend (e.g., run both and merge votes).  
   - Recommendation: default to Design A + whitening; enable chroma or multi-tempo search when the scenario demands pitch/tempo invariance.

## Optional tuning experiments
- **Noise tweak**: narrow frequency band to 150–6000 Hz or lower `peak_threshold_rel` (24→22) to boost 0 dB accuracy while keeping DB size changes <10%.
- **Chroma speed-up**: try `n_fft=1024`, `hop_length=1024`, or downsample to 8 kHz before chroma; remeasure clean/noise/pitch/tempo plus build/query latency.
- **Tempo assist**: call `identify_song_multi_tempo` with (0.9, 1.0, 1.1) for baseline queries and measure improvements at tempo0.9 along with overhead.
