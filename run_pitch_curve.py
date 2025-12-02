import numpy as np, matplotlib.pyplot as plt
from scipy.signal import resample
from shazam_system import compute_fingerprints, compute_fingerprints_chroma, identify_song, identify_song_chroma, load_music_db, _iter_music_db
fs=16000
clip_sec=3.0
nq=2
pitch_vals=[-6,-4,-2,0,2,4,6]
rng=np.random.default_rng(123)

def sample_clip(sig):
    L=int(fs*clip_sec)
    if len(sig)<=L: return sig.copy()
    start=int(rng.integers(0,len(sig)-L))
    return sig[start:start+L]

def pitch_shift_resample(clip, semi):
    factor=2.0**(semi/12.0)
    new_len=max(1,int(round(len(clip)/factor)))
    return resample(clip, new_len)

def eval_curve(vals, identify_fn, flat_db):
    accs=[]
    for v in vals:
        hits=0
        for _ in range(nq):
            idx=int(rng.integers(0,len(flat_db)))
            sig=flat_db[idx][2]
            clip=pitch_shift_resample(sample_clip(sig), v)
            pred,_=identify_fn(clip)
            hits += (pred==idx)
        accs.append(hits/float(nq))
    return accs

print('loading db...')
music_db=load_music_db('Project6_musicDB.mat')
flat=list(_iter_music_db(music_db))
print('building fps...')
fp_base=compute_fingerprints(music_db)
fp_chroma=compute_fingerprints_chroma(music_db)

identify_base=lambda clip: identify_song(clip, fp_base, fs=fs, return_info=True)
identify_chroma=lambda clip: identify_song_chroma(clip, fp_chroma, fs=fs, return_info=True)

print('evaluating...')
pitch_base=eval_curve(pitch_vals, identify_base, flat)
pitch_chroma=eval_curve(pitch_vals, identify_chroma, flat)
print('pitch_base', pitch_base)
print('pitch_chroma', pitch_chroma)

plt.figure(figsize=(7,4))
plt.plot(pitch_vals, pitch_base, marker='o', label='Baseline')
plt.plot(pitch_vals, pitch_chroma, marker='o', label='Chroma')
plt.xlabel('Pitch shift (semitones)')
plt.ylabel('Accuracy')
plt.ylim(0,1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.title('Accuracy vs Pitch shift (n=2)')
plt.tight_layout()
plt.savefig('plots/pitch_curve.png', dpi=200)
print('saved plots/pitch_curve.png')
