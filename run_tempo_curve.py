import numpy as np, matplotlib.pyplot as plt
from scipy.signal import resample
from shazam_system import compute_fingerprints, compute_fingerprints_chroma, identify_song, identify_song_chroma, load_music_db, _iter_music_db
fs=16000
clip_sec=3.0
nq=2
tempo_vals=[0.8,0.9,1.0,1.1,1.2]
rng=np.random.default_rng(456)

def sample_clip(sig):
    L=int(fs*clip_sec)
    if len(sig)<=L: return sig.copy()
    start=int(rng.integers(0,len(sig)-L))
    return sig[start:start+L]

def time_stretch_resample(clip, rate):
    new_len=max(1,int(round(len(clip)/rate)))
    return resample(clip, new_len)

def eval_curve(vals, identify_fn, flat_db):
    accs=[]
    for v in vals:
        hits=0
        for _ in range(nq):
            idx=int(rng.integers(0,len(flat_db)))
            sig=flat_db[idx][2]
            clip=time_stretch_resample(sample_clip(sig), v)
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
tempo_base=eval_curve(tempo_vals, identify_base, flat)
tempo_chroma=eval_curve(tempo_vals, identify_chroma, flat)
print('tempo_base', tempo_base)
print('tempo_chroma', tempo_chroma)

plt.figure(figsize=(7,4))
plt.plot(tempo_vals, tempo_base, marker='o', label='Baseline')
plt.plot(tempo_vals, tempo_chroma, marker='o', label='Chroma')
plt.xlabel('Tempo factor')
plt.ylabel('Accuracy')
plt.ylim(0,1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.title('Accuracy vs Tempo change (n=2)')
plt.tight_layout()
plt.savefig('plots/tempo_curve.png', dpi=200)
print('saved plots/tempo_curve.png')
