import numpy as np, librosa, matplotlib.pyplot as plt
from shazam_system import compute_fingerprints_chroma, identify_song_chroma, load_music_db, _iter_music_db
fs=16000
clip_sec=3.0
nq=2
rng=np.random.default_rng(99)
pitch_vals=[-4,0,4]
tempo_vals=[0.9,1.0,1.1]

print('loading db...')
music_db=load_music_db('Project6_musicDB.mat')
flat=list(_iter_music_db(music_db))
print('building chroma fp...')
fp=compute_fingerprints_chroma(music_db)

identify=lambda clip: identify_song_chroma(clip, fp, fs=fs, return_info=True)

def sample_clip(sig):
    L=int(fs*clip_sec)
    if len(sig)<=L: return sig.copy()
    start=int(rng.integers(0,len(sig)-L))
    return sig[start:start+L]

H=np.zeros((len(tempo_vals), len(pitch_vals)))
for i,t in enumerate(tempo_vals):
    for j,p in enumerate(pitch_vals):
        hits=0
        for _ in range(nq):
            idx=int(rng.integers(0,len(flat)))
            sig=flat[idx][2]
            clip=sample_clip(sig)
            clip=librosa.effects.pitch_shift(y=clip, sr=fs, n_steps=p)
            clip=librosa.effects.time_stretch(y=clip, rate=t)
            pred,_=identify(clip)
            hits += (pred==idx)
        H[i,j]=hits/float(nq)
print('heatmap accuracies')
print(H)
plt.figure(figsize=(6,4.5))
im=plt.imshow(H, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=1)
plt.colorbar(im, label='Accuracy')
plt.xticks(range(len(pitch_vals)), pitch_vals)
plt.yticks(range(len(tempo_vals)), tempo_vals)
plt.xlabel('Pitch shift (semitones)')
plt.ylabel('Tempo factor')
plt.title('Chroma accuracy heatmap (n=2)')
plt.tight_layout()
plt.savefig('plots/pitch_tempo_heatmap.png', dpi=200)
print('saved plots/pitch_tempo_heatmap.png')
