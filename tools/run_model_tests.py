"""Repro: old vs fixed alert logic on synthetic + nonsense patches.
Mel extraction mirrors the firmware (mel_features.c): 16kHz, 400-sample
periodic Hann, 512 FFT magnitude, HTK mel 125-7500Hz, 64 bands, log(x+0.001),
96-frame patches, hop 160. Model: models/yamnet/yamnet.tflite (HF chayuto INT8).
"""
import sys; sys.path.insert(0, "pylibs")
import numpy as np, csv
from ai_edge_litert.interpreter import Interpreter

SR=16000; WIN=400; FFT=512; HOP=160; BANDS=64; FRAMES=96; FMIN=125.; FMAX=7500.
def hz2mel(h): return 1127*np.log(1+h/700)
def mel2hz(m): return 700*(np.exp(m/1127)-1)
pts = mel2hz(np.linspace(hz2mel(FMIN), hz2mel(FMAX), BANDS+2)) * FFT / SR
W = np.zeros((257, BANDS))
for m in range(1, BANDS+1):
    l,c,r = pts[m-1], pts[m], pts[m+1]
    for k in range(257):
        if l <= k < c: W[k, m-1] = (k-l)/(c-l)
        elif c <= k <= r: W[k, m-1] = (r-k)/(r-c)
HANN = 0.5*(1-np.cos(2*np.pi*np.arange(WIN)/WIN))

def logmel(audio):
    n_frames = 1 + max(0, (len(audio)-WIN)//HOP)
    out = np.zeros((n_frames, BANDS), np.float32)
    for i in range(n_frames):
        seg = audio[i*HOP:i*HOP+WIN] * HANN
        spec = np.abs(np.fft.rfft(seg, FFT)) / FFT  # FIXED: FFT-normalized magnitude (firmware mel_features.c change)
        out[i] = np.log(spec @ W + 0.001)
    return out

interp = Interpreter(model_path="SoundAlert-main/models/yamnet/yamnet.tflite")
interp.allocate_tensors(); inp=interp.get_input_details()[0]; outd=interp.get_output_details()[0]
iscale, izp = inp['quantization']; oscale, ozp = outd['quantization']
names={}
with open("SoundAlert-main/models/yamnet/yamnet_class_map.csv") as f:
    for row in csv.DictReader(f): names[int(row['index'])]=row['display_name']
def infer(patch96):
    q=(patch96/iscale+izp).round().clip(-128,127).astype(np.int8)
    interp.set_tensor(inp['index'], q[None]); interp.invoke()
    y=interp.get_tensor(outd['index'])[0]
    return oscale*(y.astype(np.float32)-ozp)

FAMILY={19,20,21,22,304,317,318,319,382,389,390,391,393,394}
def old_decision(p):
    emerg=max(p[i] for i in [317,318,319,390,391,393,394]); baby=p[20]; phone=p[384]
    if baby>=0.50 and (max(emerg,phone)-baby)<=0.12: return True,"BABY(special-case)"
    best=max(emerg,baby,phone)
    if best<0.60: return False,"low"
    second=sorted([emerg,baby,phone])[-2]
    if best-second<0.08: return False,"low-margin"
    return True, "ALERT"
def new_decision(p):
    baby=p[20]; siren=max(p[i] for i in [317,318,319,390,391]); fire=max(p[i] for i in [393,394])
    cand,thr,cname = max([(baby,0.65,'BABY'),(siren,0.65,'SIREN'),(fire,0.60,'FIRE')], key=lambda t:t[0])
    unk_i = max((i for i in range(521) if i not in FAMILY), key=lambda i:p[i])
    if cand<thr: return False,f"low {cand:.2f}<{thr}"
    if p[unk_i] > cand+0.05: return False,f"UNKNOWN {names[unk_i]}={p[unk_i]:.2f} > target {cand:.2f}"
    return True,f"pass {cname} ({cand:.2f}, unk {names[unk_i]}={p[unk_i]:.2f})"

def evaluate(label, audio, dbfs_note=""):
    lm = logmel(audio.astype(np.float32))
    # non-overlapping 96-frame patches like firmware
    patches=[lm[i:i+FRAMES] for i in range(0, max(1,len(lm)-FRAMES+1), FRAMES)]
    print(f"\n=== {label} {dbfs_note} patches={len(patches)}")
    for pi,patch in enumerate(patches):
        if len(patch)<FRAMES: patch=np.pad(patch,((0,FRAMES-len(patch)),(0,0)),constant_values=-6.9)
        p=infer(patch); top=np.argsort(-p)[:3]
        o=old_decision(p); n=new_decision(p)
        print(f" patch{pi}: top={[(names[int(i)],round(float(p[i]),2)) for i in top]} baby={p[20]:.2f} siren={max(p[i] for i in [317,318,319,390,391]):.2f} fire={max(p[i] for i in [393,394]):.2f} | OLD={o} NEW={n}")

t=np.arange(SR*5)/SR; rng=np.random.default_rng(1)
# POSITIVE CONTROLS (synthetic, close to real for tonal alarms)
evaluate("SIREN wail 600->1400Hz sweep", 0.5*np.sin(2*np.pi*(1000+400*np.sin(2*np.pi*0.25*t))*t))
smoke=np.zeros_like(t); smoke[((t%1.5)<0.6)]=0.5*np.sin(2*np.pi*3100*t)[((t%1.5)<0.6)]
evaluate("SMOKE ALARM 3.1kHz beeps", smoke)
cry_f0=420+120*np.sin(2*np.pi*1.4*t); cry=np.sum([ (0.4/k)*np.sin(2*np.pi*k*cry_f0*t + 0.3*np.sin(2*np.pi*8*t)) for k in range(1,5)],axis=0)*(0.6+0.4*np.sin(2*np.pi*1.4*t))
evaluate("BABY-CRY-LIKE harmonic wail (synthetic, rough)", cry)
# NEGATIVES
evaluate("WHITE NOISE loud", 0.3*rng.standard_normal(len(t)))
evaluate("WHITE NOISE quiet (-40dB)", 0.003*rng.standard_normal(len(t)))
evaluate("MUSIC melody (piano-like harmonics)", np.sum([0.3*np.sin(2*np.pi*f*t)*(np.floor(t*2)%4==i) for i,f in enumerate([262,330,392,523])],axis=0))
evaluate("CLAPS / transients 1 per sec", np.where((t%1)<0.01, 0.8*rng.standard_normal(len(t)), 0.0))
evaluate("SPEECH-LIKE formant babble (synthetic)", np.sin(2*np.pi*180*t)*np.sin(2*np.pi*(900+500*np.sin(2*np.pi*3*t))*t)*0.4)
evaluate("SILENCE (digital zeros -> log-mel -6.9 via extractor on 1e-4 noise)", 0.0001*rng.standard_normal(len(t)))
