import glob, os, re
import numpy as np
import librosa
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

SR = 16000  # resample target for consistent analysis

def speaker_from_train_path(p):
    # ./train/speakerXX/filename.ogg
    return int(re.search(r"speaker(\d+)", p).group(1))

def speaker_from_test_path(p):
    # ./test/TestX/TestX-Y.ogg
    return int(re.search(r"Test(\d+)", os.path.basename(os.path.dirname(p))).group(1))

def extract_features(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    if len(y) < sr * 0.1:
        y, sr = librosa.load(path, sr=SR, mono=True)  # fallback if trim too aggressive

    feats = {}

    # MFCC (13) + delta + delta2
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=1024, hop_length=256)
    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    for i in range(13):
        feats[f"mfcc{i+1}_mean"] = np.mean(mfcc[i])
        feats[f"mfcc{i+1}_std"] = np.std(mfcc[i])
    for i in range(13):
        feats[f"dmfcc{i+1}_mean"] = np.mean(delta1[i])
    for i in range(13):
        feats[f"ddmfcc{i+1}_mean"] = np.mean(delta2[i])

    # Pitch (F0) via pyin
    try:
        f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
                                            fmax=librosa.note_to_hz('C7'), sr=sr)
        f0v = f0[~np.isnan(f0)]
        if len(f0v) > 0:
            feats["f0_mean"] = np.mean(f0v)
            feats["f0_std"] = np.std(f0v)
        else:
            feats["f0_mean"] = 0.0
            feats["f0_std"] = 0.0
    except Exception:
        feats["f0_mean"] = 0.0
        feats["f0_std"] = 0.0

    # Spectral features
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flat = librosa.feature.spectral_flatness(y=y)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]

    feats["centroid_mean"] = np.mean(cent); feats["centroid_std"] = np.std(cent)
    feats["bandwidth_mean"] = np.mean(bw); feats["bandwidth_std"] = np.std(bw)
    feats["rolloff_mean"] = np.mean(roll); feats["rolloff_std"] = np.std(roll)
    feats["flatness_mean"] = np.mean(flat)
    feats["contrast_mean"] = np.mean(contrast)
    feats["zcr_mean"] = np.mean(zcr); feats["zcr_std"] = np.std(zcr)
    feats["rms_mean"] = np.mean(rms); feats["rms_std"] = np.std(rms)

    # duration
    feats["duration"] = len(y) / sr

    return feats

def build_dataset():
    rows = []
    train_files = sorted(glob.glob("./train/speaker*/*.ogg"))
    test_files = sorted(glob.glob("./test/Test*/*.ogg"))

    for p in train_files:
        spk = speaker_from_train_path(p)
        f = extract_features(p)
        f["speaker"] = spk
        f["split"] = "train"
        f["path"] = p
        f["filename"] = os.path.basename(p)
        rows.append(f)

    for p in test_files:
        spk = speaker_from_test_path(p)
        f = extract_features(p)
        f["speaker"] = spk
        f["split"] = "test"
        f["path"] = p
        f["filename"] = os.path.basename(p)
        rows.append(f)

    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    df = build_dataset()
    feat_cols = [c for c in df.columns if c not in ("speaker","split","path","filename")]
    print("n_features =", len(feat_cols))
    print(df.groupby(["split","speaker"]).size())
    df.to_csv("./features.csv", index=False)
    print("saved to ./features.csv, shape=", df.shape)
