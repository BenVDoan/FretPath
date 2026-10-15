import json
from pathlib import Path
import librosa
import numpy as np
import os
from scorer import check, score

def note_estimate(filepath):
    y, sr = librosa.load(filepath)
    stft = librosa.stft(y, n_fft = 2048)
    pyin_f0, voiced_flag, voiced_probs = librosa.pyin(y, sr = sr, fmin = 80, fmax = 670, frame_length = 2048)
    notes = librosa.hz_to_note(pyin_f0, unicode=False)
    logS = librosa.amplitude_to_db(np.abs(stft), ref = np.max)
    diffS = np.diff(logS, axis=-1, prepend=logS[:, :1])

    diffS_thresh = np.maximum(diffS, 0)
    onset_env = np.mean(diffS_thresh, axis=0)
    onset_peaks = librosa.util.localmax(onset_env)
    onset_detect = librosa.onset.onset_detect(onset_envelope=onset_env)

    note_trim = np.array([])
    for i in range(len(onset_detect)):
        start = onset_detect[i]
        end = onset_detect[i + 1] if i + 1 < len(onset_detect) else len(notes)

        segment_notes = notes[start:end]
        segment = [n for n in segment_notes if n != "" ]
        if len(segment) == 0:
            continue

        cnt = {}
        segment_notes = []
        for n in segment:
            cnt[n] = cnt.get(n, 0) + 1
            segment_notes.append(n)
        common_note = max(cnt, key=cnt.get)
        note_trim = np.append(note_trim, common_note)

    return note_trim

def pipeline(paths, truths):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(script_dir, "..", "audio")
    total = {"lcs": 0, "n_det": 0, "n_exp": 0}
    for name in paths.values():
        file = os.path.join(audio_dir, name + ".mp3")
        if name not in truths or not os.path.exists(file):
            continue
        note_det = note_estimate(file)
        note_real = truths[name]["notes"]
        current = check(note_real, note_det)
        total["lcs"] += current["lcs"]
        total["n_det"] += current["n_det"]
        total["n_exp"] += current["n_exp"]

        recall = current["lcs"] / current["n_exp"] if current["n_exp"] else 0
        precision = current["lcs"] / current["n_det"] if current["n_det"] else 0

        print(f"{name}: (precision: {precision}, recall: {recall})")
    recall, precision, F1 = score(total["lcs"], total["n_det"], total["n_exp"])
    return (recall, precision, F1)