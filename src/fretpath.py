import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import math
import json
from pathlib import Path
from scorer import check, score
from pipeline import pipeline
# real unplugged electric guitar audio provides incorrect note assignments

script_dir = os.path.dirname(os.path.abspath(__file__))
audio_dir = os.path.join(script_dir, "..", "audio")

file_paths = {
    1: "DoIWannaKnow",
    2: "guitarTest1",
    3: "guitarTest2",
    4: "guitarTest3",
    5: "guitarTest4",
    6: "guitarTest5",
    7: "guitarTest6",
    8: "guitarTest7",
    9: "kneeSocksSimple",
    10: "roommates",
    11: "roommatesEffects",
    12: "realTest",
    13: "rUMine",
    14: "rUMineReal",
    15: "sweetChildOMine",
}
audio_name = file_paths.get(8, "default")
filename = os.path.join(audio_dir, audio_name + ".mp3")

y, sr = librosa.load(filename)
stft = librosa.stft(y, n_fft = 2048)
pyin_f0, voiced_flag, voiced_probs = librosa.pyin(y, sr = sr, fmin = 80, fmax = 670, frame_length = 2048)
notes = librosa.hz_to_note(pyin_f0, unicode=False)
logS = librosa.amplitude_to_db(np.abs(stft), ref = np.max)
diffS = np.diff(logS, axis=-1, prepend=logS[:, :1])

diffS_thresh = np.maximum(diffS, 0)
onset_env = np.mean(diffS_thresh, axis=0)
onset_peaks = librosa.util.localmax(onset_env)
onset_detect = librosa.onset.onset_detect(onset_envelope=onset_env)

standard = {
    1: "E4",
    2: "B3",
    3: "G3",
    4: "D3",
    5: "A2",
    6: "E2"
}

# makes dictionary of notes with list of each (string, fret) position for that note
def build_notes(frets = 12):
    positions = {}
    for string, note in standard.items():
        open = librosa.note_to_midi(note)
        for fret in range(frets + 1):
            note = librosa.midi_to_note(open + fret, unicode=False)
            positions.setdefault(note, []).append((string, fret))
    return positions

fretboard = build_notes()

# uses DP to estimate best fretboard by minimizing total distance between each position with time urgency weight
def fret_estimate_dp(notes, urgencies, string_weight = 1, fret_weight = 1):
    n = len(notes)
    dp = [{} for _ in range(n)]
    
    for start in fretboard[notes[0]]:
        dp[0][start] = (0, None)
    for i in range(1,n):
        urgency = urgencies[i - 1]
        for cand in fretboard[notes[i]]:
            best_prev = None
            best_cost = None
            for prev_pos, (prev_cost, _) in dp[i - 1].items():
                string_dist = abs((cand[0] - prev_pos[0]) * string_weight)**2
                fret_dist = abs((cand[1] - prev_pos[1]) * fret_weight)**2 
                step_cost = math.sqrt(fret_dist + string_dist) * urgency
                                    
                total_cost = step_cost + prev_cost

                if best_cost is None or total_cost < best_cost:
                    best_cost = total_cost
                    best_prev = prev_pos
            dp[i][cand] = (best_cost, best_prev)
    last_positions = dp[n - 1]
    final_pos = min(last_positions, key = lambda p: last_positions[p][0])

    path = [None] * n
    path[n - 1] = final_pos
    for i in range(n - 1, 0, -1):
        path[i - 1] = dp[i][path[i]][1]
    return path


# uses DP to estimate best fretboard by minimizing total distance between each position with time urgency weight and directional weight 
def fret_estimate_dp_dirr(notes, urgencies, string_weight = 1, fret_weight = 1, direction_weight = 1.5):
    n = len(notes)
    dp = [{} for _ in range(n)]
    for start in fretboard[notes[0]]:
        dp[0][(start, None)] = (0, None)
        
    for i in range(1,n):
        urgency = urgencies[i - 1]
        for cand in fretboard[notes[i]]:
            best_state = None
            best_cost = None
            for (prev_pos, prev2_pos), (prev_cost, _) in dp[i - 1].items():
                string_dist = abs((cand[0] - prev_pos[0]) * string_weight)**2
                fret_dir = cand[1] - prev_pos[1]
                fret_dist = abs((cand[1] - prev_pos[1]) * fret_weight)**2 
                step_cost = math.sqrt(fret_dist + string_dist) * urgency
                if prev2_pos is not None:
                    if (prev_pos[1] - prev2_pos[1] > 0 and fret_dir > 0) or (prev_pos[1] - prev2_pos[1] < 0 and fret_dir < 0):
                        step_cost *= direction_weight
                
                total_cost = step_cost + prev_cost

                if best_cost is None or total_cost < best_cost:
                    best_cost = total_cost
                    best_state = (prev_pos, prev2_pos)
            dp[i][(cand, best_state[0])] = (best_cost, best_state)
    last_positions = dp[n - 1]
    final_pos = min(last_positions, key = lambda p: last_positions[p][0])

    path = [None] * n
    path[n - 1] = final_pos
    for i in range(n - 1, 0, -1):
        path[i - 1] = dp[i][path[i]][1]
    return [state[0] for state in path]


# formats fretboard to have even spaces between frets for printing
def format_row(row, width=2):
    return " ".join(f"{cell:>{width}}" for cell in row)

# Turn onset_detect frames into list of notes, trims out empty notes
note_times = []
note_trim = np.array([])
for i in range(len(onset_detect)):
    start = onset_detect[i]
    end = onset_detect[i + 1] if i + 1 < len(onset_detect) else len(notes)

    segment_notes = notes[start:end]
    segment = [n for n in segment_notes if n != ""]
    if len(segment) == 0:
        continue

    cnt = {}
    segment_notes = []
    for n in segment:
        cnt[n] = cnt.get(n, 0) + 1
        segment_notes.append(n)
    common_note = max(cnt, key=cnt.get)
    note_trim = np.append(note_trim, common_note)
    note_times.append(librosa.frames_to_time(onset_detect[i]))
playable_idx = [i for i, n in enumerate(note_trim) if n in fretboard]
playable_notes = [note_trim[i] for i in playable_idx]
playable_times = [note_times[i] for i in playable_idx]
skipped = [(i, note_trim[i]) for i in range(len(note_trim)) if note_trim[i] not in fretboard]

data_path = Path(__file__).resolve().parents[1] / "data"
with open(data_path / "real_notes.json", "r") as f:
    real_notes = json.load(f)

# calculates relative timings between notes to average pacing of song to weigh urgency of each note transition
def relative_timings(note_times, time_weight):
    all_gaps = np.diff(note_times)
    median_gap = np.median(all_gaps)
    median_gap = max(median_gap, 0.01)

    urgencies = []
    for gap in all_gaps:
        gap = max(gap, 0.01)
        difference = gap / median_gap
        urgency = 1 / (difference ** time_weight)
        urgencies.append(urgency)
    return urgencies

if len(playable_notes) > 0:
    urgencies = relative_timings(playable_times, time_weight=2)
    frets = fret_estimate_dp(playable_notes, urgencies)
    frets2 = fret_estimate_dp_dirr(playable_notes, urgencies)
else:
    frets = []
    frets2 = []

# use note_trim for onset detection
boardpath = Path(__file__).resolve().parents[1] / "fretboards"
fretboard_str = np.full((6, len(note_trim)), "-", dtype="<U2")
fretboard_str2 = np.full((6, len(note_trim)), "-", dtype="<U2")
print(f"{audio_name}")
print(f"note_trim: {note_trim}\n")
print(f"estimated note count: {len(note_trim)} (playable: {len(playable_notes)})")
print(f"skipped (not on fretboard): {skipped}\n")

try:
    fretboard_real = np.full((6, len(real_notes[audio_name]['frets'])), "-", dtype="<U2")
    print(f"real_notes: {real_notes[audio_name]['notes']}\n")
    print(f"real note count: {fretboard_real.shape[-1]}")
    print("\n Real Frets")
    for idx, fret in enumerate(real_notes[audio_name]['frets']):
        fretboard_real[fret[0] - 1][idx] = fret[-1]
    for row in fretboard_real:
        print(format_row(row))
    with open(boardpath / "fretboard_real.txt", "w") as f:
        for row in fretboard_real:
            f.write(format_row(row) + "\n")
except KeyError:
    print("\n No real frets available for this audio file.")


print()
for (string, fret), col in zip(frets, playable_idx):
    fretboard_str[string - 1][col] = fret
print("DP Weighted Frets Estimate (minimized total distance with urgency weight)")
for row in fretboard_str:
    print(format_row(row))
with open(boardpath / "fretboard.txt", "w") as f:
    for row in fretboard_str:
        f.write(format_row(row) + "\n")

for (string, fret), col in zip(frets2, playable_idx):
    fretboard_str2[string - 1][col] = fret
with open(boardpath / "fretboard_dirr.txt", "w") as f:
    for row in fretboard_str2:
        f.write(format_row(row) + "\n")
print("\n DP Weighted Frets + Direction Weight Estimate")
for row in fretboard_str2:
    print(format_row(row))
try:
    note_check = check(real_notes[audio_name]["notes"], note_trim)
    lcs = note_check["lcs"]
    n_det = note_check["n_det"]
    n_exp = note_check["n_exp"]
    print(score(lcs, n_det, n_exp))
    recall, precision, F1 = (pipeline(file_paths, real_notes))
    print(f"Recall: {recall} ")
    print(f"Precision: {precision}")
    print(f"F1: {F1}")
except KeyError:
    print("No Truth Values found")
# spectogram, waveform, and onset envelope display

fig, ax = plt.subplots(nrows=3, sharex=True, height_ratios=(1, 3, 1))
librosa.display.waveshow(y=y, sr=sr, ax=ax[0], label="Waveform")
img = librosa.display.specshow(stft, vscale="dBFS", x_axis="time", y_axis="log", ax=ax[2], sr=sr)
librosa.display.colorbar_db(img, label="dBFS")
times_env = librosa.times_like(onset_env, sr=sr)
ax[1].plot(times_env, onset_env, label="Onset envelope", color="C1")
ax[1].scatter(times_env[onset_peaks], onset_env[onset_peaks], color="k", marker="^", label="Peaks")
ax[1].scatter(times_env[onset_detect], onset_env[onset_detect], edgecolor="C2", marker="o", label="Onsets")
ax[1].legend()
ax[0].legend()
ax[0].label_outer()
ax[1].label_outer()
plt.show()


# librosa pyin/yin f0 estimate and spectrogram display
"""
fig, ax = plt.subplots()
librosa.display.specshow(stft, vscale="dBFS",
                         x_axis="time", y_axis="log", ax=ax)
hl = librosa.display.highlight(ax=ax, alpha=0.8, linewidth=4)
ax.plot(times, pyin_f0, label="pyin f0 estimate", path_effects=hl)
ax.legend(loc="upper right")
plt.show()
"""

