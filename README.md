# FretPath

Monophonic Audio-to-tablature tool for guitar. Detects notes from a recording of guitar 
and estimates the optimal fret position of each note using a dynamic-programming optimizer

## How it Works
1. **Pitch Tracking:** Utilizes Librosa's pYIN to estimate the pitch of each frame
2. **Onset Detection:** Utilizes onset_detection to detect when a note starts by detecting change in the spectrum
3. **Note Segmentation:** Takes the most common notes between each onset to estimate the note played
4. **Fret Optimizer:** Estimates optimal fret position by utilizing timing, minimal total distance, and a direction-movement weight.
5. **Display:** Prints out the real frets and notes to the song along with both fret estimates using both algorithms to compare which is better for that song 

When running program, it displays the fretboards for only one song. To change the song that is displayed, change the audio_name variable at the top of the fretpath.py file to the key value of the song you want

## Results
Evaluated accuracy by comparing detected note sequences with the expected notes and taking the longest common subsequence between the two. 
The accuracy is calculated after pooling all notes in each song

Achieved a 89% F1 across 15 monophonic audio recordings (not all are included in the github repository for copyright reasons)

### Example

guitarTest7.mp3

```
Real Frets
 -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  1  2  3  4
 -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  1  2  3  4  -  -  -  -
 -  -  -  -  -  -  -  -  -  -  -  -  1  2  3  4  -  -  -  -  -  -  -  -
 -  -  -  -  -  -  -  -  1  2  3  4  -  -  -  -  -  -  -  -  -  -  -  -
 -  -  -  -  1  2  3  4  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
 1  2  3  4  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -

DP Weighted Frets Estimate (minimized total distance with urgency weight)
 -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
 -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  6  7  8  9
 -  -  -  -  -  -  -  -  -  -  -  -  1  2  3  4  5  6  7  8  -  -  -  -
 -  -  -  -  -  -  -  -  -  -  -  4  -  -  -  -  -  -  -  -  -  -  -  -
 -  -  -  -  1  2  3  4  5  6  7  -  -  -  -  -  -  -  -  -  -  -  -  -
 1  2  3  4  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -

 DP Weighted Frets + Direction Weight Estimate
 -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
 -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  6  7  8  9
 -  -  -  -  -  -  -  -  -  -  -  -  1  2  3  4  5  6  7  8  -  -  -  -
 -  -  -  -  -  -  -  -  -  -  -  4  -  -  -  -  -  -  -  -  -  -  -  -
 -  -  -  -  1  2  3  4  5  6  7  -  -  -  -  -  -  -  -  -  -  -  -  -
 1  2  3  4  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
```

## Important Notes

### Included Audio Files
**The original program has 7 audio files that are not included in the github repository to prevent any possible copyright issues in the future. The 8 audios available for use are**
- guitarTest1.mp3
- guitarTest2.mp3
- guitarTest3.mp3
- guitarTest4.mp3
- guitarTest5.mp3
- guitarTest6.mp3
- guitarTest7.mp3
- realTest.mp3

### Limitations
As of now the program only accepts monophonic guitar audios and works best for clean recordings. The program also does not transcribe guitar techniques like sliding, bending, or hammer-ons as of now.

### Adding more audios
If you want to add your own audio files for further testing, you would need to record an audio of guitar (preferably an original audio on a virtual guitar), then hand transcribe each real note and fret position in the real_notes.json file where the first index is the string (1-6, from high e to low E) and the second index is the fret on the string. After putting the audio in the audio folder and hand transcribing the necessary data, you would then need to add the filename of the audio into the file_paths dictionary in the fretpath.py file

**EXAMPLE**
```
"guitarTest6": {
    "notes": ["C4", "C#4", "D4", "D#4", "C4", "C4", "C4", "C4", "C4", "C4", "G3", "D#3"],\
    "frets": [[2,1], [2,2], [2,3], [2,4], [2,1], [2,1], [2,1], [2,1], [2,1], [2,1], [3,0], [4,1]]
}
```