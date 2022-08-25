# --- Arrangming Pre-Sliced Audio ---
import numpy as np
import random
import os
import librosa
import librosa.display
import soundfile as sf
import argparse

parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--folder', type=str, help='the song you wish to Bach-ify', required=True)
# Parse the argument
args = parser.parse_args()
# Print arguemnt
print("Bach-ify", args.folder)
#print (args.folder + ".wav")

#import the audio file
audio_file = (args.folder)

#-- getting a list of files in a dir --
path_Main= (args.folder)
Song_Title= (args.folder)

path_to_A= (path_Main + "/A" )
path_to_B= (path_Main + "/B" )
path_to_C= (path_Main + "/C" )
path_to_D= (path_Main + "/D" )
#where are the files? (slashes /)

dir_A = os.listdir(path_to_A)
dir_B = os.listdir(path_to_B)
dir_C = os.listdir(path_to_C)
dir_D = os.listdir(path_to_D)
#call the new list you want something
#os.listdir(path) telling where and what to fill the list with

#print (dir_list)

#-- pick a random file from each folder --
for i in range(1):
    subject = random.choice(dir_A)
    print ("subject:", subject)
#randomly pick one audio file from the folder stated
#call this file subject
subject_path = (path_to_A + "/" + subject)
#make the path to the file to call later
#print (subject_path)

for i in range(1):
    counter = random.choice(dir_B)
    print ("counter:", counter)
counter_path = (path_to_B + "/" + counter)
#print (counter_path)

for i in range(1):
    answer = random.choice(dir_C)
    print ("answer/counter 2:", answer)
answer_path = (path_to_C + "/" + answer)
#print (answer_path)

for i in range (1):
    free = random.choice(dir_D)
    print("free:", free)
free_path = (path_to_D + "/" + free)
#print (free_path)

# -- beat track onset detection from librosa --

y, sr = librosa.load(subject_path)
#load in the audio
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
#using librosa pakage fidn the tempo and the beats
librosa.frames_to_time(beats, sr=sr)
beats_Aos = librosa.frames_to_time(beats, sr=sr)
#change into a usable format
round_onset = [round(num, 3) for num in beats_Aos]
A_os= round_onset[2]
#round to a usaable numer
#print (A_os)

y, sr = librosa.load(counter_path)
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
librosa.frames_to_time(beats, sr=sr)
beats_Bos = librosa.frames_to_time(beats, sr=sr)
round_onset = [round(num, 3) for num in beats_Bos]
B_os= round_onset[2]
#print (B_os)

y, sr = librosa.load(answer_path)
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
librosa.frames_to_time(beats, sr=sr)
beats_Cos = librosa.frames_to_time(beats, sr=sr)
round_onset = [round(num, 3) for num in beats_Cos]
C_os= round_onset[2]
#print (C_os)

y, sr = librosa.load(free_path)
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
librosa.frames_to_time(beats, sr=sr)
beats_Dos = librosa.frames_to_time(beats, sr=sr)
round_onset = [round(num, 3) for num in beats_Dos]
D_os= round_onset[2]
#print (D_os)

# -- make the files in the right format using Pydub --
from pydub import AudioSegment

audA = AudioSegment.from_file(subject_path, format="wav", start_second=A_os)
audB = AudioSegment.from_file(counter_path, format="wav", start_second=B_os)
audC = AudioSegment.from_file(answer_path, format="wav", start_second=C_os)
audD = AudioSegment.from_file(free_path, format="wav", start_second=D_os)
#where is the file, what format is the file, start point of the file (to start on beat)

# -export a copy of each file to allow for key change -
file_handle = audA.export(path_Main + "/audA_V.wav", format="wav")
file_handle = audB.export(path_Main + "/audB_V.wav", format="wav")
file_handle = audC.export(path_Main + "/audC_V.wav", format="wav")
file_handle = audD.export(path_Main + "/audD_V.wav", format="wav")

# - make a path to Key change items -
audA_V_path= (path_Main + "/" + "audA_V.wav")
audB_V_path= (path_Main + "/" + "audB_V.wav")
audC_V_path= (path_Main + "/" + "audC_V.wav")
audD_V_path= (path_Main + "/" + "audD_V.wav")

audA_V = audA_V_path
audB_V = audB_V_path
audC_V = audC_V_path
audD_V = audD_V_path
#print ("audA:", type(audA))
#print ("audA_V:", type(audA_V))

#make one louder so then we have volume control
#louder = audA + 6

# - change the pitch -

sampling_rate= 48000
y, sr = librosa.load(audA_V, sr=sampling_rate) # y is a numpy array of the wav file, sr = sample rate
#load the aduio file with the sampling rate as a numpy array
y_shifted = librosa.effects.pitch_shift(y, sr, n_steps=-7, bins_per_octave=12)
# shifted by n-steps (half steps)
sf.write(audA_V, y_shifted, sr)
#write out the file overtop

y, sr = librosa.load(audB_V, sr=sampling_rate)
y_shifted = librosa.effects.pitch_shift(y, sr, n_steps=-7, bins_per_octave=12)
sf.write(audB_V, y_shifted, sr)

y, sr = librosa.load(audC_V, sr=sampling_rate)
y_shifted = librosa.effects.pitch_shift(y, sr, n_steps=-7, bins_per_octave=12)
sf.write(audC_V, y_shifted, sr)

y, sr = librosa.load(audD_V, sr=sampling_rate)
y_shifted = librosa.effects.pitch_shift(y, sr, n_steps=-7, bins_per_octave=12)
sf.write(audD_V, y_shifted, sr)

# - change audio file type to overlap-

audA_V= AudioSegment.from_file(audA_V_path, format="wav", start_second=A_os)
audB_V= AudioSegment.from_file(audB_V_path, format="wav", start_second=B_os)
audC_V= AudioSegment.from_file(audC_V_path, format="wav", start_second=C_os)
audD_V= AudioSegment.from_file(audD_V_path, format="wav", start_second=D_os)
#where is the file, what format is the file, start point of the file (to start on beat)

# -Combining and adding fades -
#A_FadeIn = audA.fade(from_gain= -24.0, start=0, duration=5000)
A_FadeIn = audA.fade_in(duration=3000)
#Audio A + a fade in (to start the track)

A_FadeOut = audA.fade(from_gain= -24.0, start = 0, duration =150)
B_FadeOut = audB.fade(from_gain= -24.0, start = 0, duration =150)
C_FadeOut = audC.fade(from_gain= -24.0, start = 0, duration =150)
D_FadeOut = audD.fade(from_gain= -24.0, start = 0, duration =150)
#Audio D/B/C + fade out (to end the track, depending what you need)

ABFadeOut= A_FadeOut.overlay(B_FadeOut)
ABC_FadeOut = ABFadeOut.overlay(C_FadeOut)

AB = audB.overlay(audA)
# audio B and A play at the same time (will keep playing until longer clip done)

AB_V = audB_V.overlay(audA_V)
#Dom key of AB overlay

ABC = AB.overlay(audC)
# audio a b c plaay together

ABFade = audA.append(AB)
# name = start-sound.append(next-sound) this way makes a defult 100 ms fade for no clicks

# - arrange files in the order needed (must start and end with a #_FadeIn/Out file)
# audX(audY, position=# * len(audX))

Bach866 = A_FadeIn + AB_V + ABC*2 + audD + D_FadeOut
Bach848= A_FadeIn + AB_V + ABC + audD*2 + ABC + audD_V + AB + D_FadeOut
Bach847 = A_FadeIn + AB + audD + ABC + audD + ABC + audD + ABC_FadeOut
Bach852= A_FadeIn + audD_V + AB + audD + AB + audD + AB + D_FadeOut

# -- export the files with a new name and type --
file_handle = Bach866.export(path_Main + "/BachFugue866_" + Song_Title + ".wav", format="wav")
file_handle = Bach848.export(path_Main + "/BachFugue848_" + Song_Title + ".wav", format="wav")
file_handle = Bach847.export(path_Main + "/BachFugue847_" + Song_Title + ".wav", format="wav")
file_handle = Bach852.export(path_Main + "/BachFugue852_" + Song_Title + ".wav", format="wav")
