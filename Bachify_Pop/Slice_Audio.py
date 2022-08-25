# to run in anaconda: python Slice_Audio.py --song song_name
# -- import --
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sklearn.cluster
import librosa
import librosa.display
import argparse
#print ("import done")

parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--song', type=str, help='the song you wish to Bach-ify', required=True)
# Parse the argument
args = parser.parse_args()
# Print arguemnt
print("Bach-ify", args.song)
print (args.song + ".wav")

#import the audio file
audio_file = (args.song + ".wav")

y, sr = librosa.load(audio_file)
#librosa.get_duration(y=y, sr=sr)
print ("audio loaded")

# -- Tempo detection --
librosa.get_duration(y=y, sr=sr)
durationcalc = librosa.get_duration(y=y, sr=sr)
print (durationcalc)
y, sr = librosa.load((audio_file), duration=durationcalc)
onset_env = librosa.onset.onset_strength(y=y, sr=sr)
tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
#find the tempo using onset detection

round_tempo = [round(num) for num in tempo]
print("tempo", round_tempo)
#round tempo to a usable number

# -- Lapalacian Segmentation -- code from Librosa
# Code source: Brian McFee
# License: ISC

#Log Power CQT
BINS_PER_OCTAVE = 12 * 1
#12*3 is what is used orionally, too accuarte for what is needed
N_OCTAVES = 7
C = librosa.amplitude_to_db(np.abs(librosa.cqt(y=y, sr=sr,
                                        bins_per_octave=BINS_PER_OCTAVE,
                                        n_bins=N_OCTAVES * BINS_PER_OCTAVE)),
                            ref=np.max)

fig, ax = plt.subplots()
librosa.display.specshow(C, y_axis='cqt_hz', sr=sr,
                         bins_per_octave=BINS_PER_OCTAVE,
                         x_axis='time', ax=ax)

print ("Power CQT done")

#Reduce dimentionality
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
Csync = librosa.util.sync(C, beats, aggregate=np.median)
# For plotting purposes, we'll need the timing of the beats
# we fix_frames to include non-beat frames 0 and C.shape[1] (final frame)
beat_times = librosa.frames_to_time(librosa.util.fix_frames(beats,
                                                            x_min=0),
                                    sr=sr)

fig, ax = plt.subplots()
librosa.display.specshow(Csync, bins_per_octave=12*3,
                         y_axis='cqt_hz', x_axis='time',
                         x_coords=beat_times, ax=ax)
print ("redudcing done")

#weighted reccurence matrix
R = librosa.segment.recurrence_matrix(Csync, width=3, mode='affinity',
                                      sym=True)

# Enhance diagonals with a median filter (Equation 2)
df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
Rf = df(R, size=(1, 7))
print ("wrm done")

#median distance
mfcc = librosa.feature.mfcc(y=y, sr=sr)
Msync = librosa.util.sync(mfcc, beats)
path_distance = np.sum(np.diff(Msync, axis=1)**2, axis=0)
sigma = np.median(path_distance)
path_sim = np.exp(-path_distance / sigma)

R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)
print ("median distance done")

#Conversion and maths
deg_path = np.sum(R_path, axis=1)
deg_rec = np.sum(Rf, axis=1)

mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)

A = mu * Rf + (1 - mu) * R_path
print ("maths done")

#plot graphs
fig, ax = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(10, 4))
librosa.display.specshow(Rf, cmap='inferno_r', y_axis='time', x_axis='s',
                         y_coords=beat_times, x_coords=beat_times, ax=ax[0])
ax[0].set(title='Recurrence similarity')
ax[0].label_outer()
librosa.display.specshow(R_path, cmap='inferno_r', y_axis='time', x_axis='s',
                         y_coords=beat_times, x_coords=beat_times, ax=ax[1])
ax[1].set(title='Path similarity')
ax[1].label_outer()
librosa.display.specshow(A, cmap='inferno_r', y_axis='time', x_axis='s',
                         y_coords=beat_times, x_coords=beat_times, ax=ax[2])
ax[2].set(title='Combined graph')
ax[2].label_outer()
print("Recuurence, Path and Combo Graphs done")

# -compute Lapalacian equation-
L = scipy.sparse.csgraph.laplacian(A, normed=True)
# and its spectral decomposition
evals, evecs = scipy.linalg.eigh(L)

# We can clean this up further with a median filter.
# This can help smooth over small discontinuities
evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))

# cumulative normalization is needed for symmetric normalize laplacian eigenvectors
Cnorm = np.cumsum(evecs**2, axis=1)**0.5
# If we want k clusters, use the first k normalized eigenvectors.
# Fun exercise: see how the segmentation changes as you vary k
k = 5
X = evecs[:, :k] / Cnorm[:, k-1:k]

# - Plot the resulting representation (Figure 1, center and right) -

fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(10, 5))
librosa.display.specshow(Rf, cmap='inferno_r', y_axis='time', x_axis='time',
                         y_coords=beat_times, x_coords=beat_times, ax=ax[1])
ax[1].set(title='Recurrence similarity')
ax[1].label_outer()

librosa.display.specshow(X,
                         y_axis='time',
                         y_coords=beat_times, ax=ax[0])
ax[0].set(title='Structure components')
print("graphs 4 and 5")

# - cluster compnets with K values-
KM = sklearn.cluster.KMeans(n_clusters=k)

seg_ids = KM.fit_predict(X)

# - plot the results -
fig, ax = plt.subplots(ncols=3, sharey=True, figsize=(10, 4))
colors = plt.get_cmap('Paired', k)

librosa.display.specshow(Rf, cmap='inferno_r', y_axis='time',
                         y_coords=beat_times, ax=ax[1])
ax[1].set(title='Recurrence matrix')
ax[1].label_outer()

librosa.display.specshow(X,
                         y_axis='time',
                         y_coords=beat_times, ax=ax[0])
ax[0].set(title='Structure components')

img = librosa.display.specshow(np.atleast_2d(seg_ids).T, cmap=colors,
                         y_axis='time',
                         x_coords=[0, 1], y_coords=list(beat_times) + [beat_times[-1]],
                         ax=ax[2])
ax[2].set(title='Estimated labels')

ax[2].label_outer()
fig.colorbar(img, ax=[ax[2]], ticks=range(k))
print ("K Values and Recurrence Matrix calculated and ploted")

# -locate segmented boundaries -
bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])

# Count beat 0 as a boundary
bound_beats = librosa.util.fix_frames(bound_beats, x_min=0)

# Compute the segment label for each boundary
bound_segs = list(seg_ids[bound_beats])

# Convert beat indices to frames
bound_frames = beats[bound_beats]

# Make sure we cover to the end of the track
bound_frames = librosa.util.fix_frames(bound_frames,
                                       x_min=None,
                                       x_max=C.shape[1]-1)
print ("boundry location done")

# -plot the final graph with segments and coloured layers -
import matplotlib.patches as patches

bound_times = librosa.frames_to_time(bound_frames)
new_bound_times = librosa.frames_to_time(bound_frames)
#new_bound_times are the start and end points of segmentation before rounding

freqs = librosa.cqt_frequencies(n_bins=C.shape[0],
                                fmin=librosa.note_to_hz('C1'),
                                bins_per_octave=BINS_PER_OCTAVE)


fig, ax = plt.subplots()
librosa.display.specshow(C, y_axis='cqt_hz', sr=sr,
                         bins_per_octave=BINS_PER_OCTAVE,
                         x_axis='time', ax=ax)

for interval, label in zip(zip(bound_times, bound_times[1:]), bound_segs):
    ax.add_patch(patches.Rectangle((interval[0], freqs[0]),
                                   interval[1] - interval[0],
                                   freqs[-1],
                                   facecolor=colors(label),
                                   alpha=0.50))
#plt.show()
#this will show all of the graphs not just the last one
print ("plot done")

# --Make the Data Usable--
round_seconds = [round(num, 1) for num in new_bound_times]
print(round_seconds)
#round the data to tenths to allow for slicing

a = round_seconds
#the name of the list to run the slicing


# -- Slice the file according to the chunks  --
#this uses sliceFromList, sliceDef, listDem and UseExample**
#all made by Martin, thank you Martin Parker

from sliceDef import sliceFromList
sliceFromList(audio_file,a)
#this will output the audio in the calculated chunks with the times from the chunks in the name
