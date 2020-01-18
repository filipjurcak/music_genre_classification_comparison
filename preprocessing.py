import os
import csv
import librosa
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


def save_features_csv(header, genres, dataset, csv_file, normalized_csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

    for g in genres:
        genre_dir = '{}/genres/{}'.format(dataset, g)
        for filename in [subdir for subdir in os.listdir(genre_dir) if not subdir.startswith('.')]:
            song_name = genre_dir + '/' + filename
            y, sr = librosa.load(song_name, mono=True)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            row = [filename, np.mean(chroma_stft), np.mean(spectral_centroid), np.mean(spectral_bandwidth),
                   np.mean(rolloff), np.mean(zero_crossing_rate)]
            for f in mfcc:
                row.append(np.mean(f))
            row.append(g)
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)

    # normalize data
    data_csv = pd.read_csv(csv_file)
    csv_normalized = np.array(data_csv)
    features = np.array(csv_normalized[:, 1:-1], dtype=float)
    means_np = np.array(data_csv.mean())
    stds_np = np.array(data_csv.std())
    csv_normalized[:, 1:-1] = (features - means_np) / stds_np

    with open(normalized_csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
    with open(normalized_csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_normalized)


def splitsongs(X, y, window=0.1, overlap=0.5):
    # Empty lists to hold our results
    temp_X = []
    temp_y = []

    # Get the input song array size
    xshape = X.shape[0]
    chunk = int(xshape * window)
    offset = int(chunk * (1. - overlap))

    # Split the song and create new ones on windows
    spsong = [X[i:i + chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsong:
        temp_X.append(s)
        temp_y.append(y)

    return np.array(temp_X), np.array(temp_y)


def to_melspectrogram(songs, n_fft=1024, hop_length=512):
    # Transformation function
    melspec = lambda x: librosa.feature.melspectrogram(x, n_fft=n_fft, hop_length=hop_length)[:, :, np.newaxis]

    # map transformation of input songs to melspectrogram using log-scale
    tsongs = map(melspec, songs)
    return np.array(list(tsongs))


def create_melspectogram_data(genres, source_dir, song_samples):
    arr_specs = []
    arr_genres = []

    # Read files from the folders
    for index, (x, _) in enumerate(genres.items()):
        print(x)
        folder = source_dir + x

        for root, subdirs, files in os.walk(folder):
            for idx, file in enumerate([f for f in files if not f.startswith('.')]):
                print((index * 1000) + idx, file)
                # Read the audio file
                file_name = folder + "/" + file
                signal, sr = librosa.load(file_name)
                signal = signal[:song_samples]

                # Convert to dataset of spectograms/melspectograms
                signals, y = splitsongs(signal, genres[x])

                # Convert to "spec" representation
                specs = to_melspectrogram(signals)

                # Save files
                arr_genres.extend(y)
                arr_specs.extend(specs)
                print()

    return np.array(arr_specs), np.array(arr_genres)


if __name__ == "__main__":
    samples = 660000

    header = ['filename', 'chroma_stft', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate']
    for i in range(1, 21):
        header.append('mfcc{}'.format(i))
    header.append('label')

    # gtzan csv data
    gtzan_csv_file = 'gtzan/data.csv'
    normalized_gtzan_csv_file = 'gtzan/normalized_data.csv'
    gtzan_genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    save_features_csv(header, gtzan_genres, 'gtzan', gtzan_csv_file, normalized_gtzan_csv_file)

    # gtzan melspectograms
    gtzan_source_dir = 'gtzan/genres/'
    gtzan_genres_dict = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4,
                         'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}
    gtzan_melspectogram, gtzan_targets = create_melspectogram_data(gtzan_genres_dict, gtzan_source_dir, samples)
    np.save('x_gtzan_npy.npy', gtzan_melspectogram)
    np.save('y_gtzan_npy.npy', gtzan_targets)

    # fma_small csv data
    fma_small_csv_file = 'fma_small/data.csv'
    normalized_fma_small_csv_file = 'fma_small/normalized_data.csv'
    fma_small_genres = ['electronic', 'experimental', 'folk', 'hip-hop', 'instrumental', 'international', 'pop', 'rock']
    save_features_csv(header, fma_small_genres, 'fma_small', fma_small_csv_file, normalized_fma_small_csv_file)

    # fma_small melspectograms
    fma_small_source_dir = 'fma_small/genres/'
    fma_small_genres_dict = {'electronic': 0, 'experimental': 1, 'folk': 2, 'hip-hop': 3,
                             'instrumental': 4, 'international': 5, 'pop': 6, 'rock': 7}
    fma_small_melspectogram, fma_small_targets = create_melspectogram_data(fma_small_genres_dict,
                                                                           fma_small_source_dir, samples)
    np.save('x_fma_small.npy', fma_small_melspectogram)
    np.save('y_fma_small.npy', fma_small_targets)


