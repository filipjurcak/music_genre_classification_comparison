import os
import numpy as np
import pandas as pd

if __name__ == "__main__":
    track_genre_csv = pd.read_csv('track_genre.csv')
    track_genre_csv.head()
    track_genre_np = np.array(track_genre_csv)

    os.mkdir('genres')
    genres = ['electronic', 'experimental', 'folk', 'hip-hop', 'instrumental', 'international', 'pop', 'rock']
    for g in genres:
        os.mkdir('genres/{}'.format(g))

    subdirs = os.listdir('data')
    file_no = 1
    for idx, subdir in enumerate([s for s in subdirs if not s.startswith('.')]):
        sub_dir_path = 'data/' + subdir
        for idz, audio_file in enumerate(os.listdir(sub_dir_path)):
            audio_file_id = int(audio_file[:-4])
            audio_file_path = sub_dir_path + '/' + audio_file
            for track in track_genre_np:
                track_id = int(track[0])
                if not isinstance(track[1], float):
                    track_genre = str.lower(track[1])
                    if track_id == audio_file_id:
                        new_location = 'genres/' + track_genre + '/' + audio_file
                        os.rename(audio_file_path, new_location)
                        print('Audio file no. {} moved successfully'.format(file_no))
                        file_no += 1
                        break
