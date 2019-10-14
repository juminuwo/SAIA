import os
import subprocess
from glob import glob

import numpy as np
from natsort import natsorted

import nn
from utils.feature_extract import music_features
from utils.sm import sm


class data():
    def __init__(self, input_sm, input_song):
        try:
            subprocess.check_output([
                'sox', '-v', '0.98', input_song, 'output.wav', 'remix', "1,2"
            ])
        except:
            subprocess.call(['cp', input_song, 'output.wav'])
        self.s = sm(input_sm)

    def generate_data(self, n_chart):
        self.s.load_chart(n_chart)
        self.s.generate_data()
        self.output_data = self.s.output_data
        m_f = music_features('output.wav',
                             bpm_overwrite=self.output_data.shape[1],
                             offset=self.s.offset)
        m_f.generate_data()
        m_f.input_data = np.vstack([
            m_f.input_data,
            np.full((m_f.input_data.shape[1], ), self.s.numeric_meter)
        ])
        m_f.input_data = np.vstack([
            m_f.input_data,
            np.arange(m_f.input_data.shape[1])
        ])
        self.input_data = m_f.input_data


def get_files(songs_dir):
    sm_files = glob(os.path.join(songs_dir, '*', '*', '*.sm'))
    types = ('*.mp3', '*.ogg', '*.wav')
    song_files = []
    for t in types:
        song_files += glob(os.path.join(songs_dir, '*', '*', t))

    sm_files = natsorted(sm_files)
    song_files = natsorted(song_files)

    sm_dirname = []
    song_dirname = []
    for s, s_ in zip(sm_files, song_files):
        sm_dirname.append(os.path.dirname(s))
        song_dirname.append(os.path.dirname(s_))

    song_count = np.unique(song_dirname, return_counts=True)
    song_count = song_count[0][song_count[1] == 2]
    sm_count = np.unique(sm_dirname, return_counts=True)
    sm_count = sm_count[0][sm_count[1] == 2]

    assert len(song_count) == 0, 'Multiple song files found in {}.'.format(
        song_count)
    assert len(sm_count) == 0, 'Multiple sm file found in {}.'.format(sm_count)

    for s, s_ in zip(sm_files, song_files):
        assert os.path.dirname(s) == os.path.dirname(
            s_), 'Error, missing files. Try looking in: {1}, {2}'.format(
                s, s_)
    return sm_files, song_files


class dataset():
    def __init__(self, songs_dir):
        self.sm_files, self.song_files = get_files(songs_dir)
        self.input_list = np.array([])
        self.output_list = np.array([])
        for sm_file, song_file in zip(self.sm_files, self.song_files):
            print('LOADING sm file: {}'.format(sm_file))
            print('LOADING song file: {}'.format(song_file))
            try:
                d = data(sm_file, song_file)
            except:
                print('ERROR READING SM FILE, THUS SKIPPED.')
                continue
            for n_chart in range(d.s.n_charts):
                try:
                    d.generate_data(n_chart)
                except:
                    print('ERROR READING SONG FILE, THUS SKIPPED.')
                    break  # go back to parent loop
                if self.input_list.size:
                    self.input_list = np.column_stack(
                        [self.input_list, d.input_data])
                else:
                    self.input_list = d.input_data
                if self.output_list.size:
                    self.output_list = np.column_stack(
                        [self.output_list, d.output_data])
                else:
                    self.output_list = d.output_data

    @staticmethod
    def train_test_split(input_list, output_list, test, val=False):
        '''
        Makes a train/test split by test value, then splits the test set by val
        value.
        '''

        assert len(input_list) == len(output_list)
        length = input_list.shape[1]
        arr = np.arange(length)
        np.random.shuffle(arr)

        def split(arr, split):
            length = len(arr)
            split = int(length * split)
            return arr[:split], arr[split:]

        train_ind, test_ind = split(arr, test)

        def select_ind(l, ind):
            return l[:, ind]

        if val:
            test_ind, val_ind = split(test_ind, val)
            X_train = select_ind(input_list, train_ind)
            X_test = select_ind(input_list, test_ind)
            y_train = select_ind(output_list, train_ind)
            y_test = select_ind(output_list, test_ind)
            X_val = select_ind(input_list, val_ind)
            y_val = select_ind(output_list, val_ind)
            return X_train, X_test, y_train, y_test, (X_val, y_val)
        else:
            X_train = select_ind(input_list, train_ind)
            X_test = select_ind(input_list, test_ind)
            y_train = select_ind(output_list, train_ind)
            y_test = select_ind(output_list, test_ind)
            return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    sm_file = "/media/adrian/Main/Games/StepMania 5/test_packs/You're Streaming Forever/Block Control VIP/Block Control VIP.sm"
    song = '/media/adrian/Main/Games/StepMania 5/train_packs/Cirque du Zonda/Zaia - Apocynthion Drive/HertzDevil - Apocynthion Drive.ogg'
    song = 'shihen.ogg'
    d = data(sm_file, song)
    try:
        d.generate_data(d.s.n_charts - 1)
    except:
        'die'

    songs_dir = '/media/adrian/Main/Games/StepMania 5/train_packs/'
    d = dataset(songs_dir)
    X_train, X_test, y_train, y_test = d.train_test_split(
        d.input_list, d.output_list, 0.9)

    nn_model = nn.nn()
    nn_model.create_model(len(d.input_list[0]), len(d.output_list[0]))
    nn_model.train(X_train,
                   y_train,
                   save_path='backup',
                   batch_size=2,
                   epochs=150,
                   validation_data=(X_test, y_test))

