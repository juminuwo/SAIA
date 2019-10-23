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

    def generate_data(self, n_chart, max_=False):
        self.s.load_chart(n_chart)
        self.s.generate_data(max_=max_)
        self.output_data = self.s.output_data
        m_f = music_features('output.wav',
                             bpm_overwrite=self.output_data.shape[1],
                             offset=self.s.offset)
        m_f.generate_data()
        m_f.input_data = np.vstack([
            m_f.input_data,
            np.full((m_f.input_data.shape[1], ), self.s.numeric_meter)
        ])
        m_f.input_data = np.vstack(
            [m_f.input_data,
             np.arange(m_f.input_data.shape[1])])

        assert m_f.input_data.shape[1] == self.output_data.shape[1]
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
                    d.generate_data(n_chart, max_=True)
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
        self.input_list = self.input_list.transpose()
        self.output_list = self.output_list.transpose()


def train_test_split(input_list, output_list, test, val=False):
    '''
    Makes a train/test split by test value, then splits the test set by val
    value.
    '''

    assert input_list.shape[0] == output_list.shape[0]
    length = input_list.shape[0]
    arr = np.arange(length)
    np.random.shuffle(arr)

    def split(arr, split):
        length = len(arr)
        split = int(length * split)
        return arr[:split], arr[split:]

    train_ind, test_ind = split(arr, test)

    def select_ind(l, ind):
        return l[ind, :]

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
    # sm_file = "/media/adrian/Main/Games/StepMania 5/test_packs/You're Streaming Forever/Block Control VIP/Block Control VIP.sm"
    # song = '/media/adrian/Main/Games/StepMania 5/train_packs/Cirque du Zonda/Zaia - Apocynthion Drive/HertzDevil - Apocynthion Drive.ogg'
    # song = 'shihen.ogg'
    # d = data(sm_file, song)
    # d.generate_data(d.s.n_charts - 1)

    #songs_dir = '/media/adrian/Main/Games/StepMania 5/train_packs/'
    #d = dataset(songs_dir)
    #np.save('output_list.npy', d.output_list)
    #np.save('input_list.npy', d.input_list)

    # ---

    output_list = np.load('output_list.npy')
    input_list = np.load('input_list.npy')
    X_train, X_test, y_train, y_test = train_test_split(
        input_list, output_list, 0.9)
    del output_list
    del input_list
    #import keras
    from keras.models import Sequential
    from keras.layers import Dense, Activation

    model = Sequential()
    model.add(Dense(32, activation='relu', input=X_train.shape[1]))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(X_train[0:1000, :], y_train[0:1000, :], epochs=10, batch_size=2)

