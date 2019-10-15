import numpy as np
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction as aF
from pydub import AudioSegment


class music_features():
    def __init__(self, song, m_step=35, bpm_overwrite=False, offset=False):
        '''
        step is interval of when the get features through the song
        m_step is the number of steps per measure
        '''
        self.m_step = m_step
        self.bpm_overwrite = bpm_overwrite

        duration = AudioSegment.from_wav(song).duration_seconds
        if offset:
            duration = duration + offset
        [Fs, x] = audioBasicIO.readAudioFile(song)

        if self.bpm_overwrite:
            bpm = (self.bpm_overwrite * 4 * 60) / duration
            step = ((60 / bpm) * 4) / self.m_step
            self.get_features(x, Fs, offset, step)
        else:
            # does get_features twice
            # first to find bpm, to get step
            # second for actual features
            step = 0.050
            self.get_features(x, Fs, offset, step)
            bpm, _ = aF.beatExtraction(self.F, 0.050)
            step = ((60 / bpm) * 4) / self.m_step
            self.get_features(x, Fs, offset, step)

    def get_features(self, x, Fs, offset, step):
        '''
        adds silence to the beginning if offset makes chart start before song
        '''
        self.F, _ = aF.stFeatureExtraction(x, Fs, 0.050 * Fs, step * Fs)
        if offset:
            if offset == 0:
                pass
            elif offset < 0:
                offset = -round(offset / step)
                self.F = self.F[:, offset:]
            elif offset > 0:
                silence = np.load('silence.npy')
                offset = round(offset / step)
                silence = silence[:, :offset]
                self.F = np.concatenate([silence, self.F], axis=1)

    def generate_data(self):
        self.input_data = np.array([])

        for i in range(self.bpm_overwrite):
            i_1 = int(round(self.m_step * (i)))
            i_2 = int(round(self.m_step * (i + 1)))
            check = self.F.shape[1] - i_2
            if check < 0:
                i_1 = i_1 + check
                i_2 = i_2 + check
            data = self.F[:, i_1:i_2]
            
            assert data.shape[1] == self.m_step
            if self.input_data.size:
                self.input_data = np.column_stack([self.input_data, data.reshape(-1)])
            else:
                self.input_data = data.reshape(-1)

if __name__ == '__main__':
    song = "shihen.wav"
    m_f = music_features(song, bpm_overwrite=410)
    m_f.generate_data()
    print(len(m_f.input_data))
    print(len(m_f.input_data[0]))
    m_f = music_features(song)
    m_f.generate_data()
    print(len(m_f.input_data))
    print(len(m_f.input_data[0]))
