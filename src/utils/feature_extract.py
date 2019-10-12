from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction as aF
import math

class music_features():
    def __init__(self, song):
        [Fs, x] = audioBasicIO.readAudioFile(song)

        step = 0.050 # ms
        self.F, _ = aF.stFeatureExtraction(
            x, Fs, 0.050 * Fs, step * Fs)
        bpm, _ = aF.beatExtraction(self.F, 0.050)

        beat = 60/bpm # how long a beat lasts

        self.m_step = int(round(beat/step)) * 4 # how many steps (columns of features) in a measure

    def generate_data(self):
        self.input_data = []
        for i in range(math.ceil(self.F.shape[1] / self.m_step)):
            self.input_data.append(self.F[:, self.m_step*(i):self.m_step*(i+1)])

if __name__ == '__main__':
    song = "shihen.wav"
    m_f = music_features(song)
    m_f.generate_data()