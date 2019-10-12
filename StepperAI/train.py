import subprocess

from utils.feature_extract import music_features
from utils.sm import sm

input_song = 'shihen.wav'
input_sm = 'shihen.sm'


class data():
    def __init__(self, input_song, input_sm):
        try:
            subprocess.check_output(
                ['sox', input_song, 'output.wav', 'remix', "'1,2'"])
        except:
            subprocess.call(['cp', input_song, 'output.wav'])
        s = sm(input_sm)
        s.load_chart(0)
        s.generate_data()
        self.output_data = s.output_data
        if s.bpm:
            m_f = music_features('output.wav',
                                 bpm_overwrite=len(self.output_data),
                                 offset=s.offset)
        else:
            m_f = music_features('output.wav', offset=s.offset)
        m_f.generate_data()
        self.input_data = m_f.input_data


if __name__ == '__main__':
    d = data(input_song, input_sm)
    print(len(d.output_data))
    print(len(d.input_data))