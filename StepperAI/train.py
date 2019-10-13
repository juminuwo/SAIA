import subprocess

import nn
from utils.feature_extract import music_features
from utils.sm import sm


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
        m_f = music_features('output.wav',
                             bpm_overwrite=len(s.output_data),
                             offset=s.offset)
        m_f.generate_data()
        self.input_data = m_f.input_data


if __name__ == '__main__':
    input_song = 'shihen.wav'
    input_sm = 'shihen.sm'

    d = data(input_song, input_sm)
    assert len(d.output_data) == len(d.input_data)
    print('Output len: {}'.format(len(d.output_data[0])))
    print('Input len: {}'.format(len(d.input_data[0])))
