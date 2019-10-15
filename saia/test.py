import unittest
import os

import train


class test(unittest.TestCase):
    def test_feature_gettin(self):
        for input_song, input_sm in zip(['shihen.wav', '30min.wav'],
                                        ['shihen.sm', '30min.sm']):
            d = train.data(input_sm, input_song)
            print('Number of measures: {}'.format(len(d.input_data)))
            print('Output len: {}'.format(len(d.output_data[0])))
            print('Input len: {}'.format(len(d.input_data[0])))
            self.assertEqual(len(d.output_data), len(d.input_data))

    def test_getting_song_sm_lists(self):
        songs_dir = '/media/adrian/Main/Games/StepMania 5/training_packs/'
        sm_files, song_files = train.get_files(songs_dir)
        for sm_file, song_file in zip(sm_files, song_files):
            self.assertEqual(os.path.dirname(sm_file), os.path.dirname(song_file))


if __name__ == '__main__':
    unittest.main()
