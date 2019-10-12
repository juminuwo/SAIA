import pandas as pd
import numpy as np

class sm():
    def __init__(self, sm_file):
        '''
        load info from chart, including the chart itself
        ignoring BPMS
        ignoring STOPS
        '''
        f = open(sm_file, 'r')
        sm = f.read().splitlines()
        sm = pd.Series(sm)

        m = sm.str.contains('#')
        meta = sm[m]

        # meta (data before notes)
        self.meta_dict = {}
        for me in meta:
            data = me.split(':')
            self.meta_dict.update({data[0]: data[1]})

        # note values
        n = sm.str.contains('#NOTES')
        n = np.where(n == True)[0]
        assert len(n) == 1, 'Multiple #NOTES appear in sm file.'
        n = n[0]
        dance_type = sm[n + 1]
        assert 'dance-single' in dance_type, 'Not a singles chart.'
        self.numeric_meter = sm[n + 4]

        # note chart
        chart = sm[n + 6:]
        self.chart = chart[chart != '']


if __name__ == '__main__':
    sm_file = '/media/adrian/Main/Games/StepMania 5/Songs/GIRLS/Idol - [Mad Matt]/Idol.sm'
    s = sm(sm_file)
