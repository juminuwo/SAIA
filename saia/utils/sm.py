import os
import itertools

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def permutate_step(hit=('0', '1', '2', '3', '4', 'M', 'K', 'L', 'F')):
    hit = np.repeat(hit, 4)
    hit = [comb for comb in itertools.combinations(hit, 4)]
    hit = list(set(hit))
    steps = []
    for h in hit:
        step = np.unique([''.join(p) for p in itertools.permutations(h)])
        steps += step.tolist()
    return steps

def label_encoder(save_loc='.',
                    hit=('0', '1', '2', '3', '4', 'M', 'K', 'L', 'F')):
    le = LabelEncoder()
    steps = permutate_step(hit)
    le.fit(steps)
    joblib.dump(le, os.path.join(save_loc, 'label_encoder.pickle'))

class sm():
    def __init__(self, sm_file):
        '''
        load info from chart, including the chart itself
        ignoring BPMS
        ignoring STOPS
        '''
        f = open(sm_file, 'r')
        sm = f.read().splitlines()
        self.sm = pd.Series(sm)

        m = self.sm.str.contains('#')
        meta = self.sm[m]

        # meta (data before notes)
        self.meta_dict = {}
        for me in meta:
            data = me.split(':')
            self.meta_dict.update({data[0]: data[1]})

        # get offset
        self.offset = float(self.meta_dict['#OFFSET'][:-1])

        # get indexs where chart starts/ends
        self.chart_index = self.sm[self.sm.str.contains(
            '//---')].index.tolist()
        self.chart_index.append(len(self.sm))

        # count of how many charts
        self.n_charts = len(self.chart_index) - 1

    def load_chart(self, n_chart):
        '''
        One sm file can contain many charts.
        Load by picking order
        n_chart: which chart to pick from sm file, index starts at 0
        '''
        assert n_chart < self.n_charts, 'n_chart input > n_charts'
        # note values
        sm = self.sm[self.chart_index[n_chart] + 1:self.chart_index[n_chart +
                                                                    1]]
        n = sm.str.contains('#NOTES')
        n = np.where(n == True)[0]
        assert len(n) == 1, 'Multiple #NOTES appear in sm file.'
        n = n[0]
        dance_type = sm.iloc[n + 1]
        assert 'dance-single' in dance_type, 'Not a singles chart.'
        self.numeric_meter = float(sm.iloc[n + 4][:-1])

        # note chart
        chart = sm[n + 6:]
        self.chart = chart[chart != '']
        self.chart = self.chart.loc[self.chart.str.startswith(
            ('0', '1', '2', '3', '4', 'M', 'K', 'L', 'F', ','))]

    @staticmethod
    def measure_to_data(measure, note_type=32, max_=False):
        measure_length = 192
        measure = measure[~measure.str.startswith(',')].reset_index(drop=True)
        measure_step = measure_length / len(measure)
        assert (measure_step).is_integer()
        out_measure = pd.Series(['0000' for _ in range(measure_length)])
        for n, m in enumerate(measure):
            out_measure[n * measure_step] = m

        if max_ == 'label_encoder':
            le = joblib.load('label_encoder.pickle')

        out_data = []
        for measure in out_measure:
            if max_ == 'label_encoder':
                out_data.append(le.transform([measure])[0])
            elif max_ == True:
                # returns 1 if there is a step
                if float(measure) > 0:
                    out_data.append('1')
                else:
                    out_data.append('0')
            else:
                # returns all cols flattened
                out_data += [note for note in measure]

        out_data = [
            note for n, note in enumerate(out_data)
            if n % (measure_length / note_type) == 0
        ]
        return np.array(out_data)

    @staticmethod
    def chart_exclude(chart, exclude=('2', '3', '4', 'M', 'K', 'F')):
        for i in range(len(chart)):
            temp_measure = chart.iloc[i]
            for a in exclude:
                temp_measure = np.char.replace(temp_measure, a, '1')
            chart.iloc[i] = str(temp_measure)
        return chart

    def generate_data(self, max_=False, exclude=('2', '3', '4', 'M', 'K', 'F')):
        self.chart = self.chart_exclude(self.chart, exclude)
        measure_break = [
            self.chart.index[0]
        ] + self.chart[self.chart.str.startswith(',')].index.tolist()
        self.output_data = np.array([])
        for i in range(len(measure_break) - 1):
            measure = self.chart.loc[measure_break[i]:measure_break[i + 1]]
            if self.output_data.size:
                self.output_data = np.column_stack([
                    self.output_data,
                    self.measure_to_data(measure, max_=max_)
                ])
            else:
                self.output_data = self.measure_to_data(measure, max_=max_)

    @staticmethod
    # TODO: REFACTOR with new numpy stuff
    def output_data_to_df(output_data):
        measure_df = pd.DataFrame()
        for measure in output_data:
            measure_strings = []
            for i in range(int(len(measure) / 4)):
                measure_strings.append(''.join(measure[i * 4:(i + 1) * 4]))
            assert len(measure_strings) / 192
            measure_strings.append(',')
            measure_strings = pd.DataFrame(measure_strings)
            measure_df = measure_df.append(measure_strings)
        measure_df.iloc[-1] = ';'
        return measure_df


if __name__ == '__main__':
    sm_file = '/media/adrian/Main/Games/StepMania 5/Songs/GIRLS/30Min Harder/30 Minutes.sm'
    sm_file = "/media/adrian/Main/Games/StepMania 5/test_packs/You're Streaming Forever/-273.15/-273.15.sm"
    sm_file = "/media/adrian/Main/Games/StepMania 5/test_packs/You're Streaming Forever/Block Control VIP/Block Control VIP.sm"
    #sm_file = '/media/adrian/Main/Games/StepMania 5/train_packs/Cirque du Zonda/Zaia - Lifeguard/DJ Myosuke — Lifeguard.sm'
    try:
        self = sm(sm_file)
    except:
        'die'
    self.load_chart(0)
    self.chart_exclude(self.chart)
    self.generate_data(max_=True)
    label_encoder(hit=('0', '1'))
    self.generate_data(max_='label_encoder')
