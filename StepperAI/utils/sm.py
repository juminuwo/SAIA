import numpy as np
import pandas as pd


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
    def measure_to_data(measure, measure_length=192):
        measure = measure[~measure.str.startswith(',')].reset_index(drop=True)
        measure_step = measure_length / len(measure)
        assert (measure_step).is_integer()
        out_measure = pd.Series(['0000' for _ in range(measure_length)])
        for n, m in enumerate(measure):
            out_measure[n * measure_step] = m

        out_data = []
        for measure in out_measure:
            out_data += [note for note in measure]

        assert len(out_data) == measure_length * 4
        return out_data

    def generate_data(self):
        measure_break = [
            self.chart.index[0]
        ] + self.chart[self.chart.str.startswith(',')].index.tolist()

        self.output_data = []
        for i in range(len(measure_break) - 1):
            measure = self.chart.loc[measure_break[i]:measure_break[i + 1]]
            self.output_data.append(self.measure_to_data(measure))

    @staticmethod
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
    self = sm(sm_file)
    self.load_chart(0)
    self.generate_data()
