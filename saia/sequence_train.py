import joblib
import numpy as np
from keras.layers import LSTM, Dense, Input
from keras.models import Model

from utils.sm import sm
from train import get_files

class dataset():
    def __init__(self, songs_dir):
        # TODO:
        # - Figure out a way to add song features
        # - Get my features for both input (binary) and output (for each step)
            




if __name__ == '__main__':
    songs_dir = '/media/adrian/Main/Games/StepMania 5/train_packs/'
    sm_files, song_files = get_files(songs_dir)

    from utils.sm import label_encoder
    label_encoder(hit=('0', '1'))
    le = joblib.load('label_encoder.pickle')

    from keras.utils import to_categorical
    sm_file = sm_files[0]
    x = sm(sm_file)
    x.load_chart(0)
    x.generate_data(max_=True)
    x.generate_data(max_='label_encoder')
    x.output_data = x.output_data.transpose()
    to_categorical(x.output_data, num_classes=len(le.classes_))
    


    latent_dim = 1000
    num_encoder_tokens = len(le.classes_)
    num_decoder_tokens = num_encoder_tokens

    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                        initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)

    model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    model.fit([encoder_input_data, decoder_input_data],
            decoder_target_data,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2)
