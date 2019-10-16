import os

from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD


class nn():
    def create_model(self, input_dim, output_dim, save_path):
        self.model = Sequential()
        self.model.add(Dense(1000, activation='relu', input_dim=input_dim))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(80, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(80, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(80, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(80, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(80, activation='relu'))
        self.model.add(Dense(output_dim, activation='softmax'))
        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])

        self.checkpointer = ModelCheckpoint(filepath=os.path.join(
            save_path, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                            verbose=1)

    def save(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.load_model(path)
