from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint


class nn():
    def create_model(self, input_dim, output_dim):
        self.model = Sequential()
        self.model.add(Dense(1000, activation='relu', input_dim=input_dim))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(800))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(800))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(800))
        self.model.add(Dense(output_dim, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])

    def train(self,
              x_train,
              y_train,
              save_path,
              batch_size=1500,
              epochs=10,
              validation_data=False):

        checkpointer = ModelCheckpoint(filepath=save_path,
                                         verbose=1)
        if validation_data:
            self.model.fit(x_train,
                           y_train,
                           batch_size=batch_size,
                           epochs=epochs,
                           shuffle=False,
                           validation_data=validation_data,
                           callbacks=[checkpointer])
        else:
            self.model.fit(
                x_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                shuffle=False,
                callbacks=[checkpointer]
            )

    def save(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.load_model(path)
