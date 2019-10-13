from keras.layers import Dense, Dropout
from keras.models import Sequential


class nn():
    def __init__(self, input_dim, output_dim):
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
              batch_size=1500,
              epochs=10,
              validation_data=False):
        if validation_data:
            self.model.fit(x_train,
                           y_train,
                           batch_size=batch_size,
                           epochs=epochs,
                           shuffle=False,
                           validation_data=validation_data)
        else:
            self.model.fit(
                x_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                shuffle=False,
            )
