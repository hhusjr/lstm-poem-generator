"""
LSTM训练模型
@author Junru Shen
"""
import os

import numpy as np
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.python.keras.layers import LSTM, Dense, Dropout, Activation
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.optimizer_v2.adam import Adam

from utils import log

class PoemModel:
    def __init__(self, preprocessed, weight_file, window_size, learning_rate, batch_size):
        self._learning_rate = learning_rate
        self._model = None
        self._batch_size = batch_size
        self._preprocessed = preprocessed
        self._window_size = window_size
        self._weight_file = weight_file
        #self._predictor = Predictor(window_size=self._window_size, preprocessed=self._preprocessed)
        if not os.path.exists(weight_file):
            self._model = load_model(weight_file)
        else:
            self.train()

    def build_model(self):
        log('Building LSTM-RNN model...')

        self._model = Sequential()
        self._model.add(LSTM(512, input_shape=(self._window_size, len(self._preprocessed.words)), dropout=.20, recurrent_dropout=.20, return_sequences=True))
        self._model.add(LSTM(256, dropout=.20, recurrent_dropout=.20))
        self._model.add(Dropout(0.2))
        self._model.add(Dense(len(self._preprocessed.words)))
        self._model.add(Activation('softmax'))
        self._model.summary()
        # self._predictor.set_model(self._model)

        log('Builded LSTM-RNN model.', True)

    def compile_model(self):
        log('Compiling LSTM-RNN model...')

        optimizer = Adam(lr=self._learning_rate)
        self._model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def _get_num_of_epoch(self):
        poems_count = len(self._preprocessed.get_poems())
        cleaned_data_count = len(self._preprocessed.cleaned_data)

        num_of_epoch = cleaned_data_count - (self._window_size + 1) * poems_count
        num_of_epoch /= self._batch_size
        num_of_epoch = int(num_of_epoch / 1.5)

        log('Epoches={}'.format(num_of_epoch))
        log('|Poems|={}'.format(poems_count))
        log('|CleanedPoemTrainingData|={}'.format(cleaned_data_count))

        return num_of_epoch

    def data_generator(self):
        word_cnt = len(self._preprocessed.words)

        for poem in self._preprocessed.get_poems():
            for index, ch in enumerate(poem):
                upper = index + self._window_size
                if upper >= len(poem):
                    break

                x = poem[index: upper]
                x_vec = np.zeros(
                    shape=(1, self._window_size, word_cnt),
                    dtype=np.bool
                )
                for word_pos, word in enumerate(x):
                    x_vec[0, word_pos, self._preprocessed.word2id(word)] = True

                y = poem[upper]
                y_vec = np.zeros(
                    shape=(1, word_cnt),
                    dtype=np.bool
                )
                y_vec[0, self._preprocessed.word2id(y)] = True

                yield x_vec, y_vec

    def test_sample(self, epochs, logs):
        if epochs % 4 != 0:
            return

        log('Test sample generated.')
        print(self._predictor.predict_random())

    def train(self):
        log('Training model...')
        tb_callback = TensorBoard(log_dir='./logs', histogram_freq=1, write_grads=True)
        if not self._model:
            self.build_model()
            self.compile_model()
        self._model.fit(
            self.data_generator(),
            verbose=True,
            steps_per_epoch=self._batch_size,
            epochs=self._get_num_of_epoch(),
            callbacks=[
                tb_callback,
                #LambdaCallback(on_epoch_end=self.test_sample),
                ModelCheckpoint(self._weight_file, save_weights_only=False)
            ]
        )
