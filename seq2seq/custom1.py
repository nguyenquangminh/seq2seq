from __future__ import print_function

from keras.models import Model, load_model
from keras.layers import Input, Embedding, LSTM, Dense
from nltk.tokenize import word_tokenize
# import tensorflow as tf

from keras import optimizers

import numpy as np
import os, json, pickle

BATCH_SIZE      = 64        # Batch size for training.
EPOCHS          = 100       # Number of EPOCHS to train for.
LATENT_DIM      = 256       # Latent dimensionality of the encoding space.
NUM_SAMPLES     = 10000     # Number of samples to train on.
                            # Path to the data txt file on disk.


class SeqToSeq:

    MODEL_PATH = os.path.abspath('../seq2seq/model')

    def __init__(self):

        # Vectorize the data.
        self.input_texts  = []
        self.target_texts = []

        self.input_vocab  = set()
        self.target_vocab = set(['<START>', '<END>'])

        file_path = self.MODEL_PATH + '/s2s.h5'

        if not os.path.isdir(self.MODEL_PATH):

            os.mkdir(self.MODEL_PATH)

        if os.path.isfile(file_path):

            self.model = load_model(file_path)

            extra_data = pickle.load(open(self.MODEL_PATH + '/extra_data.bin', 'rb'))

            self.encoder_input_data = extra_data['encoder_input_data']
            self.num_encoder_vocab  = extra_data['num_encoder_vocab']
            self.input_word_index   = extra_data['input_word_index']

            self.max_encoder_seq_length = extra_data['max_encoder_seq_length']
            self.max_decoder_seq_length = extra_data['max_decoder_seq_length']

            self.num_decoder_vocab = extra_data['num_decoder_vocab']
            self.target_word_index = extra_data['target_word_index']

            self.reverse_target_word_index = extra_data['reverse_target_word_index']

            encoder_inputs = self.model.input[0]   # input_1
            encoder_outputs, state_h_enc, state_c_enc = self.model.layers[2].output   # lstm_1
            encoder_states = [state_h_enc, state_c_enc]
            self.encoder_model  = Model(encoder_inputs, encoder_states)

            decoder_inputs = self.model.input[1]   # input_2
            decoder_state_input_h = Input(shape=(LATENT_DIM,), name='input_3')
            decoder_state_input_c = Input(shape=(LATENT_DIM,), name='input_4')
            decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
            decoder_lstm = self.model.layers[3]

            decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
                decoder_inputs, initial_state=decoder_states_inputs)

            decoder_states  = [state_h_dec, state_c_dec]
            decoder_dense   = self.model.layers[4]
            decoder_outputs = decoder_dense(decoder_outputs)
            self.decoder_model   = Model(
                [decoder_inputs] + decoder_states_inputs,
                [decoder_outputs] + decoder_states)

    def train(self, data):

        self.__get_data(data)

        self.__create_useful_matrix()

        self.__create_one_hot_vector()

        self.__create_model()


    def __get_data(self, data):

        # Tìm kiếm text có độ dài lớn nhất trong dữ liệu question và answer

        self.max_encoder_seq_length = 0

        self.max_decoder_seq_length = 0

        for index, text in enumerate(data[: min(NUM_SAMPLES * 2, len(data) - 1)]):

            word_array = word_tokenize(text)

            if index % 2 == 0:
                self.input_texts.append(text)

                if len(word_array) > self.max_encoder_seq_length:
                    self.max_encoder_seq_length = len(word_array)

                for word in word_array:
                    if word not in self.input_vocab:
                        self.input_vocab.add(word)

            else:
                self.target_texts.append(text)

                if len(word_array) > self.max_decoder_seq_length:

                    # Chiều dài lớn nhất của tập câu tring decoder phải + thêm 2 từ <START>, <END>
                    self.max_decoder_seq_length = len(word_array) + 2

                for word in word_array:
                    if word not in self.target_vocab:
                        self.target_vocab.add(word)

    def __create_useful_matrix(self):

        self.input_vocab  = sorted(list(self.input_vocab))

        self.target_vocab = sorted(list(self.target_vocab))

        self.num_encoder_vocab = len(self.input_vocab)

        self.num_decoder_vocab = len(self.target_vocab)

        # Khởi tạo dict lưu các từ xuất hiện trong corpus cùng index của các ký tự đó

        self.input_word_index = dict(
            [(word, i) for i, word in enumerate(self.input_vocab)])

        self.target_word_index = dict(
            [(word, i) for i, word in enumerate(self.target_vocab)])

        # Tạo ma trận 0 3 chiều với kích thước
        # row = Số lượng text trong corpus
        # column = Chiều dài tối đa của text
        # depth = Số lượng ký tự độc lập trong corpus

        self.encoder_input_data = np.zeros(
            (len(self.input_texts), self.max_encoder_seq_length, self.num_encoder_vocab),
            dtype='float32')

        self.decoder_input_data = np.zeros(
            (len(self.input_texts), self.max_decoder_seq_length, self.num_decoder_vocab),
            dtype='float32')

        self.decoder_target_data = np.zeros(
            (len(self.input_texts), self.max_decoder_seq_length, self.num_decoder_vocab),
            dtype='float32')

        self.reverse_target_word_index = dict(
            (i, word) for word, i in self.target_word_index.items())

    def __create_one_hot_vector(self):

        for i, (input_text, target_text) in enumerate(zip(self.input_texts, self.target_texts)):

            input_array  = word_tokenize(input_text)

            target_array = ['<START>']  + word_tokenize(target_text) + ['<END>']

            for t, word in enumerate(input_array):

                self.encoder_input_data[i, t, self.input_word_index[word]] = 1.

            for t, word in enumerate(target_array):

                self.decoder_input_data[i, t, self.target_word_index[word]] = 1.

                if t > 0:

                    self.decoder_target_data[i, t - 1, self.target_word_index[word]] = 1.

    def __create_model(self):

        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, self.num_encoder_vocab))
        encoder        = LSTM(LATENT_DIM, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)

        # We discard `encoder_outputs` and only keep the states.

        encoder_states = [state_h, state_c]

        # embedding_layer = Embedding(
        #     input_dim    = vocab_size,
        #     output_dim   = embedding_dimension,
        #     input_length = sequence_length)

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.num_decoder_vocab))

        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.

        decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)

        decoder_outputs, _, _ = decoder_lstm(
            decoder_inputs,
            initial_state=encoder_states)

        decoder_dense = Dense(self.num_decoder_vocab, activation='softmax')

        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `self.encoder_input_data` & `self.decoder_input_data` into `self.decoder_target_data`
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Run training
        rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

        self.model.compile(optimizer=rmsprop, loss='categorical_crossentropy')

        self.model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data,
            batch_size       = BATCH_SIZE,
            epochs           = EPOCHS,
            validation_split = 0.2)
        # Save model

        file_path = self.MODEL_PATH + '/s2s.h5'

        self.model.save(file_path)

        file_instance = open(self.MODEL_PATH + '/extra_data.bin', 'wb')

        pickle.dump({
            'encoder_input_data'        : self.encoder_input_data,
            'num_encoder_vocab'         : self.num_encoder_vocab,
            'input_word_index'          : self.input_word_index,
            'max_encoder_seq_length'    : self.max_encoder_seq_length,
            'num_decoder_vocab'         : self.num_decoder_vocab,
            'target_word_index'         : self.target_word_index,
            'reverse_target_word_index' : self.reverse_target_word_index,
            'max_decoder_seq_length'    : self.max_decoder_seq_length,
        }, file_instance)

    def search(model, src_input, k=1, sequence_max_len=25):
        # (log(1), initialize_of_zeros)
        k_beam = [(0, [0]*(sequence_max_len+1))]

        # l : point on target sentence to predict
        for l in range(sequence_max_len):
            all_k_beams = []
            for prob, sent_predict in k_beam:
                predicted = model.predict([np.array([src_input]), np.array([sent_predict])])[0]
                # top k!
                possible_k = predicted[l].argsort()[-k:][::-1]

                # add to all possible candidates for k-beams
                all_k_beams += [
                    (
                        sum(np.log(predicted[i][sent_predict[i+1]]) for i in range(l)) + np.log(predicted[l][next_wid]),
                        list(sent_predict[:l+1])+[next_wid]+[0]*(sequence_max_len-l-1)
                    )
                    for next_wid in possible_k
                ]

            # top k
            k_beam = sorted(all_k_beams)[-k:]

        return k_beam

    def __create_text_one_hot_vector(self, text):

        one_hot_vector = np.zeros(
            (1, self.max_encoder_seq_length, self.num_encoder_vocab),
            dtype='float32')

        for t, word in enumerate(text.split()):

            one_hot_vector[0, t, self.input_word_index[word]] = 1.

        return one_hot_vector

    def decode_sequence(self, input_seq):

        input_seq = self.__create_text_one_hot_vector(input_seq)

        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_decoder_vocab))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.target_word_index['<START>']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
            output_words, h, c = self.decoder_model.predict(
                [target_seq] + states_value)

            # Sample a word
            sampled_word_index = np.argmax(output_words[0, -1, :])
            sampled_char = self.reverse_target_word_index[sampled_word_index]
            decoded_sentence.append(sampled_char)

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '<END>' or
            len(decoded_sentence) > self.max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_vocab))
            target_seq[0, 0, sampled_word_index] = 1.

            # Update states
            states_value = [h, c]

        return ' '.join(decoded_sentence)
