from __future__ import print_function

from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense
import tensorflow as tf

import numpy as np
import os, json, pickle

BATCH_SIZE      = 6        # Batch size for training.
EPOCHS          = 110       # Number of EPOCHS to train for.
LATENT_DIM      = 256       # Latent dimensionality of the encoding space.
NUM_SAMPLES     = 10000     # Number of samples to train on.
                            # Path to the data txt file on disk.


class SeqToSeq:

    MODEL_PATH = os.path.abspath('../seq2seq/model')

    def __init__(self):

        # Vectorize the data.
        self.input_texts  = []
        self.target_texts = []

        self.input_characters  = set()
        self.target_characters = set()

        file_path = self.MODEL_PATH + '/s2s.h5'

        if os.path.isfile(file_path):

            self.model = load_model(file_path)

            self.encoder_input_data = pickle.load(open(self.MODEL_PATH + '/encoder_input_data.bin', 'rb'))

            self.num_encoder_tokens = pickle.load(open(self.MODEL_PATH + '/num_encoder_tokens.bin', 'rb'))

            self.input_token_index = pickle.load(open(self.MODEL_PATH + '/input_token_index.bin', 'rb'))

            self.max_encoder_seq_length = pickle.load(open(self.MODEL_PATH + '/max_encoder_seq_length.bin', 'rb'))

            self.max_decoder_seq_length = pickle.load(open(self.MODEL_PATH + '/max_decoder_seq_length.bin', 'rb'))

            self.num_decoder_tokens = pickle.load(open(self.MODEL_PATH + '/num_decoder_tokens.bin', 'rb'))

            self.target_token_index = pickle.load(open(self.MODEL_PATH + '/target_token_index.bin', 'rb'))

            self.reverse_input_char_index  = pickle.load(open(self.MODEL_PATH + '/reverse_input_char_index.bin', 'rb'))

            self.reverse_target_char_index = pickle.load(open(self.MODEL_PATH + '/reverse_target_char_index.bin', 'rb'))

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

        for index, text in enumerate(data[: min(NUM_SAMPLES * 2, len(data) - 1)]):

            if index % 2 == 0:
                self.input_texts.append(text)

                for char in text:
                    if char not in self.input_characters:
                        self.input_characters.add(char)
            else:
                text = '\t' + text + '\n'

                self.target_texts.append(text)

                for char in text:
                    if char not in self.target_characters:
                        self.target_characters.add(char)

    def __create_useful_matrix(self):

        self.input_characters  = sorted(list(self.input_characters))
        self.target_characters = sorted(list(self.target_characters))

        self.num_encoder_tokens = len(self.input_characters)
        self.num_decoder_tokens = len(self.target_characters)

        # Tìm kiếm text có độ dài lớn nhất trong dữ liệu question và answer

        self.max_encoder_seq_length = max([len(txt) for txt in self.input_texts])

        self.max_decoder_seq_length = max([len(txt) for txt in self.target_texts])

        # Khởi tạo dict lưu các ký tự xuất hiện trong corpus cùng index của
        # các ký tự đó

        self.input_token_index = dict(
            [(char, i) for i, char in enumerate(self.input_characters)])

        self.target_token_index = dict(
            [(char, i) for i, char in enumerate(self.target_characters)])

        # Tạo ma trận 0 3 chiều với kích thước
        # row = Số lượng text trong corpus
        # column = Chiều dài tối đa của text
        # depth = Số lượng ký tự độc lập trong corpus

        self.encoder_input_data = np.zeros(
            (len(self.input_texts), self.max_encoder_seq_length, self.num_encoder_tokens),
            dtype='float32')

        self.decoder_input_data = np.zeros(
            (len(self.input_texts), self.max_decoder_seq_length, self.num_decoder_tokens),
            dtype='float32')

        self.decoder_target_data = np.zeros(
            (len(self.input_texts), self.max_decoder_seq_length, self.num_decoder_tokens),
            dtype='float32')

        self.reverse_input_char_index = dict(
            (i, char) for char, i in self.input_token_index.items())
        self.reverse_target_char_index = dict(
            (i, char) for char, i in self.target_token_index.items())

    def __create_one_hot_vector(self):

        for i, (input_text, target_text) in enumerate(zip(self.input_texts, self.target_texts)):
            for t, char in enumerate(input_text):

                # Đánh dấu các ký tự xuất hiện trong input_text vào ma trận đã khởi tạo sẵn
                # Giá trị phần tử trong ma trận = 1 nếu ký tự xuất hiện trong text

                # i = thứ tự của text trong corpus
                # t = thứ tự của ký tự trong text
                # self.input_token_index[char] = thứ tự của ký tự trong dict chứa các ký tự
                # xuất hiện trong corpus

                self.encoder_input_data[i, t, self.input_token_index[char]] = 1.
            for t, char in enumerate(target_text):

                # Đánh dấu các ký tự xuất hiện trong target_text vào ma trận đã khởi tạo sẵn
                # Giá trị phần tử trong ma trận = 1 nếu ký tự xuất hiện trong text

                self.decoder_input_data[i, t, self.target_token_index[char]] = 1.

                if t > 0:

                    # Trong decoder_target_data, giá trị của phần tử đầu tiên trong
                    # từng ma trận con tương ứng với từng target_text sẽ bị bỏ qua
                    # và không chứa ký tự bắt đầu của text

                    self.decoder_target_data[i, t - 1, self.target_token_index[char]] = 1.

    def __create_model(self):

        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, self.num_encoder_tokens))
        encoder        = LSTM(LATENT_DIM, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)

        # We discard `encoder_outputs` and only keep the states.

        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.num_decoder_tokens))

        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.

        decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)

        decoder_outputs, _, _ = decoder_lstm(
            decoder_inputs,
            initial_state=encoder_states)

        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')

        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `self.encoder_input_data` & `self.decoder_input_data` into `self.decoder_target_data`
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Run training
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        self.model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=0.2)
        # Save model

        file_path = self.MODEL_PATH + '/s2s.h5'

        self.model.save(file_path)

        file_instance = open(self.MODEL_PATH + '/encoder_input_data.bin', 'wb')

        pickle.dump(self.encoder_input_data, file_instance)

        file_instance = open(self.MODEL_PATH + '/num_encoder_tokens.bin', 'wb')

        pickle.dump(self.num_encoder_tokens, file_instance)

        file_instance = open(self.MODEL_PATH + '/input_token_index.bin', 'wb')

        pickle.dump(self.input_token_index, file_instance)

        file_instance = open(self.MODEL_PATH + '/max_encoder_seq_length.bin', 'wb')

        pickle.dump(self.max_encoder_seq_length, file_instance)

        file_instance = open(self.MODEL_PATH + '/num_decoder_tokens.bin', 'wb')

        pickle.dump(self.num_decoder_tokens, file_instance)

        file_instance = open(self.MODEL_PATH + '/target_token_index.bin', 'wb')

        pickle.dump(self.target_token_index, file_instance)

        file_instance = open(self.MODEL_PATH + '/reverse_input_char_index.bin', 'wb')

        pickle.dump(self.reverse_input_char_index, file_instance)

        file_instance = open(self.MODEL_PATH + '/reverse_target_char_index.bin', 'wb')

        pickle.dump(self.reverse_target_char_index, file_instance)

        file_instance = open(self.MODEL_PATH + '/max_decoder_seq_length.bin', 'wb')

        pickle.dump(self.max_decoder_seq_length, file_instance)

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
            (1, self.max_encoder_seq_length, self.num_encoder_tokens),
            dtype='float32')

        for t, char in enumerate(text):

            one_hot_vector[0, t, self.input_token_index[char]] = 1.

        return one_hot_vector

    def decode_sequence(self, input_seq):

        input_seq = self.__create_text_one_hot_vector(input_seq)

        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)
        print(states_value)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.target_token_index['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
            len(decoded_sentence) > self.max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence
