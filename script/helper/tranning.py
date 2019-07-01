import sys, os, json

sys.path.append(os.path.dirname('../..'))

from retrival_based.bot_trainer import BotTrainer

from seq2seq import SeqToSeq

class Traning:

    DATA_PATH = os.path.abspath('../data')

    def __init__(self):

        self.json_data = {}

        self.processed_data = []

    def train(self):

        self.__get_data()

        self.__process_data()

        self.__chatterbot_training()

        self.__seq_to_seq_training()

    def __chatterbot_training(self):

        bot_trainer = BotTrainer.get_trainer('FQAListTrainer')

        # total_corpus = len(self.processed_data)

        # for index, data_array in enumerate(self.processed_data):

        #     print('Training Corpus ({}/{}) : '.format(index + 1, total_corpus))

        bot_trainer.train(self.processed_data)

    def __seq_to_seq_training(self):

        seq = SeqToSeq()

        seq.train(self.processed_data)

    def __get_data(self):

        file_path = self.DATA_PATH + '/extracted_conversation.json'

        # self.data = pickle.load(open(file_path, 'rb'))

        with open(file_path) as json_file:

            self.json_data = json.load(json_file)

    def __process_data(self):

        for id, con in self.json_data.items():

            for index, sen_pair in enumerate(con):

                self.processed_data.append(sen_pair['question'])

                self.processed_data.append(sen_pair['answer'])
