import os
import json
import pickle

class ExtractConversation:

    DATA_PATH = os.path.abspath('../data')

    def __init__(self):

        self.data = {}

        self.extracted_conversation = {}

    def process(self):

        self.__read_file()

        self.__extract_conversation()

        self.__save_output_file()


    def __read_file(self):

        file_path = self.DATA_PATH + '/data_conversations.json'

        # self.data = pickle.load(open(file_path, 'rb'))

        with open(file_path) as json_file:

            self.data = json.load(json_file)

    def __extract_conversation(self):

        for id, content in self.data.items():

            self.extracted_conversation[id] = []

            next_sentence_type = 'answer'

            qna_pair = { 'question': None, 'answer': None }

            speaker  = None

            for index, sentence in enumerate(content['sentences']):

                if speaker == sentence['speaker']:

                    qna_pair[next_sentence_type] += ' ' + sentence['content']

                else:

                    if next_sentence_type == 'answer':
                        next_sentence_type = 'question'
                    else:
                        next_sentence_type = 'answer'

                    if qna_pair['question'] and qna_pair['answer']:

                        self.extracted_conversation[id].append(qna_pair)

                        qna_pair = { 'question': None, 'answer': None }

                    qna_pair[next_sentence_type] = sentence['content']

                speaker = sentence['speaker']

    def __save_output_file(self):

        file_path = self.DATA_PATH + '/extracted_conversation.json'

        with open(file_path, 'w') as outfile:
            json.dump(self.extracted_conversation, outfile)
