import pickle
import os

from .transformer import TfidfTransformerCustom
from .vectorizer  import TfidfVectorizerCustom

class TfIdfCore:

    METHODS = {
        'vectorizer': {
            'name'  : 'Vectorizer',
            'class' :  TfidfVectorizerCustom
        },
        'transformer': {
            'name'  : 'Transformer',
            'class' :  TfidfTransformerCustom
        }
    }

    def __init__(self, **kwargs):

        self.ngram_range   = kwargs.get('ngram_range', (1, 1))

        self.tf_idf_storage_folder = kwargs.get('bot_storage_folder') + '/' + 'tf_idf'

        self.corpus = kwargs.get('corpus', [])

        self.__select_tfidf_method(kwargs.get('method_key'))

    def __select_tfidf_method(self, method_key):

        self.method_key = method_key

        if  self.method_key == 'transformer':

            self.ifidf_instance = self.METHODS[self.method_key]['class'](
                ngram_range=self.ngram_range,
                corpus=self.corpus,
                tf_idf_storage_folder=self.tf_idf_storage_folder)

    def create_weight_of_corpus(self):

        if  self.method_key == 'transformer':

            self.ifidf_instance.create_weight()

    def process(self, text):
        """
        Process your statements with if-idf weight already created before
        """

        return self.ifidf_instance.process([text])
