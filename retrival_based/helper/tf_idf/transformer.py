import os, pickle

# from yourbot.storage import storage_manager

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


class TfidfTransformerCustom():

    def __init__(self, **kwargs):

        self.corpus = kwargs.get('corpus', [])

        self.weight_storage_folder = kwargs.get('tf_idf_storage_folder') + '/' + 'weight'

        self.ngram_range  = kwargs.get('ngram_range', (1, 1))

        self.tf_idf_vectors = []

        self._count_vectorizer  = CountVectorizer(ngram_range=self.ngram_range)

        self._tfidf_transformer = TfidfTransformer(
            smooth_idf=True, use_idf=True)

    def create_weight(self):

        word_count_vector = self._count_vectorizer.fit_transform(self.corpus)

        self._tfidf_transformer.fit(word_count_vector)

        self.__create_save_dir()

        self.__save_tfidf_vector()

        self.__save_count_vector()

    def __create_save_dir(self):

        storage_manager.create_folder(self.weight_storage_folder)

    def __save_tfidf_vector(self):

        file_path     = self.weight_storage_folder + '/tfidf_vector.bin'

        file_instance = open(file_path, 'wb')

        pickle.dump(self._tfidf_transformer, file_instance)

    def __save_count_vector(self):

        file_path     = self.weight_storage_folder + '/count_vector.bin'

        file_instance = open(file_path, 'wb')

        pickle.dump(self._count_vectorizer, file_instance)

    def process(self, corpus):

        self.__load_tfidf_vector()

        self.__load_count_vector()

        count_vector   = self._count_vectorizer.transform(corpus)

        tf_idf_vectors = self._tfidf_transformer.transform(count_vector)

        return tf_idf_vectors

    def get_tfidf_transformer(self):
        return self._tfidf_transformer

    def get_count_vectorizer(self):
        return self._count_vectorizer

    def __load_tfidf_vector(self):
        file_path = self.weight_storage_folder + '/tfidf_vector.bin'

        self._tfidf_transformer = pickle.load(open(file_path, 'rb'))

    def __load_count_vector(self):
        file_path = self.weight_storage_folder + '/count_vector.bin'

        self._count_vectorizer = pickle.load(open(file_path, 'rb'))
