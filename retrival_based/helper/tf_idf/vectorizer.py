from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfVectorizerCustom():

    def __init__(self, **kwargs):

        self.corpus = kwargs.get('corpus')

        self.ngram_range = kwargs.get('ngram_range', (1, 1))

        self.tfidf_vectorizer = TfidfVectorizer(use_idf=True, ngram_range=self.ngram_range)

    def process(self):

        self.tfidf_vectorizer_vectors = self.tfidf_vectorizer.fit_transform(self.corpus)

        return self.tfidf_vectorizer_vectors
