import re
import math
import numpy as np
import nltk

from itertools   import chain
from collections import Counter

from nltk.util import ngrams
from textblob  import TextBlob

class NgramCore:

    DEFAULT_METHOD = 'get_tuples_manual_sentences'

    def __init__(self, **kwargs):
        self.re_sent_ends_naive = re.compile(r'[.\n]')
        self.re_stripper_alpha  = re.compile('[^a-zA-Z]+')
        self.re_stripper_naive  = re.compile('[^a-zA-Z\.\n]')

        self.text  = kwargs.get('text')
        self.flag  = kwargs.get('flag', self.DEFAULT_METHOD)
        self.ngram = kwargs.get('ngram')

        self.splitter_naive = lambda x: self.re_sent_ends_naive.split(
            self.re_stripper_naive.sub(' ', x))

        self.N_GRAM_METHODS = {
            'get_tuples_nosentences'        : self.__get_tuples_nosentences,
            'get_tuples_manual_sentences'   : self.__get_tuples_manual_sentences,
            'get_tuples_textblob_sentences' : self.__get_tuples_textblob_sentences
        }

    def process(self):

        return self.N_GRAM_METHODS[self.flag](self.text)

    def __get_tuples_nosentences(self, txt):
        """Get tuples that ignores all punctuation (including sentences)."""
        if not txt: return None

        ng = ngrams(self.re_stripper_alpha.sub(' ', txt).split(), self.ngram)

        return list(ng)

    def __get_tuples_manual_sentences(self, txt):
        """Naive get tuples that uses periods or newlines to denote sentences."""
        if not txt: return None

        sentences = (x.split() for x in self.splitter_naive(txt) if x)

        ng = (ngrams(x, self.ngram) for x in sentences if len(x) >= self.ngram)

        return list(chain(*ng))

    def __get_tuples_textblob_sentences(self, txt):
        """New get_tuples that does use textblob."""
        if not txt: return None

        tb = TextBlob(txt)

        ng = (ngrams(x.words, self.ngram) for x in tb.sentences if len(x.words) > self.ngram)

        return [ item for sublist in ng for item in sublist ]
