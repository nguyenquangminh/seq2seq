from chatterbot import ChatBot
from ..helper   import CosinSimilarityNgram

from chatterbot.comparisons import levenshtein_distance, sentiment_comparison, jaccard_similarity

SIMILARITY_DISTANCE = {
    'cosine_similarity_ngram' : CosinSimilarityNgram,
    'levenshtein_distance'    : levenshtein_distance,
    'sentiment_comparison'    : sentiment_comparison,
    'jaccard_similarity'      : jaccard_similarity
}

class VietnameseFQABot(ChatBot):

    DEFAULT_STATEMENT_COMPARISON_FUNCTION = 'cosine_similarity_ngram'

    def __init__(self, **kwargs):

        super().__init__(
            kwargs.get('name'),
            # input_adapter='chatterbot.input.TerminalAdapter',
            # output_adapter='chatterbot.output.TerminalAdapter',
            preprocessors=[
                'retrival_based.helper.preprocessor._core.preprocess_vi'
            ],

            logic_adapters=[{
                'import_path': 'chatterbot.logic.BestMatch',
                'default_response': kwargs.get('fallback_response')
            }],

            statement_comparison_function=self.__setup_similar_distance(**kwargs),

            storage_adapter='chatterbot.storage.MongoDatabaseAdapter',
            database_uri=kwargs.get('database_uri'),
            read_only=True
        )


    def __setup_similar_distance(self, **kwargs):
        self.bot_storage_folder = kwargs.get('bot_storage_folder')

        self.cosine_similarity_ngram = kwargs.get(
            'statement_comparison_function',
            self.DEFAULT_STATEMENT_COMPARISON_FUNCTION)

        if  self.cosine_similarity_ngram == 'cosine_similarity_ngram':

            return SIMILARITY_DISTANCE[self.cosine_similarity_ngram](
                bot_storage_folder=kwargs.get('bot_storage_folder'))

        elif  self.cosine_similarity_ngram == 'cosine_similarity_ngram':

            return SIMILARITY_DISTANCE[self.cosine_similarity_ngram]

        elif  self.cosine_similarity_ngram == 'sentiment_comparison':

            return SIMILARITY_DISTANCE[self.cosine_similarity_ngram]

        elif  self.cosine_similarity_ngram == 'jaccard_similarity':

            return SIMILARITY_DISTANCE[self.cosine_similarity_ngram]

    def get_name(self):
        return self.name
