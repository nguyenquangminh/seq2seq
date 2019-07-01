from chatterbot import ChatBot

from chatterbot.comparisons \
import \
    levenshtein_distance, \
    sentiment_comparison, \
    jaccard_similarity,   \
    synset_distance

SIMILARITY_DISTANCE = {
    'levenshtein_distance' : levenshtein_distance,
    'sentiment_comparison' : sentiment_comparison,
    'jaccard_similarity'   : jaccard_similarity,
    'synset_distance'      : synset_distance
}

class EnglishFQABot(ChatBot):

    DEFAULT_STATEMENT_COMPARISON_FUNCTION = 'sentiment_comparison'

    def __init__(self, **kwargs):
        super().__init__(kwargs.get('name'),
            input_adapter='chatterbot.input.TerminalAdapter',
            output_adapter='chatterbot.output.TerminalAdapter',
            # preprocessors=[
            #     'yourbot.chatterbot.preprocessors.preprocess_en'
            # ],

            logic_adapters=[{
                'import_path': 'chatterbot.logic.BestMatch',
                'default_response': 'K hiểu nhé'
            }],

            statement_comparison_function=self.__setup_similar_distance(**kwargs),
            read_only=True
        )

    def __setup_similar_distance(self, **kwargs):
        self.cosine_similarity_ngram = kwargs.get(
            'statement_comparison_function',
            self.DEFAULT_STATEMENT_COMPARISON_FUNCTION)

        return SIMILARITY_DISTANCE[self.cosine_similarity_ngram]
