from chatterbot.comparisons import Comparator

from ..tf_idf import TfIdfCore

class CosinSimilarityNgram(Comparator):

    def __init__(self, **kwargs):
        super().__init__()

        self.bot_storage_folder = kwargs.get('bot_storage_folder', '')

        self.tf_idf_core = TfIdfCore(**{
            'method_key'        : 'transformer',
            'ngram_range'       : (1, 2),
            'bot_storage_folder': self.bot_storage_folder
        })

    def compare(self, statement, other_statement):
        from sklearn.metrics.pairwise import cosine_similarity

        """
        Compare the two input statements.

        :return: The percent of similarity between the text of the statements.
        :rtype: float
        """

        # Return 0 if either statement has a falsy text value
        if not statement.text or not other_statement.text:
            return 0

        # Get the lowercase version of both strings
        statement_text       = str(statement.text.lower())
        other_statement_text = str(other_statement.text.lower())

        statement_score       = self.tf_idf_core.process(statement_text)
        other_statement_score = self.tf_idf_core.process(other_statement_text)

        similarity = cosine_similarity(statement_score, other_statement_score)

        percent = round(similarity.flat[0], 2)

        return percent
