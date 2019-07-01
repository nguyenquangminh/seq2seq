from .trainer.fqa_list_trainer import FQAListTrainer
from chatterbot.trainers       import ListTrainer
from .bot.en_fqa_bot           import EnglishFQABot

TRAINERS = {
    'FQAListTrainer': FQAListTrainer
}

class BotTrainer:

    @classmethod
    def get_trainer(cls, name):

        chatbot = EnglishFQABot(name='EnglishFQABot')

        return ListTrainer(chatbot)
