from chatterbot import utils

from chatterbot.trainers import Trainer

from chatterbot.conversation import Statement


class FQAListTrainer(Trainer):

    def __init__(self, storage, **kwargs):

        super(FQAListTrainer, self).__init__(storage, **kwargs)
