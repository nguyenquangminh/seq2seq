import sys, os

sys.path.append(os.path.dirname('../..'))

from retrival_based.bot import EnglishFQABot

from seq2seq import SeqToSeq

class Testing:

    def __init__(self):

        self.chatterbot = EnglishFQABot(name='EnglishFQABot')

        self.seq2seq    = SeqToSeq()

    def test(self):

        index = 0

        while True:

            index += 1

            input_var = input(
                "\n********************************** \n" +
                "{}/Enter your sentence : ".format(index))

            if not input_var:
                print('Your sentence is invalid!')

                next


            chatterbot_res = self.chatterbot.get_response(input_var)

            seq2seq_res    = self.seq2seq.decode_sequence(input_var)

            print(
                "\n********************************** \n" +
                "{}/ChatterBot response : {} ".format(index, chatterbot_res))

            print(
                "\n********************************** \n" +
                "{}/Seq2Seq response : {} ".format(index, seq2seq_res))
