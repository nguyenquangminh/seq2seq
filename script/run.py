import os

class TaskSolver:

    def __init__(self):
        pass

    def solve(self, task_name):

        if task_name == 'training':

            self.task_training()

        elif task_name == 'testing':

            self.task_testing()

        elif task_name == 'extract_conversation':

            self.task_extract_conversation()

    def task_training(self):

        from helper.tranning import Traning

        traning = Traning()

        traning.train()

    def task_extract_conversation(self):

        from helper.extract_conversation import ExtractConversation

        extract_conversation = ExtractConversation()

        extract_conversation.process()

    def task_testing(self):

        from helper.testing import Testing

        testing = Testing()

        testing.test()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Helper')

    parser.add_argument(
      '--task',
      metavar='path',
      required=True,
      help='Task')

    args = parser.parse_args()

    # reading argument input

    task_name = args.task

    print (
        """Argument:
        Task => {}
        """
        .format(
            task_name))

    task_solver = TaskSolver()

    task_solver.solve(task_name)
