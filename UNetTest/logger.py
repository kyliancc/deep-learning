import time


class UNetLogger:
    def __init__(self, log_dir='./logs'):
        self.path = log_dir + '/' + time.strftime('%Y%m%d_%H%M%S') + '.txt'

    def log(self, message):
        print(message)
        with open(self.path, 'a') as f:
            f.write(message + '\n')
