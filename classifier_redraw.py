import os
from utils import Logger, subplot, savefig


def train_loop(logger):

    resume =  os.path.isfile(logger[0])
    logger = Logger(*logger, resume=resume)

    logger.plot()
    # logger.savefig('log.eps')
    logger.savefig('log.pdf')
    logger.savefig('log.png', dpi=300)
    logger.close()


loggers = [
   ('Classifier/checkpoints/log.txt', "VGG16 network", '训练次数'),
   ('Classifier/checkpoints/pretrained/log.txt', "Pretrained VGG16 network", '训练次数'),
   ('Classifier/checkpoints/eval-train/log.txt', "VGG16 network", '训练次数'),
   ('Classifier/checkpoints/eval-train/pretrained/log.txt', "Pretrained VGG16 network", '训练次数')
]

# Set pretrained to True to perform transfer learning.
def main():
    for task in loggers:
        train_loop(task)


if __name__ == '__main__':
    main()

    subplot([logger for logger in loggers[:2]])
    savefig("Classifier/VGG16 training process.png")
    savefig("Classifier/VGG16 training process.pdf")

    subplot([logger for logger in loggers[2:]])
    savefig("Classifier/VGG16 training process (eval-train).png")
    savefig("Classifier/VGG16 training process (eval-train).pdf")
