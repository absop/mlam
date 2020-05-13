import os
import shutil
import random


train_dir = "Data/train"
test_dir = "Data/test"
rate_test = 0.2


def move():
    for d in os.listdir(train_dir):
        class_train_dir = os.path.join(train_dir, d)
        class_test_dir = os.path.join(test_dir, d)

        os.makedirs(class_test_dir, exist_ok=True)

        imglist = list(os.listdir(class_train_dir))
        numtest = int(rate_test * len(imglist))
        test_list = random.sample(imglist, numtest)

        for img in test_list:
            origin = os.path.join(class_train_dir, img)
            target = os.path.join(class_test_dir, img)
            shutil.move(origin, target)
            print("Move {} to {}".format(origin, target))

        print()


def rename():
    pass
