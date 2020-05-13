import re


__all__ = ['AverageMeter', 'Accuracy']


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return "{:.2f} ({:.2f}/{:n})".format(self.avg, self.sum, self.count)


accuracy_regex = re.compile(r'(?P<right>[0-9]+)/(?P<total>[0-9]+)\s*\((?P<value>[0-9]*\.[0-9]+|[0-9]+)%\)')


class Accuracy(object):
    def __init__(self, fromstr="0/0 (0.00%)"):
        group = accuracy_regex.match(fromstr).group
        self.right = int(group("right"))
        self.total = int(group("total"))
        self.value = float(group("value")) / 100

    def update(self, predict_labels, target_labels):
        plen, tlen = len(predict_labels), len(target_labels)
        assert plen == tlen and plen != 0

        self.right += predict_labels.eq(target_labels.long()).sum().item()
        self.total += tlen
        self.value = self.right / self.total

    def __str__(self):
        return "{}/{} ({:.2%})".format(self.right, self.total, self.value)


    def __gt__(self, b):
        return self.value > b.value

    def __lt__(self, b):
        return self.value < b.value

    def __eq__(self, b):
        return self.value == b.value
