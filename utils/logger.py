# A simple torch style logger
# (C) Wei YANG 2017
# (C) Modified by zlang 2020
from __future__ import absolute_import
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import matplotlib as mpl


__all__ = ['Logger', 'subplot', 'savefig']


mpl.style.use('seaborn')
plt.rc('font', family='serif')
# plt.rc('text', usetex=True)

SimSun = {'family': 'SimSun'}
SimHei = {'family': 'SimHei'}
Kaiti = {'family': 'Kaiti'}


class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, xlabel=None, ylabel=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = title or ''
        self.xlabel = xlabel or ''
        self.ylabel = ylabel or ''
        self.fpath = fpath
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(float(numbers[i]))
                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):
        names = names or self.names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(1, 1 + len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]), linewidth=1.0, label=name)
        plt.legend(loc='right')
        plt.title(self.title)
        plt.xlabel(self.xlabel, Kaiti)
        plt.ylabel(self.ylabel, Kaiti)
        plt.grid(True)

    def savefig(self, figname, dpi=None):
        savefig(os.path.join(os.path.dirname(self.fpath), figname), dpi)


    def close(self):
        if self.file is not None:
            self.file.close()
        plt.close('all')


def subplot (nets, layout=121):
    plt.figure()
    for i, args in enumerate(nets):
        plt.subplot(layout + i)
        logger = Logger(*args, resume=True)
        logger.plot()

    plt.grid(True)


def savefig(figname, dpi=None):
    plt.savefig(figname, dpi=dpi or 300)
