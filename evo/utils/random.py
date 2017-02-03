# -*- coding: utf-8 -*-
"""This package contains utility functions and classes for working with random
numbers and number generators.
"""

import enum
import random

__author__ = 'Jan Žegklitz'


class RandomIntIterable(object):
    """An iterable that, when iterated, generates a sequence of random
    integers.
    """

    def __init__(self, iterations_total, iterations_iterator, min_val,
                 max_val, **kwargs):
        self.iterations_total = float('inf')
        if iterations_total >= 0:
            self.iterations_total = iterations_total

        self.iterations_iterator = float('inf')
        if iterations_iterator >= 0:
            self.iterations_iterator = iterations_iterator

        self.min_val = min_val
        self.max_val = max_val
        self.total_count = 0
        self.iterator_count = 0

        self.generator = random
        if 'generator' in kwargs:
            self.generator = kwargs['generator']

        self.sequence = None
        if 'save' in kwargs and kwargs['save']:
            self.sequence = []

    def __iter__(self):
        self.iterator_count = 0
        return self

    def __next__(self):
        if ((self.total_count < self.iterations_total or
             self.iterations_total < 0) and
            (self.iterator_count < self.iterations_iterator or
             self.iterations_iterator < 0)):
            self.total_count += 1
            self.iterator_count += 1

            val = self.generator.randint(self.min_val, self.max_val)
            if self.sequence is not None:
                self.sequence.append(val)
            return val
        else:
            raise StopIteration

    def next(self):
        return self.__next__()

    def get_sequence(self):
        if self.sequence is None:
            return None
        return list(self.sequence)


class Distribution(enum.Enum):
    UNIFORM = 1
    TRIANGULAR = 2
    BETAVARIATE = 3
    EXPOVARIATE = 4
    GAMMAVARIATE = 5
    GAUSS = 6
    LOGNORMVARIATE = 7
    NORMALVARIATE = 8
    VONMISESVARIATE = 9
    PARETOVARIATE = 10
    WEIBULLVARIATE = 11

    @staticmethod
    def uniform(a, b):
        if a <= b:
            return Distribution.UNIFORM, a, b
        raise ValueError('Lower bound must be lower than the upper bound.')

    @staticmethod
    def triangular(low, high, mode=None):
        if mode is None:
            return Distribution.TRIANGULAR, low, high, (low + high) / 2
        if low <= mode <= high:
            return Distribution.TRIANGULAR, low, high, mode
        raise ValueError('The mode must be between the low and high values.')

    @staticmethod
    def betavariate(alpha, beta):
        if alpha > 0 and beta > 0:
            return Distribution.BETAVARIATE, alpha, beta
        raise ValueError('Alpha and beta parameters must both be grater than '
                         '0.')

    @staticmethod
    def expovariate(lamb):
        if lamb != 0:
            return Distribution.EXPOVARIATE, lamb
        raise ValueError('The lambda parameter must be nonzero.')

    @staticmethod
    def gammavariate(alpha, beta):
        if alpha > 0 and beta > 0:
            return Distribution.GAMMAVARIATE, alpha, beta
        raise ValueError('Alpha and beta parameters must both be grater than '
                         '0.')

    @staticmethod
    def gauss(mu, sigma):
        return Distribution.GAUSS, mu, sigma

    @staticmethod
    def lognormvariate(mu, sigma):
        if sigma > 0:
            return Distribution.LOGNORMVARIATE, mu, sigma
        raise ValueError('Sigma parameter must be grater than 0.')

    @staticmethod
    def normalvariate(mu, sigma):
        return Distribution.NORMALVARIATE, mu, sigma

    @staticmethod
    def vonmisesvariate(mu, kappa):
        if kappa >= 0:
            return Distribution.VONMISESVARIATE, mu, kappa
        raise ValueError('Kappa parameter must be grater than or equal to 0.')

    @staticmethod
    def paretovariate(alpha):
        return Distribution.PARETOVARIATE, alpha

    @staticmethod
    def weibullvariate(alpha, beta):
        return Distribution.WEIBULLVARIATE, alpha, beta

    @staticmethod
    def generate(distribution, generator=None):
        if generator is None:
            generator = random

        dist = distribution[0]
        pars = distribution[1:]

        if dist == Distribution.UNIFORM:
            return generator.uniform(*pars)
        if dist == Distribution.TRIANGULAR:
            return generator.triangular(*pars)
        if dist == Distribution.BETAVARIATE:
            return generator.betavariate(*pars)
        if dist == Distribution.EXPOVARIATE:
            return generator.expovariate(*pars)
        if dist == Distribution.GAMMAVARIATE:
            return generator.gammavariate(*pars)
        if dist == Distribution.GAUSS:
            return generator.gauss(*pars)
        if dist == Distribution.LOGNORMVARIATE:
            return generator.lognormvariate(*pars)
        if dist == Distribution.NORMALVARIATE:
            return generator.normalvariate(*pars)
        if dist == Distribution.VONMISESVARIATE:
            return generator.vonmisesvariate(*pars)
        if dist == Distribution.PARETOVARIATE:
            return generator.paretovariate(*pars)
        if dist == Distribution.WEIBULLVARIATE:
            return generator.weibullvariate(*pars)

        raise ValueError('The given distribution descriptor is does not '
                         'describe a supported distribution.')
