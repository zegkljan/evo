# -*- coding: utf-8 -*-
"""Support classes for the :mod:`evo.ge` package.
"""

import fractions
import functools
import logging
import math
import numbers
import random

import evo
import evo.utils.grammar
import evo.utils.random

__author__ = 'Jan Žegklitz'


class CodonGenotypeIndividual(evo.Individual):
    """A class representing an individual as a linear string of integers.
    """

    def __init__(self, genotype, max_codon_value):
        """Creates the individual.

        :param genotype: the genotype of the individual
        :type genotype: :class:`list` of :class:`int`\\ s
        :param int max_codon_value: the maximum value of a codon in the
            genotype (exclusive, i.e. a codon will have this value minus one or
            lower)
        """
        evo.Individual.__init__(self)
        self.genotype = genotype
        self.max_codon_value = max_codon_value
        self.first_not_used = None
        self.annotations = None

    def __str__(self):
        if hasattr(self, 'str'):
            return str(self.str)
        return str(self.genotype)

    def copy(self, carry_evaluation=True, carry_data=True):
        clone = CodonGenotypeIndividual(list(self.genotype),
                                        self.max_codon_value)
        if self.annotations is not None:
            # noinspection PyTypeChecker
            clone.annotations = list(self.annotations)
        evo.Individual.copy_evaluation(self, clone, carry_evaluation)
        evo.Individual.copy_data(self, clone, carry_data)
        if carry_evaluation:
            clone.first_not_used = self.first_not_used
        return clone

    def set_first_not_used(self, first_not_used):
        self.first_not_used = first_not_used

    def get_first_not_used(self):
        return self.first_not_used

    def set_annotations(self, annotations):
        self.annotations = annotations

    def get_annotations(self):
        return self.annotations

    def get_max_codon_value(self):
        return self.max_codon_value

    def get_codon(self, index):
        return self.genotype[index]

    def set_codon(self, index, new_codon):
        if not (0 <= new_codon < self.max_codon_value):
            raise ValueError(('Codon value must be in range [0, {0}) but was'
                              ' {1}.').format(self.max_codon_value, new_codon))
        self.genotype[index] = new_codon

    def get_codon_num(self):
        return len(self.genotype)


class RandomCodonGenotypeInitializer(evo.IndividualInitializer):
    """Generates a genotype with a random length within a given range and
    random codons.
    """

    def __init__(self, min_length, max_length, **kwargs):
        """Creates an initializer with given parameters.

        The genotypes generated by the
        :meth:`RandomCodonGenotypeInitializer.initialize` are going to have
        the minimum length of ``min_length`` and maximum length of
        ``max_length``.

        The optional keyword argument ``max_codon_value`` controls the maximum
        integer value of each codon in the generated genotypes.

        The optional keyword argument ``generator`` can be used to pass a
        random number generator to the initializer which is to be used for
        generation. If it is ``None`` or not present a standard generator is
        used which is the :mod:`random` module and its functions. If a
        generator is passed it is expected to have the corresponding methods
        to the :mod:`random` module (i.e. the class :mod:`random`\\ .Random).

        .. warning::

            If default generator is used (i.e. the methods of :mod:`random`)
            it is assumed that it is already seeded and no seed is set inside
            this class.

        :param int min_length: minimum length of the genotype
        :param int max_length: maximum length of the genotype
        :keyword int max_codon_value: (keyword argument) maximum value a codon
            can have

            if ``None`` or not present default value of 255 is used
        :keyword generator: a random number generator; if ``None`` or not
            present calls to the methods of standard python module
            :mod:`random` will be performed instead
        :type generator: :mod:`random`\\ .Random or ``None``
        :return: a randomly generated individual
        :rtype: :class:`CodonGenotypeIndividual`
        """
        evo.IndividualInitializer.__init__(self)

        self.min_length = min_length
        self.max_length = max_length

        if 'max_codon_value' in kwargs:
            self.max_codon_value = kwargs['max_codon_value']
        else:
            self.max_codon_value = 255

        if 'generator' in kwargs:
            self.generator = kwargs['generator']
        else:
            self.generator = random

    # noinspection PyUnresolvedReferences
    def initialize(self):
        genotype = []
        for _ in range(self.generator.randint(self.min_length,
                                              self.max_length)):
            genotype.append(self.generator.randint(0, self.max_codon_value))
        return CodonGenotypeIndividual(genotype, self.max_codon_value)


class RandomWalkInitializer(evo.IndividualInitializer):
    """Generates a codon genotype by random walk through a grammar (i.e. the
    resulting genotypes encode exactly a complete derivation tree).
    """

    def __init__(self, grammar, **kwargs):
        """Creates an initializer with given parameters.

        The optional keyword argument ``generator`` can be used to pass a
        random number generator to the initializer which is to be used for
        generation. If it is ``None`` or not present a standard generator is
        used which is the :mod:`random` module and its functions. If a
        generator is passed it is expected to have the corresponding methods
        to the :mod:`random` module (individual.e. the class
        :mod:`random`\\ .Random).

        .. warning::

            Whatever generator is used it is assumed that it is already seeded
            and no seed is set inside this class.

        :param evo.utils.grammar.Grammar grammar: the grammar to generate
        :keyword generator: a random number generator; if ``None`` or not
            present calls to the methods of standard python module
            :mod:`random` will be performed instead
        :type generator: :mod:`random`\\ .Random or ``None``
        :keyword int min_depth: minimum depth of the corresponding derivation
            tree; if ``None`` or not set default value of 0 is used
        :keyword int max_depth: maximum depth of the corresponding derivation
            tree; if ``None`` or not set default value of infinity is used
        :keyword multiplier: number which will be used to multiply the LCM of
            all choices numbers to get a higher maximum codon value (default
            is 1, i.e. maximum codon value will be the LCM of numbers of all
            choices in the grammar)
        :keyword extra_codons: The number of how many (randomly generated)
            codons should be appended to the genotype after those needed for
            full expansion (a tail).

            If the number is an integral number then it specifies the tail size
            absolutely. If the number is a decimal number then the tail size is
            determined by multiplying the length of the effective part of the
            genotype by this number (rounded up). E.g. the value of ``10``
            produces tails of length 10, the value of ``0.3`` and the effective
            genotype length of 10 produces a tail of length 3.

            Default value is 0.
        :return: a randomly generated individual
        :rtype: :class:`CodonGenotypeIndividual`
        """
        evo.IndividualInitializer.__init__(self)

        self.grammar = grammar

        self.generator = random
        if 'generator' in kwargs:
            self.generator = kwargs['generator']

        self.min_depth = 0
        if 'min_depth' in kwargs:
            self.min_depth = kwargs['min_depth']

        self.max_depth = float('inf')
        if 'max_depth' in kwargs:
            self.max_depth = kwargs['max_depth']

        self.multiplier = 1
        if 'multiplier' in kwargs:
            self.multiplier = kwargs['multiplier']

        self.extra_codons = 0
        if 'extra_codons' in kwargs:
            self.extra_codons = kwargs['extra_codons']

        if isinstance(self.extra_codons, numbers.Integral):
            self.tail_length = lambda _: self.extra_codons
        else:
            self.tail_length = lambda l: math.ceil(l * self.extra_codons)

        choice_nums = [r.get_choices_num() for r in grammar.get_rules()]
        m = functools.reduce(lambda a, b: a * b // fractions.gcd(a, b),
                             choice_nums)
        self.max_choices = m * self.multiplier

    def initialize(self):
        iterator = evo.utils.random.RandomIntIterable(
            -1, -1, 0, self.max_choices - 1, generator=self.generator)
        sequence = []
        (_, _, _, _,
         annotations) = self.grammar.to_tree(decisions=iterator,
                                             max_wraps=0,
                                             min_depth=self.min_depth,
                                             max_depth=self.max_depth,
                                             sequence=sequence)
        for _ in range(self.tail_length(len(sequence))):
            sequence.append(iterator.__next__())
        individual = CodonGenotypeIndividual(sequence, self.max_choices)
        individual.set_annotations(annotations)
        # individual.set_data('init-text',
        #                     evo.utils.grammar.derivation_tree_to_text(_[0]))
        return individual


class RampedHalfHalfInitializer(evo.PopulationInitializer):
    """This population initializer initializes the population using the ramped
    half-and-half method: for each depth from 1 up to maximal depth, half of
    individuals will be crated using the "grow" method and the other half using
    the "full" method.

    If the number of individuals is not divisible by the number of
    initialization setups (which is double the number of depth levels - the
    "full" and "grow" for each level) then the remainder individuals will be
    initialized using randomly chosen setups (but each of them in a unique
    setup).
    """
    LOG = logging.getLogger(__name__ + '.RampedHalfHalfInitializer')

    def __init__(self, grammar, max_depth, **kwargs):
        """Creates the initializer.

        The optional ``min_depth`` keyword argument can be used to generate
        trees from this depth instead of 1.

        :param evo.utils.grammar.Grammar grammar: grammar to generate from
        :param int max_depth: maximum depth of the derivation trees; must be
            finite
        :keyword generator: a random number generator; if ``None`` or not set
            calls to the methods of standard python module :mod:`random` will be
            performed instead
        :type generator: :class:`random.Random` or ``None``
        :keyword int min_depth: starting minimum depth of the derivation
            trees; if ``None`` or not set the
            ``grammar.get_minimum_expansion_depth()`` is used
        :keyword int multiplier: number which will be used to multiply the LCM
            of all choices numbers to get a higher maximum codon value (default
            is 1, i.e. maximum codon value will be the LCM of numbers of all
            choices in the grammar)
        :keyword extra_codons: The number of how many (randomly generated)
            codons should be appended to the genotype after those needed for
            full expansion (a tail).

            If the number is an integral number then it specifies the tail size
            absolutely. If the number is a decimal number then the tail size is
            determined by multiplying the length of the effective part of the
            genotype by this number (rounded up). E.g. the value of ``10``
            produces tails of length 10, the value of ``0.3`` and the effective
            genotype length of 10 produces a tail of length 3.

            Default value is 0.
        :keyword int max_tries: the maximum number of attempts to recreate a new
            individual if an identical one (in the derivation tree, not the
            codons) is already in the population (default is 100)

        .. seealso::

            :meth:`evo.utils.grammar.Grammar.get_minimum_expansion_depth`
        """
        super().__init__()

        self.grammar = grammar
        self.max_depth = max_depth

        self.generator = random
        if 'generator' in kwargs:
            self.generator = kwargs['generator']

        self.min_depth = grammar.get_minimum_expansion_depth()
        if 'min_depth' in kwargs:
            self.min_depth = kwargs['min_depth']
            if self.min_depth > self.max_depth:
                raise ValueError('min_depth must not be greater than max_depth')

        self.multiplier = 1
        if 'multiplier' in kwargs:
            self.multiplier = kwargs['multiplier']

        self.extra_codons = 0
        if 'extra_codons' in kwargs:
            self.extra_codons = kwargs['extra_codons']

        self.max_tries = 100
        if 'max_tries' in kwargs:
            self.max_tries = kwargs['max_tries']

    def initialize(self, pop_size):
        RampedHalfHalfInitializer.LOG.info('Initializing population of size '
                                           '%d', pop_size)
        initializer = RandomWalkInitializer(self.grammar,
                                            generator=self.generator,
                                            multiplier=self.multiplier,
                                            extra_codons=self.extra_codons)
        levels_num = self.max_depth - self.min_depth + 1
        individuals_per_setup = pop_size // (2 * levels_num)
        remainder = pop_size - individuals_per_setup * 2 * levels_num
        remainder_setups = self.generator.sample(
            [(d // 2, d % 2, (d + 1) % 2) for d in range(2 * levels_num)],
            remainder)
        remainder_setups.sort(reverse=True)

        RampedHalfHalfInitializer.LOG.info('%d levels', levels_num)
        RampedHalfHalfInitializer.LOG.info('%d regular individuals per level',
                                           individuals_per_setup)
        RampedHalfHalfInitializer.LOG.info('%d remaining individuals',
                                           len(remainder_setups))

        pop = []
        annotations_set = set()
        for d in range(levels_num):
            max_depth = self.min_depth + d
            RampedHalfHalfInitializer.LOG.debug('Initializing %d. level; '
                                                'max. depth = %d', d, max_depth)
            initializer.max_depth = max_depth
            g, f = 0, 0
            if remainder_setups and remainder_setups[-1][0] == d:
                _, g_, f_ = remainder_setups.pop()
                g += g_
                f += f_
            if remainder_setups and remainder_setups[-1][0] == d:
                _, g_, f_ = remainder_setups.pop()
                g += g_
                f += f_

            # grow
            RampedHalfHalfInitializer.LOG.debug('Initializing %d individuals '
                                                'with grow method.',
                                                individuals_per_setup + g)
            initializer.min_depth = 0
            for _ in range(individuals_per_setup + g):
                ind = initializer.initialize()
                tries = self.max_tries
                annots = tuple(ind.get_annotations())
                while annots in annotations_set and tries >= 0:
                    ind = initializer.initialize()
                    tries -= 1
                annotations_set.add(annots)
                pop.append(ind)

            # full
            RampedHalfHalfInitializer.LOG.debug('Initializing %d individuals '
                                                'with full method.',
                                                individuals_per_setup + g)
            initializer.min_depth = max_depth
            for _ in range(individuals_per_setup + f):
                ind = initializer.initialize()
                tries = self.max_tries
                annots = tuple(ind.get_annotations())
                while annots in annotations_set and tries >= 0:
                    ind = initializer.initialize()
                    tries -= 1
                annotations_set.add(annots)
                pop.append(ind)
        RampedHalfHalfInitializer.LOG.info('Initialization complete.')
        return pop
