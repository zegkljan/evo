# -*- coding: utf8 -*-
"""Support classes for the :mod:`evo.ge` package.
"""

import functools
import random
import fractions

import evo
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

        :param grammar: the grammar to generate
        :type grammar: :class:`evo.support.grammar.Grammar`
        :keyword generator: a random number generator; if ``None`` or not
            present calls to the methods of standard python module
            :mod:`random` will be performed instead
        :type generator: :mod:`random`\\ .Random or ``None``
        :keyword max_depth: maximum depth of the corresponding derivation tree;
            if ``None`` or not present default value of infinity is used
        :keyword multiplier: number which will be used to multiply the LCM of
            all choices numbers to get a higher maximum codon value (default
            is 1, i.e. maximum codon value will be a LCM of numbers of all
            choices in the grammar)
        :return: a randomly generated individual
        :rtype: :class:`DerivationTreeIndividual`
        """
        evo.IndividualInitializer.__init__(self)

        self.grammar = grammar

        if 'generator' in kwargs:
            self.generator = kwargs['generator']
        else:
            self.generator = random

        if 'max_depth' in kwargs:
            self.max_depth = kwargs['max_depth']
        else:
            self.max_depth = float('inf')

        if 'multiplier' in kwargs:
            self.multiplier = kwargs['multiplier']
        else:
            self.multiplier = 1

        choice_nums = [r.get_choices_num() for r in grammar.get_rules()]
        m = functools.reduce(lambda a, b: a * b // fractions.gcd(a, b),
                             choice_nums)
        self.max_choices = m * self.multiplier

    def initialize(self):
        iterator = evo.utils.random.RandomIntIterable(
            -1, -1, 0, self.max_choices - 1, generator=self.generator)
        sequence = []
        _ = self.grammar.to_tree(decisions=iterator,
                                 max_wraps=0,
                                 max_depth=self.max_depth,
                                 sequence=sequence)
        return CodonGenotypeIndividual(sequence,
                                       self.max_choices)
