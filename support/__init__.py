# -*- coding: utf8 -*-
"""This package contains support modules for evolutionary algorithms.
"""


class Stats(object):
    """A base class for saving statistics of a run of an evolutionary
    algorithm.

    This class itself does nothing so this is to be used where no saving is
    needed.
    """

    def save_message(self, iteration, message):
        """Saves an arbitrary message at the given iteration.

        :param int iteration: iteration of the algorithm to save the message
            for
        :param str message: the message to save
        """
        pass

    def save_bsf(self, iteration, bsf):
        """Saves a best-so-far individual ``bsf`` at the given ``iteration``.

        :param int iteration: iteration of the algorithm to save the bsf for
        :param bsf: the individual to save
        :type bsf: :class:`wopt.evo.Individual`
        """
        pass

    def save_population(self, iteration, population):
        """Saves the ``population`` at the given ``iteration``.

        :param int iteration: iteration of the algorithm to save the population
            for
        :param iterable population: individuals to save
        """
        pass


class MemoryStats(Stats):
    """A :class:`Stats` class saving the stats to memory (i.e. python
    structures).
    """

    def __init__(self):
        Stats.__init__(self)
        self.bsfs = []
        self.pops = []
        self.messages = []

    def save_message(self, iteration, message):
        self.messages.append((iteration, message))

    def save_bsf(self, iteration, bsf):
        self.bsfs.append((iteration, bsf))

    def save_population(self, iteration, population):
        self.pops.append((iteration, list(population)))


class ResourceHoldingStats(Stats):
    """A :class:`Stats` that holds some resources that have to be cleaned up.
    This class has no funtion and therefore must be subclassed.

    This class provides the :meth:`.cleanup()` method which should take care
    of cleaning up the resources.
    """

    def cleanup(self):
        pass


class SimpleFileStats(ResourceHoldingStats):
    """A :class:`Stats` class saving the stats to a file in a very simple
    format:

    .. code-block:: none

        MSG:<iteration>|<message>
        BSF:<iteration>|<fitness>|<bsf>
        POP:<iteration>|<individual1.fitness>;<individual1>|...
            ...<individual2.fitness><individual2>|...

    where ``MSG`` or``BSF`` or ``POP`` signals whether this line contains a
    message or the best-so-far individual or a population; ``<iteration>``
    stands for the iteration number, ``<fitness>`` stands for the fitness of
    the best-so-far individual, ``<bsf>`` stands for the string reperesentation
    of the best-so-far individual (its ``__str__()`` method is called) and
    ``<individualX>`` and ``<individualX.fitness>`` stand for the Xth
    individual of the population and its fitness.
    """

    def __init__(self, stats_file, field_separator='|', manage=None):
        """
        :param stats_file: Either a string or a file-like object. If it is a
            string a file with a file name of this string will be created,
            opened and closed in the :meth:`.cleanup()` method. If it is not a
            string it is assumed to be a file-like object and it will be used
            directly and will not be closed in the :meth:`.cleanup()`. This
            behavior can be overriden by setting the ``manage`` argument.
        :param field_separator: set to use different separator than ``|``
            (which is the default)
        :param bool manage: If true, the file will always be closed in the
        :meth:`.cleanup()` method. If false, it will never be closed in that
        method. If ``None``...
        """
        if isinstance(stats_file, str):
            self.stats_file = open(stats_file, mode='w')
            self.manage = False
        else:
            self.stats_file = stats_file
            self.manage = True

        if manage is not None:
            self.manage = manage

        self.field_separator = field_separator
        self.msg_template = 'MSG:{0}' + field_separator + '{1}\n'
        self.bsf_template = ('BSF:{0}' + field_separator + '{1}' +
                             field_separator + '{2}\n')
        self.pop_template = ('POP:{0}' + field_separator + '{1}\n')

    def save_message(self, iteration, message):
        self.stats_file.write(self.msg_template.format(iteration, message))
        self.stats_file.flush()

    def save_bsf(self, iteration, bsf):
        self.stats_file.write(self.bsf_template.format(iteration,
                                                       bsf.get_fitness(),
                                                       bsf.__str__()))
        self.stats_file.flush()

    def save_population(self, iteration, population):
        indivs = []
        for i in population:
            indivs.append(i.__str__())
        pop_str = self.field_separator.join(indivs)
        self.stats_file.write(self.pop_template.format(iteration, pop_str))
        self.stats_file.flush()

    def cleanup(self):
        if self.manage:
            self.stats_file.close()


class MultiplingStats(ResourceHoldingStats):
    """A :class:`Stats` class saving the stats to all "sub-stats". This is
    useful e.g. for saving stats both to memory and file, or print the stats
    to stderr/out in addition to other stats saving.
    """

    def __init__(self, *args):
        self.substats = args

    def save_message(self, iteration, message):
        for s in self.substats:
            s.save_message(iteration, message)

    def save_bsf(self, iteration, bsf):
        for s in self.substats:
            s.save_bsf(iteration, bsf)

    def save_population(self, iteration, population):
        for s in self.substats:
            s.save_population(iteration, population)

    def cleanup(self):
        for s in self.substats:
            if isinstance(s, ResourceHoldingStats):
                s.cleanup()
