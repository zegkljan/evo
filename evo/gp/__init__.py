# -*- coding: utf8 -*-
""" This package contains an implementation of classical Koza-style Genetic
Programming.
"""

import logging
import multiprocessing
import pprint
import gc
import time
import random

import evo
import evo.gp.support
import evo.utils
import evo.utils.tree


# noinspection PyAbstractClass
class Gp(multiprocessing.context.Process):
    """This class forms the whole GE algorithm.
    """

    LOG = logging.getLogger(__name__ + '.Gp')

    @staticmethod
    def generations(g):
        return lambda gp: gp.iterations >= g

    @staticmethod
    def time(seconds):
        return lambda gp: time.time() - gp.start_time >= seconds

    @staticmethod
    def any(*args):
        return lambda gp: any(arg(gp) for arg in args)

    @staticmethod
    def all(*args):
        return lambda gp: all(arg(gp) for arg in args)

    def __init__(self, fitness: evo.Fitness, pop_strategy, selection_strategy,
                 population_initializer, functions, terminals, stop, name=None,
                 **kwargs):
        """The optional keyword argument ``generator`` can be used to pass a
        random number generator. If it is ``None`` or not present a standard
        generator is used which is the :mod:`random` module and its
        functions. If a generator is passed it is expected to have the
        methods corresponding to the :mod:`random` module (i.e. the class
        :class:`random.Random`).

        .. warning::

            The generator (does not matter whether a custom or the default
            one is used) is assumed that it is already seeded and no seed is
            set inside this class.

        :param evo.Fitness fitness: fitness used to evaluate individual
            performance
        :param pop_strategy: population handling strategy - determines the size
            of the (parent) population, the number of offspring in each
            iteration and how these should be combined with the parent
            population
        :type pop_strategy: :class:`evo.PopulationStrategy`
        :param selection_strategy: selection strategy - the algorithm of
            selection of "good" individuals from the population
        :type selection_strategy: :class:`evo.SelectionStrategy`
        :param population_initializer: initializer used to initialize the
            initial population
        :type population_initializer:
            :class:`ge.init.PopulationInitializer`
        :param functions: a list of functions available for the program
            synthesis
        :type functions: :class:`list` of :class:`evo.gp.support.GpNode`
        :param terminals: a list of terminals (i.e. zero-arity functions)
            available for the program synthesis
        :type functions: :class:`list` of :class:`evo.gp.support.GpNode`
        :param stop: Either a number or a callable. If it is number:

                The number of generations the algorithm will run for. One
                generation is when ``pop_size`` number of individuals were
                created and put back to the population. In other words,
                if the algorithm runs in generational mode then one
                generation is one iteration of the algorithm; if the
                algorithm runs in steady-state then one generation is half
                the ``pop_size`` iterations (because each iteration two
                individuals are selected, possibly crossed over and put back
                into the population).

            If it is a callable:

                The callable will be called at the beginning of each
                iteration of the algorithm with one argument which is the
                algorithm instance (i.e. instance of this class). If the
                return value is evaluated as ``True`` then the algorithm stops.
        :param str name: name of the process (see
            :class:`multiprocessing.Process`)
        :keyword generator: (keyword argument) a random number generator; if
            ``None`` or not present calls to the methods of standard python
            module :mod:`random` will be performed instead
        :type generator: :class:`random.Random` , or ``None``
        :keyword crossover_prob: (keyword argument) probability of performing a
            crossover; if it does not fit into interval [0, 1] it is set to 0 if
            lower than 0 and to 1 if higher than 1; default value is 0.8
        :keyword crossover_type: (keyword argument) the type of crossover,
            for details see :ref:`Crossover types <evo.gp.Gp.xover-types>`

            The default value is ``'subtree'``.
        :keyword mutation_prob: (keyword argument) probability of performing
            a mutation; if it does not fit into interval [0, 1] it is set to
            0 if lower than 0 and to 1 if higher than 1; default value is 0.1
        :keyword mutation_type: (keyword argument) the type of mutation, for
            details see :ref:`Mutation types <evo.gp.Gp.mutation-types>`

            The default value is ``('subtree', 5)``.
        :keyword limits: specifies the size limits for the individuals, for
            details see :ref:`Limits <evo.gp.Gp.limits>`

            The default is no limits.
        :keyword evo.support.Stats stats: stats saving class
        :keyword callback: a callable which will be called at the beginning of
            every generation with a single argument which is the algorithm
            instance itself (i.e. instance of this class)

        .. _evo.gp.Gp.xover-types:

        .. rubric:: Crossover types

        The available crossover types are:

            * subtree crossover,
            * rate-based high-level crossover,
            * probabilistic meta-crossover,
            * custom crossover.

        *Subtree crossover*

        Classical Koza-style subtree crossover, i.e. a node is randomly
        chosen at each tree and the subtrees beneath these nodes are swapped
        between the trees. If a parent has more than 1 gene, the node is
        picked from a random gene. The crossover observes the
        :ref:`Limits <evo.gp.Gp.limits>`\ .

        To use this crossover set the ``crossover_type`` argument to
        ``'subtree'``\ .

        *Rate-based high-level crossover*

        This type of crossover operates on the gene level. It swaps the whole
        genes between the individuals.

        Each gene in the parents is selected for the crossover with a
        probability ``cr_rate``\ . Then the genes that were selected are
        swapped between the parents. If any of the parents' number of genes
        exceeds the limit ``max-genes`` (defined in
        :ref:`Limits <evo.gp.Gp.limits>`) then the excess genes (from the end)
        are thrown away.

        This crossover assumes that the order of the genes is not important.
        Also, if the number of genes in both parents is 1, subtree crossover
        is used instead.

        To use this crossover set the ``crossover_type`` argument to
        ``('cr-high-level', cr_rate)`` and substitute ``cr_rate`` for the actual
        crossover rate.

        *Probabilistic meta-crossover*

        The probabilistic crossover is, in fact, not a crossover method but is
        composed of a number of other crossover methods along with
        probabilites for each of them. In the end, each of the methods is
        performed with a probability assigned to it.

        To use this crossover set the ``crossover_type`` argument to
        ``('probabilistic`, (p1, m1), (p2, m2), ...)``\ . The ``p1``\ ,
        ``p2``\ , etc. are to be substituted with probabilities, the ``m1``\,
        ``m2``\ , etc. are to be substituted with the individual crossover
        methods as if they were assigned to the ``crossover_type`` argument
        themselves. If the probabilites don't sum up to 1 they are going to
        be scaled so that they do.

        Example: ``('probabilistic', (.4, 'subtree'), (.4, ('cr-high-level',
        .5, 4)))`` translates to a probabilistic crossover where with
        probability 0.4 the subtree crossover will be performed  and with
        probability 0.6 the rate-based high-level crossover will be used.

        *Custom crossover*

        A custom crossover specified by a callable which will receive the
        parents as its arguments. This crossover can be used in
        meta-crossover methods as any other crossover method.

        The callable should take three arguments: the :class:`Gp` object of
        the running algorithm, the first parent and the second parent,
        and it should return the offspring in a list. The parents are copied
        prior to their passing into the callable so any modification of them
        is safe with respect to other individuals in the population.

        To use this crossover set the ``crossover_type`` argument to
        ``('custom', callable)`` and replace ``callable`` with a callable of
        three arguments.

        .. warning:

            It is the job of the callable to unset the offspring's fitness.

        .. _evo.gp.Gp.mutation-types:

        .. rubric:: Mutation types

        The available mutation types are:

            * subtree mutation,
            * custom mutation.

        *Subtree mutation*

        Classical Koza-style subtree mutation, i.e. a node is randomly
        selected and is (along with its subtree) replaced with a randomly
        generated subtree.

        To use this mutation set the ``mutation_type`` argument to
        ``('subtree', max_depth)`` and replace ``max_depth`` with the desired
        maximum depth of the newly generated subtree. Nevertheless,
        the mutation sill observes the total depth and nodes limit (see
        :ref:`Limits <evo.gp.Gp.limits>`).

        *Custom mutation*

        A custom mutation specified by a callable which will receive the
        parent as its argument.

        The callable should take two arguments: the :class:`Gp` object of
        the running algorithm and the parent, and it should return the
        offspring. The parent is copied prior to its passing into the
        callable so any modification of it is safe with respect to other
        individuals in the population.

        To use this mutation set the ``mutation_type`` argument to
        ``('custom', callable)`` and replace ``callable`` with a callable of
        two arguments.

        .. warning:

            It is the job of the callable to unset the offspring's fitness.

        .. _evo.gp.Gp.limits:

        .. rubric:: Limits

        The limits specify constraints on the size of the individuals. The
        limits are specified via a dictionary where the keys are the names of
        the limits (strings) and the values specify the particular limit.

        *Maximum depth*

        :Name: ``'max-depth'``

        This limit specifies the maximum depth an individual can have. It is
        a hard limit, i.e. **no** individual will ever be deeper than this
        limit. For multi-gene individuals, the limit applies per-gene.

        *Maximum number of nodes*

        :Name: ``'max-nodes'``

        This limit specifies the maximum number of nodes an individual can
        have. It is a hard limit, i.e. **no** individual will ever be deeper
        than this limit. For multi-gene individuals, the limit applies per-gene.

        *Maximum number of genes*

        :Name: ``'max-genes''``

        This limit specifies the maximum number of genes an individual can
        have. It is a hard limit, i.e. **no** individual will ever have more
        genes than this limit specifies.

        """
        multiprocessing.context.Process.__init__(self, name=name)

        # Positional args
        self.fitness = fitness
        self.pop_strategy = pop_strategy
        self.selection_strategy = selection_strategy
        self.functions = functions
        self.terminals = terminals
        self.population_initializer = population_initializer
        if isinstance(stop, int):
            # noinspection PyProtectedMember
            self.stop = self.generations(stop)
        elif callable(stop):
            self.stop = stop
        else:
            raise TypeError('Argument stop is neither integer nor callable.')

        # Keyword args
        self.generator = random
        if 'generator' in kwargs:
            self.generator = kwargs['generator']

        self.crossover_prob = 0.8
        if 'crossover_prob' in kwargs:
            self.crossover_prob = kwargs['crossover_prob']
            self.crossover_prob = max(0, self.crossover_prob)
            self.crossover_prob = min(1, self.crossover_prob)

        self.crossover_method = self.subtree_crossover
        self.crossover_method_args = ()
        if 'crossover_type' in kwargs and kwargs['crossover_type'] is not None:
            self.crossover_method, self.crossover_method_args = \
                self.setup_crossover(kwargs['crossover_type'])

        self.mutation_prob = 0.1
        if 'mutation_prob' in kwargs:
            self.mutation_prob = kwargs['mutation_prob']
            self.mutation_prob = max(0, self.mutation_prob)
            self.mutation_prob = min(1, self.mutation_prob)

        self.mutate_method = self.subtree_mutate
        self.mutate_method_args = (5,)
        if 'mutation_type' in kwargs and kwargs['mutation_type'] is not None:
            self.mutate_method, self.mutate_method_args = \
                self.setup_mutation(kwargs['mutation_type'])

        self.limits = {'max-genes': 1,
                       'max-depth': float('inf'),
                       'max-nodes': float('inf')}
        if 'limits' in kwargs:
            self.limits.update(kwargs['limits'])

        self.stats = None
        if 'stats' in kwargs:
            self.stats = kwargs['stats']

        self.callback = None
        if 'callback' in kwargs:
            self.callback = kwargs['callback']
            if not callable(self.callback):
                raise TypeError('Keyword argument callback is not a callable.')

        self.population = []
        self.population_sorted = False

        self.start_time = None
        """
        The time of start of the algorithm, including the population
        initialisation.
        """
        self.end_time = None
        """
        The time of end of the algorithm, excluding the final garbage
        collection, cleanup, etc.
        """
        self.iterations = 0
        """
        The number of elapsed iterations of the algorithm (either generations
        in the generational mode or just iterations in the steady-state mode).
        """

    def run(self):
        """Runs the GE algorithm.
        """
        Gp.LOG.info('Starting algorithm.')
        try:
            self.start_time = time.time()
            self.population = self.population_initializer.initialize(
                self.pop_strategy.get_parents_number(), self.limits)

            self._run()
        finally:
            self.end_time = time.time()
            if self.fitness.get_bsf() is None:
                Gp.LOG.info('Finished. No BSF acquired.')
            else:
                Gp.LOG.info('Finished. Fitness: %f %s %s',
                            self.fitness.get_bsf().get_fitness(),
                            pprint.pformat(self.fitness.get_bsf().get_data()),
                            str(self.fitness.get_bsf().genotype))
            Gp.LOG.info('Performing garbage collection.')
            gc.collect()
            try:
                if self.stats is not None:
                    self.stats.cleanup()
            except AttributeError:
                pass

    def _run(self):
        Gp.LOG.info('Starting evolution.')
        while not self.stop(self):
            Gp.LOG.debug('Starting iteration %d', self.iterations)
            if self.callback is not None:
                self.callback(self)

            elites = self.top_individuals(self.pop_strategy.get_elites_number())

            Gp.LOG.debug('Processing selection.')
            offspring = []
            while len(offspring) < self.pop_strategy.get_offspring_number():
                a = self.selection_strategy.select_single(self.population)[1]
                if self.generator.random() < self.crossover_prob:
                    b = self.selection_strategy.select_single(
                        self.population)[1]
                    children = self.crossover(a.copy(), b.copy())
                else:
                    children = [a]

                while (children and len(offspring) <
                       self.pop_strategy.get_offspring_number()):
                    o = children.pop()
                    if self.generator.random() < self.mutation_prob:
                        o = self.mutate(o.copy())
                    offspring.append(o)
            self.population = self.pop_strategy.combine_populations(
                self.population, offspring, elites)
            Gp.LOG.info('Finished iteration %d time %.1f. Best fitness: %f | '
                        '%s | %s',
                        self.iterations, time.time() - self.start_time,
                        self.fitness.get_bsf().get_fitness(),
                        str(self.fitness.get_bsf()),
                        self.fitness.get_bsf().get_data())
            self.iterations += 1
        if self.callback is not None:
            self.callback(self)
        Gp.LOG.info('Finished evolution.')

    def setup_crossover(self, crossover_type):
        """Helper method for the constructor which sets up the crossover method.
        """
        try:
            valid = True
            if crossover_type == 'subtree':
                crossover_method = self.subtree_crossover
                crossover_method_args = ()
            elif crossover_type[0] == 'cr-high-level':
                crossover_method = self.cr_high_level_crossover
                crossover_method_args = (crossover_type[1],)
            elif crossover_type[0] == 'probabilistic':
                crossover_method = self.probabilistic_crossover
                probs_methods = []
                for prob, subcrossover in crossover_type[1:]:
                    cm, cma = self.setup_crossover(subcrossover)
                    if not probs_methods:
                        probs_methods.append([prob, (cm, cma)])
                    else:
                        probs_methods.append([probs_methods[-1][0] + prob,
                                              (cm, cma)])
                for i in range(len(probs_methods)):
                    probs_methods[i][0] = (probs_methods[i][0] /
                                           probs_methods[-1][0])

                crossover_method_args = (tuple(probs_methods),)
                return crossover_method, crossover_method_args
            elif crossover_type[0] == 'custom':
                crossover_method = self.custom_crossover
                crossover_method_args = crossover_type[1]
            else:
                valid = False
        except Exception as e:
            raise ValueError('Invalid crossover type.', e)

        if not valid:
            raise ValueError('Invalid crossover type.')

        # noinspection PyUnboundLocalVariable
        return crossover_method, crossover_method_args

    def setup_mutation(self, mutation_type):
        """Helper method for the constructor which sets up the mutation method.
        """
        try:
            valid = True
            if mutation_type[0] == 'subtree':
                mutation_method = self.subtree_mutate
                mutation_method_args = (mutation_type[1],)
            elif mutation_type[0] == 'custom':
                mutation_method = self.custom_mutate
                mutation_method_args = (mutation_type[1],)
            else:
                valid = False
        except Exception as e:
            raise ValueError('Invalid crossover type.', e)

        if not valid:
            raise ValueError('Invalid crossover type.')

        # noinspection PyUnboundLocalVariable
        return mutation_method, mutation_method_args

    def crossover(self, o1, o2):
        """Performs a crossover of two individuals.

        :param evo.gp.support.TreeIndividual o1: first parent
        :param evo.gp.support.TreeIndividual o2: second parent
        """
        Gp.LOG.debug('Performing crossover of individuals %s, %s', o1, o2)
        assert self.crossover_method is not None
        # noinspection PyArgumentList
        return self.crossover_method(o1, o2, *self.crossover_method_args)

    def mutate(self, i):
        """Performs a mutation of the individual.

        The individual is mutated in place and also returned.

        :param evo.gp.support.TreeIndividual i: the individual to be mutated
        :return: the mutated individual
        :rtype: :class:`evo.gp.support.TreeIndividual`
        """
        return self.mutate_method(i, *self.mutate_method_args)

    def subtree_crossover(self, o1, o2):
        if o1.genes_num == 1:
            k1 = 0
        else:
            k1 = self.generator.randrange(o1.genes_num)
        if o2.genes_num == 1:
            k2 = 0
        else:
            k2 = self.generator.randrange(o2.genes_num)

        max_depth = self.limits['max-depth']
        max_size = self.limits['max-nodes']

        g1 = o1.genotype[k1]
        size_1 = g1.get_subtree_size()
        nodes_depth_1 = g1.get_nodes_bfs(compute_depths=True)
        g2 = o2.genotype[k2]
        size_2 = g2.get_subtree_size()
        nodes_depth_2 = g2.get_nodes_bfs(compute_depths=True)

        nodes_depth_f_2 = []
        while not nodes_depth_f_2:
            point_1, point_depth_1 = self.generator.choice(nodes_depth_1)
            point_tree_depth_1 = point_1.get_subtree_depth()
            point_tree_size_1 = point_1.get_subtree_size()

            for point_2, point_depth_2 in nodes_depth_2:
                if point_tree_depth_1 + point_depth_2 - 1 > max_depth:
                    continue
                if point_depth_1 + point_2.get_subtree_depth() - 1 > max_depth:
                    continue
                if size_1 - point_tree_size_1 + point_2.get_subtree_size() > \
                        max_size:
                    continue
                if point_tree_size_1 + size_2 - point_2.get_subtree_size() > \
                        max_size:
                    continue
                nodes_depth_f_2.append((point_2, point_depth_2))

        point_2, _ = self.generator.choice(nodes_depth_f_2)

        # noinspection PyUnboundLocalVariable
        root_1, root_2 = evo.gp.support.swap_subtrees(point_1, point_2)
        o1.genotype[k1] = root_1
        o2.genotype[k2] = root_2

        o1.set_fitness(None)
        o2.set_fitness(None)
        return [o1, o2]

    def cr_high_level_crossover(self, o1, o2, rate):
        max_genes = self.limits['max-genes']
        if o1.genes_num == 1 and o2.genes_num == 1:
            return self.subtree_crossover(o1, o2)

        from_g1 = []
        g1 = []
        for g in o1.genotype:
            if self.generator.random() < rate:
                from_g1.append(g)
            else:
                g1.append(g)
        if len(g1) == 0:
            g1.append(from_g1.pop(self.generator.randrange(len(from_g1))))

        from_g2 = []
        g2 = []
        for g in o2.genotype:
            if self.generator.random() < rate:
                from_g2.append(g)
            else:
                g2.append(g)
        if len(g2) == 0:
            g2.append(from_g2.pop(self.generator.randrange(len(from_g2))))

        g1.extend(from_g2[:min(len(from_g2), max_genes - len(g1))])
        g2.extend(from_g1[:min(len(from_g1), max_genes - len(g2))])

        o1.genotype = g1
        o2.genotype = g2

        o1.set_fitness(None)
        o2.set_fitness(None)
        return [o1, o2]

    def probabilistic_crossover(self, o1, o2, probs_methods):
        r = self.generator.random()
        method = None
        for p, m in probs_methods:
            method = m
            if p >= r:
                break
        return method[0](o1, o2, *method[1])

    def custom_crossover(self, o1, o2, external):
        return external(self, o1, o2)

    def subtree_mutate(self, i, max_depth):
        if i.genes_num == 1:
            k = 0
        else:
            k = self.generator.randrange(i.genes_num)

        g = i.genotype[k]
        s = g.get_subtree_size()
        nodes_depths = g.get_nodes_bfs(compute_depths=True)
        n, d = self.generator.choice(nodes_depths)
        ns = n.get_subtree_size()
        subtree = evo.gp.support.generate_tree(self.functions, self.terminals,
                                               min(self.limits['max-depth'] -
                                                   d + 1,
                                                   max_depth),
                                               self.limits['max-nodes'] - s +
                                               ns,
                                               self.generator, False)
        i.genotype[k] = evo.gp.support.replace_subtree(n, subtree)
        i.set_fitness(None)
        return i

    def custom_mutate(self, i, external):
        return external(self, i)

    def top_individuals(self, k):
        Gp.LOG.debug('Obtaining top %d individuals...', k)
        if k <= 0:
            return []
        kth = evo.utils.select(self.population, k - 1,
                               cmp=lambda a, b: self.fitness.is_better(a, b))
        tops = []
        for i in self.population:
            if self.fitness.is_better(i, kth) or self.fitness.is_better(kth, i):
                tops.append(i)
            if len(tops) == k:
                break
        Gp.LOG.debug('Obtained top individuals: %s', str(tops))
        return tops
