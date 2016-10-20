# -*- coding: utf8 -*-
""" This package contains an implementation of classical Koza-style Genetic
Programming.
"""

import enum
import gc
import logging
import multiprocessing
import pprint
import random
import time

import evo
import evo.gp.support
import evo.utils
import evo.utils.tree


class Error(Exception):
    """Base class for exceptions in this module"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class OperatorNotApplicableError(Error):
    """Risen when an operator that should be applied is not applicable in the
    given context."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CrossoverOperator(object):
    """Base class for crossover operators."""

    def get_parents_number(self) -> int:
        """Returns the number of parents required for the crossover.

        Returning ``None`` means that the operator does not require any
        particular number (any number is valid).
        """
        raise NotImplementedError()

    def get_children_number(self) -> int:
        """Returns the number of children produced by the crossover.
        """
        raise NotImplementedError()

    def crossover(self, *parents):
        """Performs the crossover.

        :raises OperatorNotApplicableError: if the crossover cannot be applied
            to the given parents
        """
        raise NotImplementedError()


class MutationOperator(object):
    """Base class for mutation operators."""

    def mutate(self, individual):
        """Performs the mutation.

        :raises OperatorNotApplicableError: if the mutation cannot be applied to
            the given individual
        """
        raise NotImplementedError()


class InapplicableCrossover(CrossoverOperator):
    """A crossover operator that is never applicable.

    This operator always raises the :class:`OperatorNotApplicableError`\ ,
    regardless of the parents.
    """

    def get_parents_number(self) -> int:
        return None

    def get_children_number(self) -> int:
        return None

    def crossover(self, *parents):
        raise OperatorNotApplicableError('InapplicableCrossover.crossover() '
                                         'called')


class SubtreeCrossover(CrossoverOperator):
    """Koza-style subtree crossover.

    A random node is chosen in each of the parents. The subtrees rooting in
    these nodes are then swapped.
    """

    LOG = logging.getLogger(__name__ + '.SubtreeCrossover')

    def __init__(self, generator, limits):
        self.generator = generator
        self.limits = limits

    def get_parents_number(self) -> int:
        return 2

    def get_children_number(self) -> int:
        return 2

    def crossover(self, *parents):
        if len(parents) != 2:
            raise ValueError('Two parents required.')
        o1, o2 = parents
        SubtreeCrossover.LOG.debug(
            'Performing subtree crossover of individuals %s, %s', o1, o2)
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


class CrHighlevelCrossover(CrossoverOperator):
    """Rate-based high-level (multigene) crossover operator.

    Each gene in the parents is chosen with a given probability (rate) to be
    switched with the other parent. Excessive genes are thrown away.
    """

    LOG = logging.getLogger(__name__ + '.CrHighlevelCrossover')

    def __init__(self, rate, generator, limits):
        self.rate = rate
        self.generator = generator
        self.limits = limits

    def get_parents_number(self) -> int:
        return 2

    def get_children_number(self) -> int:
        return 2

    def crossover(self, *parents):
        if len(parents) != 2:
            raise ValueError('Two parents required.')
        o1, o2 = parents
        CrHighlevelCrossover.LOG.debug(
            'Performing high-level crossover of individuals %s, %s', o1, o2)
        max_genes = self.limits['max-genes']
        if o1.genes_num == 1 and o2.genes_num == 1:
            raise OperatorNotApplicableError('Both parents have only one gene.')

        from_g1 = []
        g1 = []
        c_from_g1 = []
        c_g1 = []
        if not hasattr(o1, 'coefficients'):
            c_from_g1 = None
            c_g1 = None
        for i, g in enumerate(o1.genotype):
            if self.generator.random() < self.rate:
                from_g1.append(g)
                if c_from_g1 is not None:
                    c_from_g1.append(o1.coefficients[i])
            else:
                g1.append(g)
                if c_g1 is not None:
                    c_g1.append(o1.coefficients[i])
        if len(g1) == 0:
            n = self.generator.randrange(len(from_g1))
            g1.append(from_g1.pop(n))
            if c_from_g1 is not None and c_g1 is not None:
                c_g1.append(c_from_g1.pop(n))

        from_g2 = []
        g2 = []
        c_from_g2 = []
        c_g2 = []
        if not hasattr(o2, 'coefficients'):
            c_from_g2 = None
            c_g2 = None
        for i, g in enumerate(o2.genotype):
            if self.generator.random() < self.rate:
                from_g2.append(g)
                if c_from_g2 is not None:
                    c_from_g2.append(o2.coefficients[i])
            else:
                g2.append(g)
                if c_g2 is not None:
                    c_g2.append(o2.coefficients[i])
        if len(g2) == 0:
            n = self.generator.randrange(len(from_g2))
            g2.append(from_g2.pop(n))
            if c_from_g2 is not None and c_g2 is not None:
                c_g2.append(c_from_g2.pop(n))

        g1.extend(from_g2[:min(len(from_g2), max_genes - len(g1))])
        g2.extend(from_g1[:min(len(from_g1), max_genes - len(g2))])

        o1.genotype = g1
        o2.genotype = g2

        if c_g1 is not None and c_g2 is not None:
            c_g1.extend(c_from_g2[:min(len(c_from_g2), max_genes - len(c_g1))])
            c_g2.extend(c_from_g1[:min(len(c_from_g1), max_genes - len(c_g2))])
            o1.coefficients = c_g1
            o2.coefficients = c_g2

        o1.set_fitness(None)
        o2.set_fitness(None)
        return [o1, o2]


class StochasticChoiceCrossover(CrossoverOperator):
    """Compound crossover operator, chooses one of actual operators based on
    probability.

    This crossover is compound of multiple crossover operators, each associated
    with a probability of it being chosen. When a crossover is to happen the
    actual operator is chosen randomly w.r.t. the associated probabilities.

    *Fallback method* is a crossover operator that will be used in case the
    (stochastically) selected operator is not applicable in the given context
    (it raises :class:`OperatorNotApplicableError`\ ). If that happens, the
    fallback method is used. If the fallback method is not set, no crossover
    happens, i.e. the parents are returned (but it is the responsibility of the
    particular crossover methods to not change the parents before raising that
    error). If the fallback method is also not applicable then the error is
    raised up but this practice is strongly discouraged.
    """

    LOG = logging.getLogger(__name__ + '.StochasticChoiceCrossover')

    def __init__(self, probs_crossovers, generator, fallback_method=None):
        """
        :param probs_crossovers: a list of pairs (tuples) where the first
            element of each pair is a probability and
            the second element is a :class:`CrossoverOperator` instance
        :param generator: random number generator compliant with the
            :class:`random.Random` class.
        :param fallback_method: a :class:`CrossoverOperator` instance that is to
            be used in case the chosen operator is not applicable in the given
            context (it raises :class:`OperatorNotApplicableError`\ ); if
            ``None`` no operator will be applied (parents will be unchanged)
        """
        self.probs_crossovers = []
        total = sum(map(lambda e: e[0], probs_crossovers))
        for p, c in probs_crossovers:
            if self.probs_crossovers:
                last_c = self.probs_crossovers[-1][1]
                if last_c.get_parents_number() != c.get_parents_number():
                    raise ValueError('All crossovers must use the same number '
                                     'of parents')
                if last_c.get_children_number() != c.get_children_number():
                    raise ValueError('All crossovers must produce the same '
                                     'number of children')
                last_p = self.probs_crossovers[-1][0]
            else:
                last_p = 0
            self.probs_crossovers.append((last_p + p / total, c))
        self.generator = generator
        self.fallback_method = fallback_method
        if self.fallback_method is not None:
            pn = self.get_parents_number()
            fpn = self.fallback_method.get_parents_number()
            if pn != fpn:
                raise ValueError('Fallback method must use the same number of '
                                 'parents as the main methods.')
            cn = self.get_children_number()
            fcn = self.fallback_method.get_children_number()
            if cn != fcn:
                raise ValueError('Fallback method must produce the same number '
                                 'of children as the main methods.')

    def get_parents_number(self) -> int:
        self.probs_crossovers[0][1].get_parents_number()

    def get_children_number(self) -> int:
        self.probs_crossovers[0][1].get_children_number()

    def crossover(self, *parents):
        StochasticChoiceCrossover.LOG.debug(
            'Performing probabilistic crossover of individuals %s', parents)
        r = self.generator.random()
        crossover = None
        for p, m in self.probs_crossovers:
            crossover = m
            if p >= r:
                break
        try:
            return crossover.crossover(*parents)
        except OperatorNotApplicableError as e:
            StochasticChoiceCrossover.LOG.debug(e)
            if self.fallback_method is None:
                return parents
            return self.fallback_method.crossover(*parents)


class InapplicableMutation(MutationOperator):
    """A mutation operator that is never applicable.

    This operator always raises the :class:`OperatorNotApplicableError`\ ,
    regardless of the individual.
    """

    def mutate(self, individual):
        raise OperatorNotApplicableError('InapplicableMutation.crossover() '
                                         'called')


class SubtreeMutation(MutationOperator):
    """Koza-style subtree mutation operator.

    Randomly chooses a node in the individual, throws away it and the subtree
    under it and generates a new subtree in place of it.
    """

    LOG = logging.getLogger(__name__ + '.SubtreeMutation')

    def __init__(self, max_depth, generator, functions, terminals, limits):
        self.max_depth = max_depth
        self.generator = generator
        self.functions = functions
        self.terminals = terminals
        self.limits = limits

    def mutate(self, individual):
        """Performs the mutation."""
        SubtreeMutation.LOG.debug(
            'Performing subtree mutation of individual %s', individual)
        if individual.genes_num == 1:
            k = 0
        else:
            k = self.generator.randrange(individual.genes_num)

        g = individual.genotype[k]
        s = g.get_subtree_size()
        nodes_depths = g.get_nodes_bfs(compute_depths=True)
        n, d = self.generator.choice(nodes_depths)
        ns = n.get_subtree_size()
        subtree = evo.gp.support.generate_tree_full_grow(
            self.functions, self.terminals,
            min(self.limits['max-depth'] - d + 1, self.max_depth),
            self.limits['max-nodes'] - s + ns, self.generator, False)
        individual.genotype[k] = evo.gp.support.replace_subtree(n, subtree)
        individual.set_fitness(None)
        return individual


class StochasticChoiceMutation(MutationOperator):
    """Compound mutation operator, chooses one of actual operators based on
    probability.

    This mutation is compound of multiple mutation operators, each associated
    with a probability of it being chosen. When a mutation is to happen the
    actual operator is chosen randomly w.r.t. the associated probabilities.

    *Fallback method* is a mutation operator that will be used in case the
    (stochastically) selected operator is not applicable in the given context
    (it raises :class:`OperatorNotApplicableError`\ ). If that happens, the
    fallback method is used. If the fallback method is not set, no mutation
    happens, i.e. the original individual is returned (but it is the
    responsibility of the particular mutation methods to not change the
    individual before raising that error). If the fallback method is also not
    applicable then the error is raised up but this practice is strongly
    discouraged.
    """
    LOG = logging.getLogger(__name__ + '.StochasticChoiceMutation')

    def __init__(self, probs_mutations, generator, fallback_method=None):
        """
        :param probs_mutations: a list of pairs (tuples) where the first
            element of each pair is a probability and the second element is a
            :class:`MutationOperator` instance
        :param generator: random number generator compliant with the
            :class:`random.Random` class.
        :param fallback_method: a :class:`MutationOperator` instance that is to
            be used in case the chosen operator is not applicable in the given
            context (it raises :class:`OperatorNotApplicableError`\ ); if
            ``None`` no operator will be applied (parents will be unchanged)
        """

        self.probs_mutations = []
        total = sum(map(lambda e: e[0], probs_mutations))
        for p, m in probs_mutations:
            if self.probs_mutations:
                last_p = self.probs_mutations[-1][0]
            else:
                last_p = 0
            self.probs_mutations.append((last_p + p / total, m))
        self.generator = generator
        self.fallback_method = fallback_method

    def mutate(self, individual):
        StochasticChoiceMutation.LOG.debug(
            'Performing probabilistic mutation of individual %s', individual)
        r = self.generator.random()
        mutation = None
        for p, m in self.probs_mutations:
            mutation = m
            if p >= r:
                break
        try:
            return mutation.mutate(individual)
        except OperatorNotApplicableError as e:
            StochasticChoiceMutation.LOG.debug(e)
            if self.fallback_method is None:
                return individual
            return self.fallback_method.mutate(individual)


class PipelineMutation(MutationOperator):
    """A compound mutation operator that applies multiple mutations sequentially
    on the given individual.

    *Fallback behaviour* specifies what to do when a mutation is not applicable
    in the given context (it raises a :class:`OperatorNotApplicableError`\ ).
    There are four possibilities:

        * ``STOP_HARD`` - the result of the last applicable mutation is
          returned; if it is the first mutation, the whole pipeline is
          inapplicable
        * ``STOP_SOFT`` - the result of the last applicable mutation is
          returned; if it is the first mutation, the original individual is
          returned, i.e. no mutation happens
        * ``SKIP_HARD`` - the inapplicable mutation is skipped, following
          mutations are still applied; if no mutation is applicable, the whole
          pipeline is inapplicable
        * ``SKIP_SOFT`` - the inapplicable mutation is skipped, following
          mutations are still applied; if no mutation is applicable, the
          original individual is returned, i.e. no mutation happens

    ``SKIP_SOFT`` is the default.
    """

    class FallbackBehaviour(enum.Enum):
        STOP_HARD = 0
        STOP_SOFT = 1
        SKIP_HARD = 2
        SKIP_SOFT = 3

    LOG = logging.getLogger(__name__ + '.PipelineMutation')

    def __init__(self, mutations,
                 fallback_behaviour=FallbackBehaviour.SKIP_SOFT):
        """
        :param mutations: the list of mutations that form the pipeline
        :param fallback_behaviour: specifies how to behave when a mutation is
            not applicable
        """
        self.mutations = mutations
        self.fallback_behaviour = fallback_behaviour

    def mutate(self, individual):
        PipelineMutation.LOG.debug(
            'Performing pipeline mutation of individual %s', individual)
        i = 0
        for m in self.mutations:
            try:
                individual = m.mutate(individual)
                i += 1
            except OperatorNotApplicableError as e:
                PipelineMutation.LOG.debug(e)
                if (self.fallback_behaviour is
                        PipelineMutation.FallbackBehaviour.STOP_HARD):
                    if i == 0:
                        raise OperatorNotApplicableError(
                            'First mutation was not applicable in STOP_HARD '
                            'behaviour.') from e
                    else:
                        break
                elif (self.fallback_behaviour is
                      PipelineMutation.FallbackBehaviour.STOP_SOFT):
                    break
                elif (self.fallback_behaviour is
                      PipelineMutation.FallbackBehaviour.SKIP_HARD or
                      self.fallback_behaviour is
                      PipelineMutation.FallbackBehaviour.SKIP_SOFT):
                    continue

        if (self.fallback_behaviour is
                PipelineMutation.FallbackBehaviour.SKIP_HARD) and i == 0:
            raise OperatorNotApplicableError('No mutation was applicable in '
                                             'SKIP_HARD behaviour.')
        return individual


# noinspection PyAbstractClass
class GpReproductionStrategy(evo.ReproductionStrategy):
    """This class is a base class for reproduction strategies for use in Genetic
    Programming.

    It is not a full implementation of :class:`evo.ReproductionStrategy` and
    does not contain the actual reproduction rules (i.e. how are the operators
    used)."""

    LOG = logging.getLogger(__name__ + '.GpReproductionStrategy')

    def __init__(self, functions, terminals, generator=None,
                 crossover: CrossoverOperator=None,
                 mutation: MutationOperator=None,
                 limits=None):
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

        :param functions: a list of functions available for the program
            synthesis
        :type functions: :class:`list` of :class:`evo.gp.support.GpNode`
        :param terminals: a list of terminals (i.e. zero-arity functions)
            available for the program synthesis
        :type functions: :class:`list` of :class:`evo.gp.support.GpNode`
        :keyword generator: (keyword argument) a random number generator; if
            ``None`` or not present calls to the methods of standard python
            module :mod:`random` will be performed instead
        :type generator: :class:`random.Random` , or ``None``
        :keyword crossover: the crossover operator to use,
            for details see :ref:`Crossover types <evo.gp.Gp.xover-types>`

            The default value is ``'subtree'``.
        :keyword mutation: (keyword argument) the mutation operator to use, for
            details see :ref:`Mutation types <evo.gp.Gp.mutation-types>`

            The default value is ``('subtree', 5)``.
        :keyword limits: specifies the size limits for the individuals, for
            details see :ref:`Limits <evo.gp.Gp.limits>`

            The default is no limits.

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
        probabilities for each of them. In the end, each of the methods is
        performed with a probability assigned to it.

        To use this crossover set the ``crossover_type`` argument to
        ``('probabilistic`, (p1, m1), (p2, m2), ...)``\ . The ``p1``\ ,
        ``p2``\ , etc. are to be substituted with probabilities, the ``m1``\,
        ``m2``\ , etc. are to be substituted with the individual crossover
        methods as if they were assigned to the ``crossover_type`` argument
        themselves. If the probabilities don't sum up to 1 they are going to
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
        self.functions = functions
        self.terminals = terminals

        self.generator = random
        if generator is not None:
            self.generator = generator

        self.limits = {'max-genes': 1,
                       'max-depth': float('inf'),
                       'max-nodes': float('inf')}
        if limits is not None:
            self.limits.update(limits)

        if crossover is None:
            self.crossover = SubtreeCrossover(self.generator, self.limits)
        else:
            self.crossover = crossover

        if mutation is None:
            self.mutation = SubtreeMutation(5, self.generator, self.limits,
                                            self.functions, self.terminals)
        else:
            self.mutation = mutation


class IndependentReproductionStrategy(GpReproductionStrategy):
    """This class represents a reproduction strategy where the crossover and
    mutation are independent events.

    That means that with a probability **px** an individual is subject to
    crossover, then, regardless whether it was subject to crossover or not, it
    is subject to mutation with probability **pm**. Summed up, these are the
    possible "paths" an individual can take when reproducing by this strategy:

        * individual is just copied - no crossover, no mutation
        * individual is just mutated - no crossover
        * individual is just crossed over - no mutation, the individual is one
          of two offspring that are product of combining two parents
        * individual is crossed over and then mutated - the individual is one of
          two offspring that are product of combining two parents and and which
          is then mutated
    """

    LOG = logging.getLogger(__name__ + '.IndependentReproductionStrategy')

    def __init__(self, functions, terminals, generator=None,
                 crossover: CrossoverOperator=None,
                 mutation: MutationOperator=None,
                 limits=None, crossover_prob=0.8, mutation_prob=0.1):
        """
        :param crossover_prob: probability of performing a crossover; if it does
            not fit into interval [0, 1] it is clipped to fit; default value is
            0.8
        :param mutation_prob: probability of performing a mutation; if it does
            not fit into interval [0, 1] it is clipped to fit; default value is
            0.1

        .. seealso: :class:`GpReproductionStrategy`
        """
        super().__init__(functions, terminals, generator, crossover,
                         mutation, limits)
        self.crossover_prob = crossover_prob
        self.crossover_prob = max(0, self.crossover_prob)
        self.crossover_prob = min(1, self.crossover_prob)
        if self.crossover_prob != crossover_prob:
            IndependentReproductionStrategy.LOG.warning(
                'Crossover probability out of range. Clipped to [0, 1].')

        self.mutation_prob = mutation_prob
        self.mutation_prob = max(0, self.mutation_prob)
        self.mutation_prob = min(1, self.mutation_prob)
        if self.mutation_prob != mutation_prob:
            IndependentReproductionStrategy.LOG.warning(
                'Mutation probability out of range. Clipped to [0, 1].')

    def reproduce(self, selection_strategy: evo.SelectionStrategy,
                  population_strategy: evo.PopulationStrategy, parents,
                  offspring):
        a = selection_strategy.select_single(parents)[1]
        if self.generator.random() < self.crossover_prob:
            b = selection_strategy.select_single(parents)[1]
            a_copy = a.copy()
            b_copy = b.copy()
            try:
                children = self.crossover.crossover(a_copy, b_copy)
            except OperatorNotApplicableError as e:
                IndependentReproductionStrategy.LOG.warn(
                    'Crossover was not applicable - children unchanged.',
                    exc_info=True)
                children = [a, b]
        else:
            children = [a]

        while (children and len(offspring) <
               population_strategy.get_offspring_number()):
            o = children.pop()
            if self.generator.random() < self.mutation_prob:
                o_copy = o.copy()
                try:
                    o = self.mutation.mutate(o_copy)
                except OperatorNotApplicableError as e:
                    IndependentReproductionStrategy.LOG.warn(
                        'Mutation was not applicable - individual unchanged.',
                        exc_info=True)
            offspring.append(o)


class ChoiceReproductionStrategy(GpReproductionStrategy):
    """This class represents a reproduction strategy where the individual is
    either copied or crossed over or mutated.

    That means that an individual is subject to crossover with a probability
    **px**\ , or it is subject to mutation with a probability **pm** or it is
    just copied to the next generation with probability **pc** while it holds
    that **px + pm + pc = 1**\ . Summed up, the things that can happen to an
    individual are these:

        * individual is just copied - no crossover, no mutation
        * individual is just mutated - no crossover
        * individual is just crossed over - no mutation, the individual is one
          of two offspring that are product of combining two parents
    """

    LOG = logging.getLogger(__name__ + '.ChoiceReproductionStrategy')

    def __init__(self, functions, terminals, generator=None,
                 crossover: CrossoverOperator=None,
                 mutation: MutationOperator=None,
                 limits=None, crossover_prob=0.8, mutation_prob=0.1,
                 crossover_both: bool=True):
        """The probabilities of crossover and mutation must sum up to a number
        from the interval [0, 1] and none of the two can be a negative number
        (if it is it will be set to 0). The "rest" that is left (i.e. 1 -
        crossover probability - mutation probability) is the probability of the
        individual just being copied.

        If the probabilities of crossover and mutation sum up to a number
        greater than 1, they are scaled so that they sum up to 1 (and therefore
        every individual will be either crossed over or mutated and never just
        copied).

        :param crossover_prob: probability of performing a crossover; if it does
            not fit into interval [0, 1] it is clipped to fit; default value is
            0.8
        :param mutation_prob: probability of performing a mutation; if it does
            not fit into interval [0, 1] it is clipped to fit; default value is
            0.1
        :param crossover_both: indicates whether both children of a crossover
            event are to be passed to the next generation (True) or just the one
            corresponding to the first selected parent (False)

        .. seealso: :class:`GpReproductionStrategy`
        """
        super().__init__(functions, terminals, generator, crossover, mutation,
                         limits)
        self.crossover_prob = crossover_prob
        if self.crossover_prob < 0:
            self.crossover_prob = 0
            ChoiceReproductionStrategy.LOG.warning(
                'Crossover probability is negative. Set to 0.')

        self.mutation_prob = mutation_prob
        if self.mutation_prob < 0:
            self.mutation_prob = 0
            ChoiceReproductionStrategy.LOG.warning(
                'Mutation probability is negative. Set to 0.')

        self.mutation_prob += self.crossover_prob

        if self.mutation_prob > 1:
            self.crossover_prob /= self.mutation_prob
            self.mutation_prob = 1
            ChoiceReproductionStrategy.LOG.warning(
                'Crossover and mutation probability sum to a number higher than'
                ' 1. Scaled to %f and %f.',
                self.crossover_prob, self.mutation_prob - self.crossover_prob)

        self.crossover_both = crossover_both

    def reproduce(self, selection_strategy: evo.SelectionStrategy,
                  population_strategy: evo.PopulationStrategy, parents,
                  offspring):
        a = selection_strategy.select_single(parents)[1]
        r = self.generator.random()
        if r < self.crossover_prob:
            b = selection_strategy.select_single(parents)[1]
            a_copy = a.copy()
            b_copy = b.copy()
            try:
                children = self.crossover.crossover(a_copy, b_copy)
            except OperatorNotApplicableError as e:
                ChoiceReproductionStrategy.LOG.warn(
                    'Crossover was not applicable - children unchanged.',
                    exc_info=True)
                children = [a, b]
            if not self.crossover_both:
                children = [children[0]]
        elif r < self.mutation_prob:
            a_copy = a.copy()
            try:
                children = [self.mutation.mutate(a_copy)]
            except OperatorNotApplicableError as e:
                ChoiceReproductionStrategy.LOG.warn(
                    'Mutation was not applicable - individual unchanged.',
                    exc_info=True)
                children = [a]
        else:
            children = [a]

        while (children and len(offspring) <
               population_strategy.get_offspring_number()):
            o = children.pop()
            offspring.append(o)


# noinspection PyAbstractClass
class Gp(multiprocessing.context.Process):
    """This class forms the whole GE algorithm.
    """

    LOG = logging.getLogger(__name__ + '.Gp')

    class CallbackSituation(enum.Enum):
        iteration_start = 1
        iteration_end = 2
        end = 3
        start = 4

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

    def __init__(self,
                 fitness: evo.Fitness,
                 pop_strategy: evo.PopulationStrategy,
                 selection_strategy: evo.SelectionStrategy,
                 reproduction_strategy: evo.ReproductionStrategy,
                 population_initializer: evo.PopulationInitializer,
                 functions, terminals, stop, name=None, **kwargs):
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
            :class:`evo.PopulationInitializer`
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
        :keyword limits: specifies the size limits for the individuals, for
            details see :ref:`Limits <evo.gp.Gp.limits>`

            The default is no limits.
        :keyword evo.support.Stats stats: stats saving class
        :keyword callback: a callable which will be called during the
            evolution. It must take two arguments, first one is the algorithm
            instance itself (i.e. instance of this class), and the second is a
            :class:`.CallbackSituation` value specifying where is the
            callback called.

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
        self.reproduction_strategy = reproduction_strategy
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
            if self.callback is not None and not callable(self.callback):
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
        # noinspection PyBroadException
        try:
            self.start_time = time.time()
            self.population = self.population_initializer.initialize(
                self.pop_strategy.get_parents_number(), self.limits)

            if self.callback is not None:
                self.callback(self, Gp.CallbackSituation.start)
            self._run()
            if self.callback is not None:
                self.callback(self, Gp.CallbackSituation.end)
        except:
            Gp.LOG.warning('Evolution terminated by exception.', exc_info=True)
        finally:
            self.end_time = time.time()
            if self.fitness.get_bsf() is None:
                Gp.LOG.info('Finished. No BSF acquired.')
            else:
                Gp.LOG.info('Finished.')
                Gp.LOG.info('BSF: %s', str(self.fitness.get_bsf()))
                Gp.LOG.info('BSF fitness: %s',
                            self.fitness.get_bsf().get_fitness())
                Gp.LOG.info('BSF data: %s',
                            pprint.pformat(self.fitness.get_bsf().get_data()))
            Gp.LOG.debug('Performing garbage collection.')
            Gp.LOG.debug('Collected %d objects.', gc.collect())
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
                self.callback(self, Gp.CallbackSituation.iteration_start)

            elites = self.top_individuals(self.pop_strategy.get_elites_number())

            Gp.LOG.debug('Processing selection.')
            offspring = []
            while len(offspring) < self.pop_strategy.get_offspring_number():
                self.reproduction_strategy.reproduce(self.selection_strategy,
                                                     self.pop_strategy,
                                                     self.population,
                                                     offspring)
            self.population = self.pop_strategy.combine_populations(
                self.population, offspring, elites)
            Gp.LOG.info('Iteration %d / %.1f s. BSF %s | '
                        '%s | %s',
                        self.iterations, self.get_runtime(),
                        self.fitness.get_bsf().get_fitness(),
                        str(self.fitness.get_bsf()),
                        self.fitness.get_bsf().get_data())
            if self.callback is not None:
                self.callback(self, Gp.CallbackSituation.iteration_end)
            self.iterations += 1
        Gp.LOG.info('Finished evolution.')

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

    def get_runtime(self):
        return time.time() - self.start_time
