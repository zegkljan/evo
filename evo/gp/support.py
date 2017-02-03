# -*- coding: utf-8 -*-
""" Support classes and stuff for GP.
"""

import logging
import random

import evo
import evo.utils.tree


class GpNode(evo.utils.tree.TreeNode):

    def get_arity(self=None):
        raise NotImplementedError()

    # noinspection PyMethodMayBeStatic
    def child_changed(self, child_index: int, data=None):
        """Notifies this node that one of its children changed.

        This method is to be called by children nodes on their parents to notify
        them that they changed. Override this method to implement the desired
        behaviour upon notification.

        :param child_index: index of this node's child the notification comes
            from
        :param data: optional data that will be passed to the ``notified_*``
            methods and to the propagation call (if any)
        .. seealso:: :meth:`notify_change`
        """
        pass

    def self_changed(self, data=None):
        """Notifies this node that it changed.

        This method is to be called on this node to notify it that it was
        changed. Override this method to implement the desired behaviour upon
        notification.

        :param data: optional data that will be passed to the other methods
        """
        pass

    def notify_change(self, data=None):
        """Notifies this node and the parent node that this node has changed.

        This method is to be used when the parent node needs to be notified of
        a change in this node (whatever that is). The only requirement is that
        the tree structure is properly set, i.e. the
        :attr:`evo.utils.tree.TreeNode.parent` and
        :attr:`evo.utils.tree.TreeNode.parent_index` are set correctly.

        :param data: optional data that will be passed to the ``self_changed``
            and ``child_changed`` methods
        """
        self.self_changed(data)
        if self.is_root():
            return

        self.parent.child_changed(self.parent_index, data)


class ForestIndividual(evo.Individual):
    """A class representing an individual as a forest, i.e. a set of trees.
    """

    def __init__(self, genotype: list):
        """Creates the individual.

        :param genotype: the list of genes (the roots of the trees) of the
            individual
        """
        evo.Individual.__init__(self)
        self.genotype = genotype

    def __str__(self):
        if hasattr(self, 'str'):
            return str(self.str)
        return str([str(g) for g in self.genotype])

    @property
    def genes_num(self):
        """The number of genes the individual has.
        """
        return len(self.genotype)

    def copy(self, carry_evaluation=True, carry_data=True):
        cg = [g.clone() for g in self.genotype]
        clone = ForestIndividual(cg)
        evo.Individual.copy_evaluation(self, clone, carry_evaluation)
        evo.Individual.copy_data(self, clone, carry_data)
        return clone


class RampedHalfHalfInitializer(evo.PopulationInitializer):
    """This population initializer initializes the population using the ramped
    half-and-half method: for each depth from 0 up to maximum depth, half of
    individuals will be crated using the "grow" method and the other half using
    the "full" method.

    If the number of individuals is not divisible by the number of
    initialization setups (which is double the number of depth levels - the
    "full" and "grow" for each level) then the remainder individuals will be
    initialized using randomly chosen setups (but each of them in a unique
    setup).

    .. seealso::

        Function :func:`evo.gp.support.generate_tree_full_grow`
            Performs the initialisation of the trees by the grow and full
            methods.
    """
    LOG = logging.getLogger(__name__ + '.RampedHalfHalfInitializer')

    def __init__(self, functions, terminals, max_depth: int, max_genes: int,
                 **kwargs):
        """Creates the initializer.

        The ``max_genes`` argument specifies the maximum number of genes an
        individual can have. For a classical GP (i.e. only a single tree per
        individual), set this argument to 1.

        The optional ``min_depth`` keyword argument can be used to generate
        trees from this depth instead of 1.

        The optional ``max_tries`` argument limits number of tries of
        generation a tree as well as the number of tries of generation the
        whole individual. If a generated tree is identical to one or more
        other trees (genes) in the genotype, it is thrown away and generated
        again from scratch unless this was tried ``max_tries`` times.
        Similarly, if a generated individual (i.e. a set of genes) is
        identical to another individual in the population it is thrown away
        and generated again unless this was tried ``max_tries`` times.

        :param functions: functional nodes to pick from, must have arity greater
            than zero
        :type functions: :class:`list` of :class:`evo.gp.support.GpNode`
        :param terminals: terminal nodes to pick from, must have zero arity
        :type terminals: :class:`list` of :class:`evo.gp.support.GpNode`
        :param int max_depth: maximum depth of the derivation trees; must be
            finite
        :param max_genes: the maximum number of genes (trees) that an individual
            can have
        :keyword generator: a random number generator; if ``None`` or not set
            calls to the methods of standard python module :mod:`random` will be
            performed instead
        :type generator: :class:`random.Random` or ``None``
        :keyword int min_depth: starting minimum depth of the trees; default 1
            (i.e. a single node)
        :keyword int max_tries: the maximum number of attempts to recreate a new
            individual if an identical one is already in the population; default
            is 10
        """
        super().__init__()

        self.functions = functions
        self.terminals = terminals
        self.max_depth = max_depth
        self.max_genes = max_genes

        self.generator = random
        if 'generator' in kwargs:
            self.generator = kwargs['generator']

        self.min_depth = 1
        if 'min_depth' in kwargs:
            self.min_depth = kwargs['min_depth']
            if self.min_depth > self.max_depth:
                raise ValueError('min_depth must not be greater than max_depth')

        self.max_tries = 10
        if 'max_tries' in kwargs:
            self.max_tries = kwargs['max_tries']

    def initialize(self, pop_size, limits):
        RampedHalfHalfInitializer.LOG.info('Initializing population of size '
                                           '%d', pop_size)
        max_nodes = limits['max-nodes']
        max_depth = min(self.max_depth, limits['max-depth'])
        max_genes = min(self.max_genes, limits['max-genes'])
        if self.min_depth > max_depth:
            raise ValueError('min-depth must not be greater than max-depth')
        levels_num = max_depth - self.min_depth + 1
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
        genotypes_set = set()
        for d in range(levels_num):
            max_depth = self.min_depth + d
            RampedHalfHalfInitializer.LOG.debug('Initializing %d. level; '
                                                'max. depth = %d', d, max_depth)
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
            for _ in range(individuals_per_setup + g):
                n_genes = self.generator.randrange(max_genes) + 1
                genes = self._generate_genes(max_depth, max_nodes, n_genes,
                                             False)
                tries = self.max_tries
                genes_str = ','.join([str(tree) for tree in genes])
                while genes_str in genotypes_set and tries > 0:
                    genes = self._generate_genes(max_depth, max_nodes,
                                                 n_genes, False)
                    genes_str = ','.join([str(tree) for tree in genes])
                    tries -= 1
                genotypes_set.add(genes_str)
                pop.append(ForestIndividual(genes))

            # full
            RampedHalfHalfInitializer.LOG.debug('Initializing %d individuals '
                                                'with full method.',
                                                individuals_per_setup + g)
            for _ in range(individuals_per_setup + f):
                n_genes = self.generator.randrange(max_genes) + 1
                genes = self._generate_genes(max_depth, max_nodes, n_genes,
                                             True)
                tries = self.max_tries
                genes_str = ','.join([str(tree) for tree in genes])
                while genes_str in genotypes_set and tries > 0:
                    genes = self._generate_genes(max_depth, max_nodes, n_genes,
                                                 True)
                    genes_str = ','.join([str(tree) for tree in genes])
                    tries -= 1
                genotypes_set.add(genes_str)
                pop.append(ForestIndividual(genes))
        RampedHalfHalfInitializer.LOG.info('Initialization complete.')
        return pop

    def _generate_genes(self, max_depth, max_nodes, n_genes, full):
        genes = []
        genes_strs = []
        for n in range(n_genes):
            tree, tree_str = self._generate_gene(genes_strs, max_depth,
                                                 max_nodes, full)
            genes.append(tree)
            genes_strs.append(tree_str)
        return genes

    def _generate_gene(self, genes_strs, max_depth, max_nodes, full):
        tree = generate_tree_full_grow(self.functions, self.terminals, max_depth,
                                       max_nodes, self.generator, full)
        tree_str = str(tree)
        tries = self.max_tries
        while tree_str in genes_strs and tries > 0:
            tree = generate_tree_full_grow(self.functions, self.terminals,
                                           max_depth, max_nodes, self.generator, full)
            tree_str = str(tree)
            tries -= 1
        return tree, tree_str


def generate_tree_full_grow(inners, leaves, depth, nodes, generator, full):
    """Generates a tree from the given list of inner and leaf nodes.

    This function provides both the grow method and the full method of
    generating trees. It is controlled by the argument ``full``\ . The two
    lists  must be disjunctive, i.e. one must contain only inners, other must
    contain only leaves otherwise the algorithm will not work properly.

    The ``inners`` and ``leaves`` lists must contain callables that,
    when called without arguments, create an instance of a :class:`.GpNode`\ .

    :param inners: list of node producers that can be anywhere in the tree
    :param leaves: list of node producers that will be used only when a limit
        is reached
    :param depth: maximum depth the generated tree will have; 1 is a single node
    :param nodes: maximum number of nodes the generated tree will have
    :param generator: random number generator used for choosing the nodes
    :param full: if set to ``True`` the full method will be used, otherwise
        grow will be used
    """
    level = []
    current_parent = None
    child_index = 0
    root = None
    must_create = 1
    this_level = 0
    next_level = 0
    while nodes > 0 and must_create > 0:
        if nodes == 1 or depth == 1:
            pool = leaves
        else:
            selected_functions = [f for f in inners
                                  if f().get_arity() < nodes - must_create + 1]
            if full:
                pool = selected_functions
                if not pool:
                    pool = leaves
            else:
                pool = selected_functions + leaves

        node_creator = generator.choice(pool)
        node = node_creator()
        arity = node.get_arity()
        must_create = must_create - 1 + arity

        if arity == 0:
            node.children = None
        else:
            node.children = [None] * arity

        if current_parent is None:
            current_parent = node
            root = node
            this_level = arity
            next_level = 0
            depth -= 1
            if arity == 0:
                break
        else:
            this_level -= 1
            next_level += arity
            if this_level == 0:
                this_level = next_level
                next_level = 0
                depth -= 1
            current_parent.children[child_index] = node
            node.parent = current_parent
            node.parent_index = child_index
            child_index += 1
            if arity > 0:
                level.append(node)

            if child_index == current_parent.get_arity():
                child_index = 0
                if level:
                    current_parent = level.pop(0)
                while level and current_parent.get_arity() == 0:
                    current_parent = level.pop(0)
        nodes -= 1

    return root


def swap_subtrees(s1: GpNode, s2: GpNode):
    """Takes two (sub)trees and swaps them in place, returning the new roots.

    Example:
    Suppose that we have two trees::

           a             i
        +--+-+       +---+---+
        b    c       j   k   l
           +-+-+     |       +--+
           d   e     m       n  o
        +--+   |             |
        f  g   h             p

    Suppose that ``s1`` corresponds to the node ``c`` in the left tree and
    ``s2`` corresponds to the node ``m`` in the right tree. After the swap the
    trees are going to look like::

           a             i
        +--+-+       +---+---+
        b    m       j   k   l
                     |       +--+
                     c       n  o
                   +-+-+     |
                   d   e     p
                +--+   |
                f  g   h

    and the function will return a tuple ``(a, i)``.

    Now suppose that ``s1`` corresponds to the ``a`` node of the original left
    tree and ``s2`` corresponds to the ``n`` node of the original right tree.
    After the swap the trees are going to look like::

           n             i
           |         +---+---+
           p         j   k   l
                     |       +--+
                     m       a  o
                             |
                          +--+-+
                          b    c
                             +-+-+
                             d   e
                          +--+   |
                          f  g   h

    and the function will return a tuple ``(n, i)``.

    This function also notifies about the change after the swapping is
    completed.

    :param s1: the first (sub)tree to be swapped
    :param s2: the second (sub)tree to be swapped
    :return: a tuple of root nodes of the corresponding trees
    :rtype: :class:`evo.utils.tree.GpNode`
    """
    s1p = s1.parent
    s1pi = s1.parent_index
    s2p = s2.parent
    s2pi = s2.parent_index

    if s1.is_root():
        if s2.is_root():
            ret = s2, s1
        else:
            s2.parent = None
            s2.parent_index = None
            s2p.children[s2pi] = s1
            s1.parent = s2p
            s1.parent_index = s2pi
            ret = s2, s2p.get_root()
    else:
        if s2.is_root():
            s1.parent = None
            s1.parent_index = None
            s1p.children[s1pi] = s2
            s2.parent = s1p
            s2.parent_index = s1pi
            ret = s1p.get_root(), s1
        else:
            s1p.children[s1pi] = s2
            s2p.children[s2pi] = s1
            s1.parent = s2p
            s1.parent_index = s2pi
            s2.parent = s1p
            s2.parent_index = s1pi
            ret = s1p.get_root(), s2p.get_root()

    s1.notify_change()
    s2.notify_change()
    return ret


def replace_subtree(old: GpNode, new: GpNode):
    """Replaces the ``old`` (sub)tree with the ``new`` one, handling correct
    connection with the possible parent of the ``old`` tree.

    Example:
    Suppose that we have two trees::

           a             i
        +--+-+       +---+---+
        b    c       j   k   l
           +-+-+
           d   e
        +--+   |
        f  g   h

    Suppose that ``old`` corresponds to the node ``c`` in the left tree and
    ``new`` corresponds to the right tree (its root, i.e. node ``i``). After the
    replace the left tree is going to look like::

           a
        +--+-+
        b    i
         +---+---+
         j   k   l

    and the function will return the root of the tree ``old`` was in, i.e. the
    node ``a`` in this case

    This function also notifies (from the ``new`` node) about the change after
    the replace is completed.

    :param old: the (sub)tree to be replaced
    :param new: the (sub)tree to be inserted in the ``old``'s place
    :return: root of the tree that was operated on
    :rtype: :class:`evo.utils.tree.GpNode`
    """
    old_parent = old.parent
    old_parent_index = old.parent_index

    if old.is_root():
        return new
    else:
        old_parent.children[old_parent_index] = new
        new.parent = old_parent
        new.parent_index = old_parent_index
        new.notify_change()
        return new.get_root()
