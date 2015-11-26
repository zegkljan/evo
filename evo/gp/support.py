# -*- coding: utf8 -*-
""" Support classes and stuff for GP.
"""

import logging

import random

import evo
import evo.utils.tree

__author__ = 'Jan Žegklitz'


class GpNode(evo.utils.tree.TreeNode):
    def get_arity(self):
        raise NotImplementedError()

    # noinspection PyMethodMayBeStatic
    def notify_child_changed(self, child_index: int, data=None):
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

    def notify_change(self, data=None):
        """Notifies the parent node that this node has changed.

        This method is to be used when the parent node needs to be notified of
        a change in this node (whatever that is). The only requirement is that
        the tree structure is properly set, i.e. the
        :attr:`evo.utils.tree.TreeNode.parent` and
        :attr:`evo.utils.tree.TreeNode.parent_index` are set correctly.

        :param data: optional data that will be passed to the parent
        """
        if self.is_root():
            return

        self.parent.notify_child_changed(self.parent_index, data)


class TreeIndividual(evo.Individual):
    """A class representing an individual as a tree.
    """

    def __init__(self, genotype):
        """Creates the individual.

        :param evo.gp.support.GpNode genotype: the genotype of the individual,
            i.e. the tree root
        """
        evo.Individual.__init__(self)
        self.genotype = genotype

    def __str__(self):
        if hasattr(self, 'str'):
            return str(self.str)
        return str(self.genotype)

    def copy(self, carry_evaluation=True, carry_data=True):
        clone = TreeIndividual(self.genotype.clone())
        evo.Individual.copy_evaluation(self, clone, carry_evaluation)
        evo.Individual.copy_data(self, clone, carry_data)
        return clone


def generate_full_grow(inners, leaves, depth, generator):
    if depth <= 0:
        node_gen = generator.choice(leaves)
        node = node_gen()
        node.children = None
        return node

    root_gen = generator.choice(inners)
    root = root_gen()
    arity = root.get_arity()
    if arity == 0:
        root.children = None
        return root
    root.children = [None] * arity
    for i in range(arity):
        child = generate_full_grow(inners, leaves, depth - 1, generator)
        root.children[i] = child
        child.parent = root
        child.parent_index = i
    return root


def generate_grow(functions, terminals, depth, generator):
    return generate_full_grow(functions + terminals, terminals, depth,
                              generator)


def generate_full(functions, terminals, depth, generator):
    return generate_full_grow(functions, terminals, depth, generator)


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

        Function :func:`evo.gp.support.generate_grow`
            Performs the initialisation by the "grow" method.
        Function :func:`evo.gp.support.generate_full`
            Performs the initialisation by the "full" method.
    """
    LOG = logging.getLogger(__name__ + '.RampedHalfHalfInitializer')

    def __init__(self, functions, terminals, max_depth, **kwargs):
        """Creates the initializer.

        The optional ``min_depth`` keyword argument can be used to generate
        trees from this depth instead of 0.

        :param functions: functional nodes to pick from, must have arity greater
            than zero
        :type functions: :class:`list` of :class:`evo.gp.support.GpNode`
        :param terminals: terminal nodes to pick from, must have zero arity
        :type terminals: :class:`list` of :class:`evo.gp.support.GpNode`
        :param int max_depth: maximum depth of the derivation trees; must be
            finite
        :keyword generator: a random number generator; if ``None`` or not set
            calls to the methods of standard python module :mod:`random` will be
            performed instead
        :type generator: :class:`random.Random` or ``None``
        :keyword int min_depth: starting minimum depth of the trees; default 0
            (i.e. a single node)
        :keyword int max_tries: the maximum number of attempts to recreate a new
            individual if an identical one is already in the population; default
            is 100
        """
        super().__init__()

        self.functions = functions
        self.terminals = terminals
        self.max_depth = max_depth

        self.generator = random
        if 'generator' in kwargs:
            self.generator = kwargs['generator']

        self.min_depth = 0
        if 'min_depth' in kwargs:
            self.min_depth = kwargs['min_depth']
            if self.min_depth > self.max_depth:
                raise ValueError('min_depth must not be greater than max_depth')

        self.max_tries = 100
        if 'max_tries' in kwargs:
            self.max_tries = kwargs['max_tries']

    def initialize(self, pop_size):
        RampedHalfHalfInitializer.LOG.info('Initializing population of size '
                                           '%d', pop_size)
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
        trees_set = set()
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
                tree = generate_grow(self.functions, self.terminals, max_depth,
                                     self.generator)
                tries = self.max_tries
                tree_str = str(tree)
                while tree_str in trees_set and tries >= 0:
                    tree = generate_grow(self.functions, self.terminals,
                                         max_depth, self.generator)
                    tree_str = str(tree)
                    tries -= 1
                trees_set.add(tree_str)
                pop.append(TreeIndividual(tree))

            # full
            RampedHalfHalfInitializer.LOG.debug('Initializing %d individuals '
                                                'with full method.',
                                                individuals_per_setup + g)
            for _ in range(individuals_per_setup + f):
                tree = generate_full(self.functions, self.terminals, max_depth,
                                     self.generator)
                tries = self.max_tries
                tree_str = str(tree)
                while tree_str in trees_set and tries >= 0:
                    tree = generate_full(self.functions, self.terminals,
                                         max_depth, self.generator)
                    tree_str = str(tree)
                    tries -= 1
                trees_set.add(tree_str)
                pop.append(TreeIndividual(tree))
        RampedHalfHalfInitializer.LOG.info('Initialization complete.')
        return pop


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
