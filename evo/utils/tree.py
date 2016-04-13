# -*- coding: utf8 -*-
"""This package contains miscellaneous utility functions and classes for working
with trees (data structure).
"""

import copy
import itertools


class TreeNode(object):
    """A node in a tree. Similar to :class:`Node` but has only one predecessor
    which is directly the parent tree node, not a list.
    """

    def __init__(self, parent=None, parent_index=None, children=None,
                 data: dict=None, **kwargs):
        self.parent = parent
        self.parent_index = parent_index
        self.children = children
        if data is None:
            self.data = dict()
        else:
            self.data = data

    def is_leaf(self):
        return self.children is None

    def is_root(self):
        return self.parent is None

    def get_root(self):
        """Returns the root of the tree this node is in.

        Climbs up from this node by parents until a node without parent is found
        and returned.
        """
        if self.parent is None:
            return self
        return self.parent.get_root()

    def is_complete(self):
        return self.children is not None and self.children

    def get_subtree_size(self):
        """Returns the total number of nodes (including leaves) in the tree
        at this node (including this node).
        """
        if self.is_leaf():
            return 1
        return sum([n.get_subtree_size() for n in self.children]) + 1

    def get_filtered_subtree_size(self, predicate):
        """Returns the total number of nodes (including leaves) that are under
        this node (including this node) and for which ``predicate(node)``
        returns ``True``.
        """
        if self.is_leaf():
            return predicate(self)
        return (sum([n.get_filtered_subtree_size(predicate)
                     for n in self.children]) +
                predicate(self))

    def get_subtree_depth(self):
        """Returns the maximum depth of the tree at this node (including this
        node).
        """
        if self.is_leaf():
            return 1
        return max([n.get_subtree_depth() for n in self.children]) + 1

    def get_depth(self):
        """Returns the depth of this node in the tree it resides in.

        If the node is the root node of the tree, 1 is returned.
        """
        node = self
        n = 1
        while node.parent is not None:
            node = node.parent
            n += 1
        return n

    def get_nodes_dfs(self, from_start=True, predicate=None):
        """Returns a list of all nodes below this node (incl.) in depth-first
        order.

        If the optional ``from_start`` argument is ``True`` (default) then the
        children of nodes are walked through in the natural order, if it is
        ``False`` then they are walked through in the reversed order.

        The optional ``predicate`` argument can be used to filter out the nodes
        for which calling ``predicate(node)`` (``node`` is the tested node)
        yields ``False``. If it is ``None`` (the default) then no filtering
        is done, otherwise it must be a callable object.

        Example:
        Suppose that the tree looks like::

               1
            +--+-+
            2    3
               +-+-+
               4   5
            +--+   |
            6  7   8

        If ``from_start`` is ``True`` then the result is
        ``[1, 2, 3, 4, 6, 7, 5, 8]``.
        If ``from_start`` is ``False`` then the result is
        ``[1, 3, 5, 8, 4, 7, 6, 2]``.
        If ``from_start`` is ``True`` and
        ``predicate = lambda node: node.is_leaf()`` (i.e. it is supposed to keep
        only the leaf nodes) then the result is ``[2, 6, 7, 8]``.
        """
        o = [self]
        ret = []
        while o:
            node = o.pop()
            if predicate is None or predicate(node):
                ret.append(node)
            if node.children is not None:
                if from_start:
                    o.extend(reversed(node.children))
                else:
                    o.extend(node.children)
        return ret

    def get_nodes_bfs(self, from_start=True, predicate=None,
                      compute_depths=False):
        """Returns a list of all nodes below this node (incl.) in breadth-first
        order.

        If the optional ``from_start`` argument is ``True`` (default) then the
        children of nodes are walked through in the natural order, if it is
        ``False`` then they are walked through in the reversed order.

        The optional ``predicate`` argument can be used to filter out the nodes
        for which calling ``predicate(node)`` (``node`` is the tested node)
        yields ``False``. If it is ``None`` (the default) then no filtering
        is done, otherwise it must be a callable object.

        If the optional ``compute_depths`` argument is ``True`` (default is
        ``False``\ ) the output list is a list of 2-tuples with the node in
        the first position and the depth the node is in in the second
        position. The depths are computed "on the fly" during assembly of
        the list and therefore is more efficient than calling the
        :meth:`.get_depth` method on each of the nodes.

        Example:
        Suppose that the tree looks like::

               1
            +--+-+
            2    3
               +-+-+
               4   5
            +--+   |
            6  7   8

        If ``from_start`` is ``True`` then the result is
        ``[1, 2, 3, 4, 5, 6, 7, 8]``.
        If ``from_start`` is ``False`` then the result is
        ``[1, 3, 2, 5, 4, 8, 7, 6]``.
        If ``from_start`` is ``True`` and
        ``predicate = lambda node: node.is_leaf()`` (i.e. it is supposed to keep
        only the leaf nodes) then the result is ``[2, 6, 7, 8]``.
        """
        o = [(self, 1)]
        ret_nodes = []
        ret_depths = []
        while o:
            node, d = o.pop(0)
            if predicate is None or predicate(node):
                ret_nodes.append(node)
                ret_depths.append(d)
            if node.children is not None:
                if from_start:
                    o.extend(zip(node.children, itertools.repeat(d + 1)))
                else:
                    o.extend(zip(reversed(node.children),
                                 itertools.repeat(d + 1)))
        if compute_depths:
            return list(zip(ret_nodes, ret_depths))
        return ret_nodes

    def preorder(self, fn):
        """Goes through the tree in pre-order and calls ``fn(node)`` on each
        node.
        """
        fn(self)
        if self.is_leaf():
            return
        for child in self.children:
            child.preorder(fn)

    def postorder(self, fn):
        """Goes through the tree in post-order and calls ``fn(node)`` on each
        node.
        """
        if self.is_leaf():
            fn(self)
            return
        for child in self.children:
            child.postorder(fn)
        fn(self)

    def clone(self):
        """Clones the tree as if this node was its root (i.e. if this node
        had a parent, it will be set to None for the cloned node).
        """
        n = self.clone_self()
        if self.is_leaf():
            return n
        children = [None] * len(self.children)
        for individual in range(len(self.children)):
            c = self.children[individual].clone()
            c.parent = n
            c.parent_index = self.children[individual].parent_index
            children[individual] = c
        n.children = children
        return n

    def clone_self(self):
        """Clones this node but not its children.

        Override this method for special cloning.
        """
        c = type(self)()
        self.copy_contents(c)
        return c

    def copy_contents(self, dest):
        dest.data = copy.deepcopy(self.data)

    def is_shape_equal(self, other):
        if self.children is None:
            return other.children is None

        if other.children is None:
            return False

        if len(self.children) != len(other.children):
            return False

        for s, o in zip(self.children, other.children):
            if not s.is_shape_equal(o):
                return False
        return True

    def __str__(self):
        if self.is_leaf():
            return self.self_str()

        children_str = ' '.join([x.__str__() for x in self.children])
        if len(self.children) > 0:
            children_str = ' ' + children_str
        name = type(self).__name__
        if 'name' in self.data:
            name = self.data['name']
        return '({0}{1})'.format(name, children_str)

    def self_str(self):
        """Returns the string representation of the node without the children
        (i.e. only the data).
        """
        if 'name' in self.data:
            return '{0}'.format(self.data['name'])
        return '{0}'.format(type(self).__name__)
