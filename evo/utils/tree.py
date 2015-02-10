# -*- coding: utf8 -*-
"""This package contains miscellaneous utility functions and classes for working
with trees (data structure).
"""

__author__ = 'Jan Žegklitz'


class TreeNode(object):
    """A node in a tree. Similar to :class:`Node` but has only one predecessor
    which is directly the parent tree node, not a list.
    """

    def __init__(self, parent, parent_index, children, data):
        self.parent = parent
        self.parent_index = parent_index
        self.children = children
        self.data = data

    def is_leaf(self):
        return self.children is None

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

    def get_nth_node(self, n):
        """Iterates through the tree in breadth-first fashion and returns the
        ``n``-th node encountered. This node is included (i.e. it's the 0th
        node).
        """
        o = [self]
        while o:
            node = o.pop(0)
            if n == 0:
                return node
            n -= 1
            if not node.is_leaf():
                o.extend(node.children)
        return None

    def get_filtered_nth_node(self, n, predicate):
        """Iterates through the tree in breadth-first fashion and returns the
        ``n``-th node encountered for which ``predicate(node)`` is ``True``.
        This node is included (i.e. it's the 0th node if it comes ``True``
        from the predicate).
        """
        o = [self]
        while o:
            node = o.pop(0)
            if predicate(node):
                if n == 0:
                    return node
                n -= 1
            if not node.is_leaf():
                o.extend(node.children)
        return None

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
        if self.is_leaf():
            return TreeNode(None, None, None, self.data)
        children = [None] * len(self.children)
        n = TreeNode(None, None, None, self.data)
        for individual in range(len(self.children)):
            c = self.children[individual].clone()
            c.parent = n
            c.parent_index = self.children[individual].parent_index
            children[individual] = c
        n.children = children
        return n

    def __str__(self):
        if self.is_leaf():
            return self.self_str()

        children_str = ' '.join([x.__str__() for x in self.children])
        if len(self.children) > 0:
            children_str = ' ' + children_str
        return '({0}{1})'.format(self.data, children_str)

    def self_str(self):
        """Returns the string representation of the node without the children
        (i.e. only the data).
        """
        return '{0}'.format(self.data)