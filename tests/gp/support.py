# -*- coding: utf8 -*-
""" TODO docstring
"""

import unittest

import evo.utils.tree as tree
import evo.gp.support as gp_support


class TestSwapSubtrees(unittest.TestCase):

    def setUp(self):
        #   a           A
        # +-+-+       +-+-+
        # b   c       B C D
        # |             |
        # d             E

        ## abc
        # a
        a = tree.TreeNode(None, None, [], 'a')

        # b
        b = tree.TreeNode(a, 0, [], 'b')
        a.children.append(b)

        # c
        c = tree.TreeNode(a, 1, None, 'c')
        a.children.append(c)

        # d
        d = tree.TreeNode(b, 0, None, 'd')
        b.children.append(d)

        self.a = a
        self.b = b
        self.c = c
        self.d = d

        ## ABCDE
        # A
        A = tree.TreeNode(None, None, [], 'A')

        # B
        B = tree.TreeNode(A, 0, None, 'B')
        A.children.append(B)

        # C
        C = tree.TreeNode(A, 1, [], 'C')
        A.children.append(C)

        # D
        D = tree.TreeNode(A, 2, None, 'D')
        A.children.append(D)

        # E
        E = tree.TreeNode(C, 0, None, 'E')
        C.children.append(E)

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E

    def test_root_root(self):
        left = self.a
        right = self.A

        self.assertEqual((self.A, self.a),
                         gp_support.swap_subtrees(left, right))

    def test_root_nonroot(self):
        left = self.a
        right = self.C

        expected_left = self.C
        expected_right = '(A B (a (b d) c) D)'

        out = gp_support.swap_subtrees(left, right)
        self.assertEqual(2, len(out))
        self.assertEqual(expected_left, out[0])
        self.assertEqual(expected_right, out[1].__str__())

    def test_nonroot_root(self):
        left = self.b
        right = self.A

        expected_left = '(a (A B (C E) D) c)'
        expected_right = self.b

        out = gp_support.swap_subtrees(left, right)
        self.assertEqual(2, len(out))
        self.assertEqual(expected_left, out[0].__str__())
        self.assertEqual(expected_right, out[1])

    def test_nonroot_nonroot(self):
        left = self.b
        right = self.C

        expected_left = '(a (C E) c)'
        expected_right = '(A B (b d) D)'

        out = gp_support.swap_subtrees(left, right)
        self.assertEqual(2, len(out))
        self.assertEqual(expected_left, out[0].__str__())
        self.assertEqual(expected_right, out[1].__str__())
