# -*- coding: utf-8 -*-
""" TODO docstring
"""

import unittest

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
        a = gp_support.GpNode()
        a.data = 'a'

        # b
        b = gp_support.GpNode()
        b.parent = a
        b.parent_index = 0
        b.data = 'b'
        a.children = [b]

        # c
        c = gp_support.GpNode()
        c.parent = a
        c.parent_index = 1
        c.data = 'c'
        a.children.append(c)

        # d
        d = gp_support.GpNode()
        d.parent = b
        d.parent_index = 0
        d.data = 'd'
        b.children = [d]

        self.a = a
        self.b = b
        self.c = c
        self.d = d

        ## ABCDE
        # A
        A = gp_support.GpNode()
        A.data = 'A'

        # B
        B = gp_support.GpNode()
        B.parent = A
        B.parent_index = 0
        B.data = 'B'
        A.children = [B]

        # C
        C = gp_support.GpNode()
        C.parent = A
        C.parent_index = 1
        C.data = 'C'
        A.children.append(C)

        # D
        D = gp_support.GpNode()
        D.parent = A
        D.parent_index = 2
        D.data = 'D'
        A.children.append(D)

        # E
        E = gp_support.GpNode()
        E.parent = C
        E.parent_index = 0
        E.data = 'E'
        C.children = [E]

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


class TestReplaceSubtree(unittest.TestCase):

    def setUp(self):
        #   a           A
        # +-+-+       +-+-+
        # b   c       B   C
        # |
        # d

        ## abcd
        # a
        a = gp_support.GpNode()
        a.data = 'a'

        # b
        b = gp_support.GpNode()
        b.parent = a
        b.parent_index = 0
        b.data = 'b'
        a.children = [b]

        # c
        c = gp_support.GpNode()
        c.parent = a
        c.parent_index = 1
        c.data = 'c'
        a.children.append(c)

        # d
        d = gp_support.GpNode()
        d.parent = b
        d.parent_index = 0
        d.data = 'd'
        b.children = [d]

        self.a = a
        self.b = b
        self.c = c
        self.d = d

        ## ABC
        # A
        A = gp_support.GpNode()
        A.data = 'A'

        # B
        B = gp_support.GpNode()
        B.parent = A
        B.parent_index = 0
        B.data = 'B'
        A.children = [B]

        # C
        C = gp_support.GpNode()
        C.parent = A
        C.parent_index = 1
        C.data = 'C'
        A.children.append(C)

        self.A = A
        self.B = B
        self.C = C

    def test_root(self):
        old = self.a
        new = self.A

        expected = new

        self.assertEqual(expected, gp_support.replace_subtree(old, new))

    def test_nonroot(self):
        old = self.b
        new = self.A

        expected = '(a (A B C) c)'

        out = gp_support.replace_subtree(old, new)
        self.assertEqual(expected, out.__str__())
