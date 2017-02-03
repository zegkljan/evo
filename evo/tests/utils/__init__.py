# -*- coding: utf-8 -*-
""" TODO docstring
"""

import unittest

import evo.utils


class SelectTest(unittest.TestCase):

    def setUp(self):
        self.sequence = [0, 2, 4, 1, 3, 5]

    def test_select(self):
        self.assertEqual(0, evo.utils.select(self.sequence.copy(), 0))
        self.assertEqual(1, evo.utils.select(self.sequence.copy(), 1))
        self.assertEqual(2, evo.utils.select(self.sequence.copy(), 2))
        self.assertEqual(3, evo.utils.select(self.sequence.copy(), 3))
        self.assertEqual(4, evo.utils.select(self.sequence.copy(), 4))
        self.assertEqual(5, evo.utils.select(self.sequence.copy(), 5))
        self.assertRaises(ValueError, evo.utils.select, self.sequence.copy(), 6)

        self.assertEqual(5, evo.utils.select(self.sequence.copy(), 0,
                                             cmp=lambda a, b: a > b))
        self.assertEqual(4, evo.utils.select(self.sequence.copy(), 1,
                                             cmp=lambda a, b: a > b))
        self.assertEqual(3, evo.utils.select(self.sequence.copy(), 2,
                                             cmp=lambda a, b: a > b))
        self.assertEqual(2, evo.utils.select(self.sequence.copy(), 3,
                                             cmp=lambda a, b: a > b))
        self.assertEqual(1, evo.utils.select(self.sequence.copy(), 4,
                                             cmp=lambda a, b: a > b))
        self.assertEqual(0, evo.utils.select(self.sequence.copy(), 5,
                                             cmp=lambda a, b: a > b))

        self.assertEqual(1, evo.utils.select(self.sequence.copy(), 0, left=2))
        self.assertEqual(3, evo.utils.select(self.sequence.copy(), 1, left=2))
        self.assertEqual(4, evo.utils.select(self.sequence.copy(), 2, left=2))
        self.assertEqual(5, evo.utils.select(self.sequence.copy(), 3, left=2))
        self.assertRaises(ValueError, evo.utils.select, self.sequence.copy(), 4,
                          left=2)

        self.assertEqual(0, evo.utils.select(self.sequence.copy(), 0, right=3))
        self.assertEqual(1, evo.utils.select(self.sequence.copy(), 1, right=3))
        self.assertEqual(2, evo.utils.select(self.sequence.copy(), 2, right=3))
        self.assertEqual(4, evo.utils.select(self.sequence.copy(), 3, right=3))
        self.assertRaises(ValueError, evo.utils.select, self.sequence.copy(), 4,
                          right=3)

        self.assertEqual(1, evo.utils.select(self.sequence.copy(), 0, left=1,
                                             right=4))
        self.assertEqual(2, evo.utils.select(self.sequence.copy(), 1, left=1,
                                             right=4))
        self.assertEqual(3, evo.utils.select(self.sequence.copy(), 2, left=1,
                                             right=4))
        self.assertEqual(4, evo.utils.select(self.sequence.copy(), 3, left=1,
                                             right=4))
        self.assertRaises(ValueError, evo.utils.select, self.sequence.copy(), 4,
                          left=1, right=4)
