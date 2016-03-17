# -*- coding: utf8 -*-

import unittest

import evo.utils.tree as tree


class TestTree(unittest.TestCase):
    class Predicate:
        def __init__(self):
            self.include = False

        def __call__(self, _):
            self.include = not self.include
            return self.include

    def setUp(self):
        #     0
        # +---+----+
        # 00  01   02
        #     |    |
        # +---+    +
        # 010 011  020
        #     |    |
        #     +    +------+-----+
        #     0110 0200   0201  0202

        # root
        n0 = tree.TreeNode(None, None, [], '0')

        # 1st level
        n00 = tree.TreeNode(n0, 0, None, '00')  # leaf
        n0.children.append(n00)
        n01 = tree.TreeNode(n0, 1, [], '01')
        n0.children.append(n01)
        n02 = tree.TreeNode(n0, 2, [], '02')
        n0.children.append(n02)

        # 2nd level
        n010 = tree.TreeNode(n01, 0, None, '010')  # leaf
        n01.children.append(n010)
        n011 = tree.TreeNode(n01, 1, [], '011')
        n01.children.append(n011)
        n020 = tree.TreeNode(n02, 0, [], '020')
        n02.children.append(n020)

        # 3rd level
        n0110 = tree.TreeNode(n011, 0, None, '0110')  # leaf
        n011.children.append(n0110)
        n0200 = tree.TreeNode(n020, 0, None, '0200')  # leaf
        n020.children.append(n0200)
        n0201 = tree.TreeNode(n020, 1, None, '0201')  # leaf
        n020.children.append(n0201)
        n0202 = tree.TreeNode(n020, 2, None, '0202')  # leaf
        n020.children.append(n0202)

        self.test_tree = n0

    def test_get_nodes_dfs(self):
        with self.subTest('from_start=True, predicate=None'):
            nodes = self.test_tree.get_nodes_dfs(from_start=True,
                                                 predicate=None)
            nodes_data = list(map(lambda node: node.data, nodes))
            self.assertEqual(['0', '00', '01', '010', '011', '0110', '02',
                              '020', '0200', '0201', '0202'], nodes_data)
        with self.subTest('from_start=False, predicate=None'):
            nodes = self.test_tree.get_nodes_dfs(from_start=False,
                                                 predicate=None)
            nodes_data = list(map(lambda node: node.data, nodes))
            self.assertEqual(['0', '02', '020', '0202', '0201', '0200', '01',
                              '011', '0110', '010', '00'], nodes_data)
        with self.subTest('from_start=True, predicate=<1st, 3rd, 5th...>'):
            nodes = self.test_tree.get_nodes_dfs(from_start=True,
                                                 predicate=TestTree.Predicate())
            nodes_data = list(map(lambda node: node.data, nodes))
            self.assertEqual(['0', '01', '011', '02', '0200', '0202'],
                             nodes_data)
        with self.subTest('from_start=False, predicate=None'):
            nodes = self.test_tree.get_nodes_dfs(from_start=False,
                                                 predicate=TestTree.Predicate())
            nodes_data = list(map(lambda node: node.data, nodes))
            self.assertEqual(['0', '020', '0201', '01', '0110', '00'],
                             nodes_data)

    def test_get_nodes_bfs(self):
        with self.subTest('from_start=True, predicate=None'):
            nodes = self.test_tree.get_nodes_bfs(from_start=True,
                                                 predicate=None)
            nodes_data = list(map(lambda node: node.data, nodes))
            self.assertEqual(['0', '00', '01', '02', '010', '011', '020',
                              '0110', '0200', '0201', '0202'], nodes_data)
        with self.subTest('from_start=False, predicate=None'):
            nodes = self.test_tree.get_nodes_bfs(from_start=False,
                                                 predicate=None)
            nodes_data = list(map(lambda node: node.data, nodes))
            self.assertEqual(['0', '02', '01', '00', '020', '011', '010',
                              '0202', '0201', '0200', '0110'], nodes_data)
        with self.subTest('from_start=True, predicate=<1st, 3rd, 5th...>'):
            nodes = self.test_tree.get_nodes_bfs(from_start=True,
                                                 predicate=TestTree.Predicate())
            nodes_data = list(map(lambda node: node.data, nodes))
            self.assertEqual(['0', '01', '010', '020', '0200', '0202'],
                             nodes_data)
        with self.subTest('from_start=False, predicate=None'):
            nodes = self.test_tree.get_nodes_bfs(from_start=False,
                                                 predicate=TestTree.Predicate())
            nodes_data = list(map(lambda node: node.data, nodes))
            self.assertEqual(['0', '01', '020', '010', '0201', '0110'],
                             nodes_data)



if __name__ == '__main__':
    unittest.main()
