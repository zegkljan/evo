# -*- coding: utf8 -*-
"""This module contains common mathematical functions and operators as
subclasses of :class:`evo.gp.support.GpNode`.
"""

import evo.gp.support


# noinspection PyAbstractClass
class MathNode(evo.gp.support.GpNode):
    """Base class for all mathematical nodes defined in
    :mod:`evo.gp.math_nodes`.
    """
    def eval(self):
        raise NotImplementedError()

    def get_variables(self):
        """Returns are nodes of class :class:`evo.gp.math_nodes.Variable` below
        and including this node.
        """
        variables = []

        def var_extractor(n):
            if isinstance(n, Variable):
                variables.append(n)
        self.preorder(var_extractor)
        return variables

    def set_variables(self, **kwargs):
        """Sets the values to the specified variables.

        The values are passed via *kwargs*::

            set_variables(x=0, y=1) # sets 0 to the variable 'x', 1 to the
                                    # variable 'y'
            set_variables(**{'var with space': 0}) # sets 0 to the variable
                                                   # 'var with space'
        """
        for var in self.get_variables():
            var.set_value(kwargs[var.data])


class Add2(MathNode):
    """Addition of two operands: ``a + b``
    """
    def __init__(self):
        super().__init__()
        self.data = '+'

    def get_arity(self):
        return 2

    def eval(self):
        return self.children[0].eval() + self.children[1].eval()


class Sub2(MathNode):
    """Subtraction of the second operand from the first one: ``a - b``
    """
    def __init__(self):
        super().__init__()
        self.data = '-'

    def get_arity(self):
        return 2

    def eval(self):
        return self.children[0].eval() - self.children[1].eval()


class Mul2(MathNode):
    """Multiplication of two operands: ``a * b``
    """
    def __init__(self):
        super().__init__()
        self.data = '*'

    def get_arity(self):
        return 2

    def eval(self):
        left = self.children[0].eval()
        right = self.children[1].eval()
        return left * right


class Div2(MathNode):
    """Division of the first operand by the second one: ``a / b``

    .. warning::

        This is an unprotected division, i.e. division by zero is not handled in
        this node.
    """
    def __init__(self):
        super().__init__()
        self.data = '/'

    def get_arity(self):
        return 2

    def eval(self):
        return self.children[0].eval() / self.children[1].eval()


class IDiv2(MathNode):
    """Integer division of the first operand by the second one: ``a // b``

    .. warning::

        This is an unprotected division, i.e. division by zero is not handled in
        this node.
    """
    def __init__(self):
        super().__init__()
        self.data = '//'

    def get_arity(self):
        return 2

    def eval(self):
        return self.children[0].eval() // self.children[1].eval()


class PDiv2(MathNode):
    """Protected division of the first operand by the second one, returns 1 if
    the second operand is zero: ``1 if b == 0 else a / b``
    """
    def __init__(self):
        super().__init__()
        self.data = '%'

    def get_arity(self):
        return 2

    def eval(self):
        b = self.children[1].eval()
        if b == 0:
            return 1
        return self.children[0].eval() / b


class PIDiv2(MathNode):
    """Protected integer division of the first operand by the second one,
    returns 1 if the second operand is zero: ``1 if b == 0 else a // b``
    """
    def __init__(self):
        super().__init__()
        self.data = '%%'

    def get_arity(self):
        return 2

    def eval(self):
        b = self.children[1].eval()
        if b == 0:
            return 1
        return self.children[0].eval() // b


class Const(MathNode):
    """A constant.
    """

    def __init__(self, val=None):
        super().__init__()
        self.data = val

    def get_arity(self):
        return 0

    def eval(self):
        return self.data


class Variable(MathNode):
    """A variable.
    """

    def __init__(self, name=None):
        super().__init__()
        self.data = name
        self.val = None

    def set_value(self, val):
        self.val = val

    def get_arity(self):
        return 0

    def eval(self):
        return self.val
