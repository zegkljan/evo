"""This package contains tools and algorithms for symbolic regression.

The package itself contains common mathematical functions and operators as
subclasses of :class:`evo.gp.support.GpNode` for use in symbolic regression
tasks.
"""

import numpy

import evo.gp.support


# noinspection PyAbstractClass
class MathNode(evo.gp.support.GpNode):
    """Base class for all mathematical nodes defined in
    :mod:`evo.gp.math_nodes`.
    """

    def __init__(self, cache=True, **kwargs):
        super().__init__(**kwargs)
        self.cache = cache
        self._cache = None

    def eval(self, args: dict=None):
        """Evaluates this node.

        If :attr:`.cache` is set to ``True`` and there is a cached value it is
        returned without the actual evaluation, otherwise the value is computed,
        stored to the cache and returned.

        .. note::

            You shouldn't override this method. Override :meth:`.operation`
            instead.

        :param args: values to set to variables, keyed by variable names
        :return: result of the evaluation
        """
        if self.cache and self._cache is not None:
            return self._cache

        result = self.operation(*[self.eval_child(i, args)
                                  for i
                                  in range(len(self.children))])
        if self.cache:
            self._cache = result

        return result

    def eval_child(self, child_no: int, args: dict=None):
        """Evaluates the child specified by its number and returns the result.

        The children are counted from 0. The ``args`` argument has the same
        meaning as in :meth:`.eval`\ .

        Override this method to implement default scheme of children evaluation.

        :param child_no: zero-based index of the children
        :param args: values to set to variables, keyed by variable names

        .. seealso: :meth:`.eval`
        """
        return self.children[child_no].eval(args)

    def operation(self, *args):
        """Evaluation function of the node.

        Override this method to implement specific nodes. This method is static
        and therefore performs only the computation of the given node, it does
        not cause the evaluation of the whole tree.

        To evaluate the whole tree, use :meth:`.eval` instead.

        .. seealso:: :meth:`.eval`

        :param args: arguments to the operation, their number must match the
            arity of the node
        :return: result of the evaluation
        """
        raise NotImplementedError()

    def child_changed(self, child_index: int, data=None):
        if self.cache:
            self.notify_change(data=data)

    def self_changed(self, data=None):
        if self.cache:
            self._cache = None

    def clear_cache(self, propagate: bool=True):
        """Clears the evaluation cache.

        If ``propagate`` is ``True`` (default) then this method will be
        recursively called on the children.

        :param propagate: whether the call should propagate or not
        """
        self._cache = None
        if not self.is_leaf() and propagate:
            for c in self.children:
                try:
                    c.clear_cache()
                except AttributeError:
                    pass

    def infix(self, **kwargs) -> str:
        raise NotImplementedError()


class Add2(MathNode):
    """Addition of two operands: ``a + b``
    """
    INFIX_FMT = '({0} + {1})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = '+'

    @staticmethod
    def get_arity():
        return 2

    def operation(self, *args):
        return args[0] + args[1]

    def infix(self, **kwargs):
        return Add2.INFIX_FMT.format(self.children[0].infix(**kwargs),
                                     self.children[1].infix(**kwargs))

    def clone_self(self):
        c = Add2(cache=self.cache)
        c.cache = self.cache
        c._cache = self._cache
        return c


class Sub2(MathNode):
    """Subtraction of the second operand from the first one: ``a - b``
    """
    INFIX_FMT = '({0} - {1})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = '-'

    @staticmethod
    def get_arity():
        return 2

    def operation(self, *args):
        return args[0] - args[1]

    def infix(self, **kwargs):
        return Sub2.INFIX_FMT.format(self.children[0].infix(**kwargs),
                                     self.children[1].infix(**kwargs))

    def clone_self(self):
        c = Sub2(cache=self.cache)
        c.cache = self.cache
        c._cache = self._cache
        return c


class Mul2(MathNode):
    """Multiplication of two operands: ``a * b``
    """
    INFIX_FMT = '({0} * {1})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = '*'

    @staticmethod
    def get_arity():
        return 2

    def operation(self, *args):
        return args[0] * args[1]

    def infix(self, **kwargs):
        return Mul2.INFIX_FMT.format(self.children[0].infix(**kwargs),
                                     self.children[1].infix(**kwargs))

    def clone_self(self):
        c = Mul2(cache=self.cache)
        c.cache = self.cache
        c._cache = self._cache
        return c


class Div2(MathNode):
    """Division of the first operand by the second one: ``a / b``

    .. warning::

        This is an unprotected division, i.e. division by zero is not handled in
        this node.
    """
    INFIX_FMT = '({0} / {1})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = '/'

    @staticmethod
    def get_arity():
        return 2

    def operation(self, *args):
        return numpy.true_divide(args[0], args[1])

    def infix(self, **kwargs):
        return Div2.INFIX_FMT.format(self.children[0].infix(**kwargs),
                                     self.children[1].infix(**kwargs))

    def clone_self(self):
        c = Add2(cache=self.cache)
        c.cache = self.cache
        c._cache = self._cache
        return c


class IDiv2(MathNode):
    """Integer division of the first operand by the second one: ``a // b``

    .. warning::

        This is an unprotected division, i.e. division by zero is not handled in
        this node.
    """
    INFIX_FMT = '({0} // {1})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = '//'

    @staticmethod
    def get_arity():
        return 2

    def operation(self, *args):
        return numpy.floor_divide(args[0], args[1])

    def infix(self, **kwargs) -> str:
        return IDiv2.INFIX_FMT.format(self.children[0].infix(**kwargs),
                                      self.children[1].infix(**kwargs))

    def clone_self(self):
        c = IDiv2(cache=self.cache)
        c.cache = self.cache
        c._cache = self._cache
        return c


class PDiv2(MathNode):
    """Protected division of the first operand by the second one, returns 1 if
    the second operand is zero: ``1 if b == 0 else a / b``
    """
    INFIX_FMT = '({0} {{/}} {1})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = '{/}'

    @staticmethod
    def get_arity():
        return 2

    def operation(self, *args):
        out = numpy.true_divide(args[0], args[1])
        if isinstance(args[1], numpy.ndarray):
            out[args[1] == 0] = 1
            return out
        if args[1] == 0:
            return 1
        return out

    def infix(self, **kwargs) -> str:
        return PDiv2.INFIX_FMT.format(self.children[0].infix(**kwargs),
                                      self.children[1].infix(**kwargs))

    def clone_self(self):
        c = PDiv2(cache=self.cache)
        c.cache = self.cache
        c._cache = self._cache
        return c


class PIDiv2(MathNode):
    """Protected integer division of the first operand by the second one,
    returns 1 if the second operand is zero: ``1 if b == 0 else a // b``
    """
    INFIX_FMT = '({0} {{//}} {1})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = '{//}'

    @staticmethod
    def get_arity():
        return 2

    def operation(self, *args):
        out = numpy.floor_divide(args[0], args[1])
        if isinstance(args[1], numpy.ndarray):
            out[args[1] == 0] = 1
            return out
        if args[1] == 0:
            return 1
        return out

    def infix(self, **kwargs) -> str:
        return PIDiv2.INFIX_FMT.format(self.children[0].infix(**kwargs),
                                       self.children[1].infix(**kwargs))

    def clone_self(self):
        c = PIDiv2(cache=self.cache)
        c.cache = self.cache
        c._cache = self._cache
        return c


class Sin(MathNode):
    """The sine function.
    """
    INFIX_FMT = 'sin({0})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = 'sin'

    @staticmethod
    def get_arity():
        return 1

    def operation(self, *args):
        return numpy.sin(args[0])

    def infix(self, **kwargs):
        return Sin.INFIX_FMT.format(self.children[0].infix(**kwargs))

    def clone_self(self):
        c = Sin(cache=self.cache)
        c.cache = self.cache
        c._cache = self._cache
        return c


class Cos(MathNode):
    """The cosine function.
    """
    INFIX_FMT = 'cos({0})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = 'cos'

    @staticmethod
    def get_arity():
        return 1

    def operation(self, *args):
        return numpy.cos(args[0])

    def infix(self, **kwargs):
        return Cos.INFIX_FMT.format(self.children[0].infix(**kwargs))

    def clone_self(self):
        c = Cos(cache=self.cache)
        c.cache = self.cache
        c._cache = self._cache
        return c


class Exp(MathNode):
    """The natural exponential function.
    """
    INFIX_FMT = 'exp({0})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = 'exp'

    @staticmethod
    def get_arity():
        return 1

    def operation(self, *args):
        return numpy.exp(args[0])

    def infix(self, **kwargs):
        return Exp.INFIX_FMT.format(self.children[0].infix(**kwargs))

    def clone_self(self):
        c = Exp(cache=self.cache)
        c.cache = self.cache
        c._cache = self._cache
        return c


class Abs(MathNode):
    """Absolute value.
    """
    INFIX_FMT = '|{0}|'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = 'abs'

    @staticmethod
    def get_arity():
        return 1

    def operation(self, *args):
        return numpy.abs(args[0])

    def infix(self, **kwargs):
        return Abs.INFIX_FMT.format(self.children[0].infix(**kwargs))

    def clone_self(self):
        c = Abs(cache=self.cache)
        c.cache = self.cache
        c._cache = self._cache
        return c


class Power(MathNode):
    """Power function, i.e. the argument raised to the power given in
    constructor.

    .. warning::

        This is an unprotected power, i.e. undefined powers (e.g. negative power
        of zero or half power of negative number) are not handled in this node.
    """
    INFIX_FMT = '({0}^{1})'

    def __init__(self, power, **kwargs):
        super().__init__(**kwargs)
        self.data = 'pow' + str(power)
        self.power = power

    @staticmethod
    def get_arity():
        return 1

    def operation(self, *args):
        return numpy.power(args[0], self.power)

    def infix(self, **kwargs):
        return Power.INFIX_FMT.format(self.children[0].infix(**kwargs),
                                      self.power)

    def clone_self(self):
        c = Power(cache=self.cache, power=self.power)
        c.cache = self.cache
        c._cache = self._cache
        return c


class Sqrt(MathNode):
    """Square root.

    .. warning::

        This is an unprotected square root, i.e. the square root of negative
        numbers is not handled in this node.
    """
    INFIX_FMT = 'sqrt({0})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = 'sqrt'

    @staticmethod
    def get_arity():
        return 1

    def operation(self, *args):
        return numpy.sqrt(args[0])

    def infix(self, **kwargs) -> str:
        return Sqrt.INFIX_FMT.format(self.children[0].infix(**kwargs))

    def clone_self(self):
        c = Sqrt(cache=self.cache)
        c.cache = self.cache
        c._cache = self._cache
        return c


class PSqrt(MathNode):
    """Protected square root, returns the square root of the absolute value of
    the argument.
    """
    INFIX_FMT = 'psqrt({0})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = 'psqrt'

    @staticmethod
    def get_arity():
        return 1

    def operation(self, *args):
        return numpy.sqrt(args[0])

    def infix(self, **kwargs) -> str:
        return PSqrt.INFIX_FMT.format(self.children[0].infix(**kwargs))

    def clone_self(self):
        c = Sqrt(cache=self.cache)
        c.cache = self.cache
        c._cache = self._cache
        return c


class Sigmoid(MathNode):
    """Sigmoid function: :math:`sig(x) = \\frac{1}{1 + \\mathrm{e}^{-x}}`
    """
    INFIX_FMT = 'sigm({0})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = 'sigm'

    @staticmethod
    def get_arity():
        return 1

    def operation(self, *args):
        return 1 / (1 + numpy.exp(-args[0]))

    def infix(self, **kwargs):
        return Sigmoid.INFIX_FMT.format(self.children[0].infix(**kwargs))

    def clone_self(self):
        c = Sigmoid(cache=self.cache)
        c.cache = self.cache
        c._cache = self._cache
        return c


class Sinc(MathNode):
    """The sinc function: :math:`sinc(x) = \\frac{\\sin{\\pi x}}{\\pi x}`,
    :math:`sinc(0) = 1`.
    """
    INFIX_FMT = 'sinc({0})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = 'sinc'

    @staticmethod
    def get_arity():
        return 1

    def operation(self, *args):
        return numpy.sinc(args[0])

    def infix(self, **kwargs):
        return Sinc.INFIX_FMT.format(self.children[0].infix(**kwargs))

    def clone_self(self):
        c = Sinc(cache=self.cache)
        c.cache = self.cache
        c._cache = self._cache
        return c


class Softplus(MathNode):
    """The softplus or rectifier function:
    :math:`softplus(x) = \\ln(1 + \\mathrm{e}^{x})`.
    """
    INFIX_FMT = 'softplus({0})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = 'softplus'

    @staticmethod
    def get_arity():
        return 1

    def operation(self, *args):
        return numpy.log1p(numpy.exp(args[0]))

    def infix(self, **kwargs):
        return Softplus.INFIX_FMT.format(self.children[0].infix(**kwargs))

    def clone_self(self):
        c = Sigmoid(cache=self.cache)
        c.cache = self.cache
        c._cache = self._cache
        return c


class Const(MathNode):
    """A constant.
    """
    INFIX_FMT = '{0}'

    def __init__(self, val=None, **kwargs):
        super().__init__(data=val, **kwargs)

    @staticmethod
    def get_arity():
        return 0

    def eval(self, args: dict=None):
        return self.data

    def infix(self, **kwargs):
        return Const.INFIX_FMT.format(self.data)

    def clone_self(self):
        c = Const(cache=self.cache, val=self.data)
        c.cache = self.cache
        c._cache = self._cache
        return c


class Variable(MathNode):
    """A variable.
    """

    def __init__(self, name=None, **kwargs):
        super().__init__(data=name, **kwargs)

    @staticmethod
    def get_arity():
        return 0

    def eval(self, args: dict=None):
        return args[self.data]

    def infix(self, **kwargs):
        return self.data

    def clone_self(self):
        c = Variable(cache=self.cache, name=self.data)
        c.cache = self.cache
        c._cache = self._cache
        return c

    def operation(self, *args):
        raise NotImplementedError('Variable does no operation.')


def prepare_args(train_inputs, var_mapping):
    args = {var_mapping[num]: train_inputs[:, num] for num in var_mapping}
    return args
