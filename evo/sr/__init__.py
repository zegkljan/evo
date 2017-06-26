# -*- coding: utf-8 -*-
"""This package contains tools and algorithms for symbolic regression.

The package itself contains common mathematical functions and operators as
subclasses of :class:`evo.gp.support.GpNode` for use in symbolic regression
tasks.
"""

import textwrap

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

    def copy_contents(self, dest):
        super().copy_contents(dest)
        dest.cache = self.cache
        dest._cache = self._cache

    def eval(self, args):
        """Evaluates this node.

        If :attr:`.cache` is set to ``True`` and there is a cached value it is
        returned without the actual evaluation, otherwise the value is computed,
        stored to the cache and returned.

        .. note::

            You shouldn't override this method. Override :meth:`.operation`
            instead.

        :param args: values to set to variables
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

    def eval_child(self, child_no: int, args):
        """Evaluates the child specified by its number and returns the result.

        The children are counted from 0. The ``args`` argument has the same
        meaning as in :meth:`.eval`\ .

        Override this method to implement default scheme of children evaluation.

        :param child_no: zero-based index of the children
        :param args: values to set to variables

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

    def to_matlab_expr(self, data_name='X', function_name_prefix='') -> str:
        """Returns a MATLAB-compatible expression of the subtree rooted in
        this node.

        The function names are going to be identical to the name of the class
        of the node, prefixed with an optional prefix that can be specified
        in the parameters. The exceptions are functions that are already
        MATLAB builtins which are going to be used instead of the custom
        function names.

        :param data_name: the optional name to give to the argument of the
            expression (i.e. to the input data at variable-nodes), default is
            ``'X'``
        :param function_name_prefix: the optional prefix to the name of the
            functions used in the expression, default is empty string (i.e. no
            prefix)
        """
        raise NotImplementedError()

    def to_matlab_def(self, argname='X', outname='Y',
                      function_name_prefix='') -> str:
        """Returns a MATLAB-compatible definition of the function this node
        performs. The returned string is meant to be a content of a MATLAB
        m-file of function.

        The function name is going to be identical to the name of the class
        of the node, prefixed with an optional prefix that can be specified
        in the parameters.

        The arguments to the function are going to be named according to the
        ``argname`` parameter. In case of functiof of more than one argument,
        this name is going to be suffixed with increasing numbers starting from
        1, e.g. ``X1, X2, ...``\ .

        :param argname: the optional name to give to the argument of the
            function, default is ``'X'``
        :param outname: the optional name to give to the output variable of
            the function, default is ``'Y'``
        :param function_name_prefix: the optional prefix to the name of the
            function, default is empty string (i.e. no prefix)
        """
        raise NotImplementedError()


class Add2(MathNode):
    """Addition of two operands: ``a + b``
    """

    INFIX_FMT = '({0} + {1})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data['name'] = '+'

    def get_arity(self=None):
        return 2

    def operation(self, *args):
        return args[0] + args[1]

    def infix(self, **kwargs):
        return Add2.INFIX_FMT.format(self.children[0].infix(**kwargs),
                                     self.children[1].infix(**kwargs))

    def to_matlab_expr(self, data_name='X', function_name_prefix='') -> str:
        c1 = self.children[0].to_matlab_expr(data_name, function_name_prefix)
        c2 = self.children[1].to_matlab_expr(data_name, function_name_prefix)
        return '({arg1} + {arg2})'.format(arg1=c1, arg2=c2)

    def to_matlab_def(self, argname='X', outname='Y',
                      function_name_prefix='') -> str:
        return None


class Sub2(MathNode):
    """Subtraction of the second operand from the first one: ``a - b``
    """
    INFIX_FMT = '({0} - {1})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data['name'] = '-'

    def get_arity(self=None):
        return 2

    def operation(self, *args):
        return args[0] - args[1]

    def infix(self, **kwargs):
        return Sub2.INFIX_FMT.format(self.children[0].infix(**kwargs),
                                     self.children[1].infix(**kwargs))

    def to_matlab_expr(self, data_name='X', function_name_prefix='') -> str:
        c1 = self.children[0].to_matlab_expr(data_name, function_name_prefix)
        c2 = self.children[1].to_matlab_expr(data_name, function_name_prefix)
        return '({arg1} - {arg2})'.format(arg1=c1, arg2=c2)

    def to_matlab_def(self, argname='X', outname='Y',
                      function_name_prefix='') -> str:
        return None


class Mul2(MathNode):
    """Multiplication of two operands: ``a * b``
    """
    INFIX_FMT = '({0} * {1})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data['name'] = '*'

    def get_arity(self=None):
        return 2

    def operation(self, *args):
        return args[0] * args[1]

    def infix(self, **kwargs):
        return Mul2.INFIX_FMT.format(self.children[0].infix(**kwargs),
                                     self.children[1].infix(**kwargs))

    def to_matlab_expr(self, data_name='X', function_name_prefix='') -> str:
        c1 = self.children[0].to_matlab_expr(data_name, function_name_prefix)
        c2 = self.children[1].to_matlab_expr(data_name, function_name_prefix)
        return '({arg1} .* {arg2})'.format(arg1=c1, arg2=c2)

    def to_matlab_def(self, argname='X', outname='Y',
                      function_name_prefix='') -> str:
        return None


class Div2(MathNode):
    """Division of the first operand by the second one: ``a / b``

    .. warning::

        This is an unprotected division, i.e. division by zero is not handled in
        this node.
    """
    INFIX_FMT = '({0} / {1})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data['name'] = '/'

    def get_arity(self=None):
        return 2

    def operation(self, *args):
        return numpy.true_divide(args[0], args[1])

    def infix(self, **kwargs):
        return Div2.INFIX_FMT.format(self.children[0].infix(**kwargs),
                                     self.children[1].infix(**kwargs))

    def to_matlab_expr(self, data_name='X', function_name_prefix='') -> str:
        c1 = self.children[0].to_matlab_expr(data_name, function_name_prefix)
        c2 = self.children[1].to_matlab_expr(data_name, function_name_prefix)
        return '({arg1} ./ {arg2})'.format(arg1=c1, arg2=c2)

    def to_matlab_def(self, argname='X', outname='Y',
                      function_name_prefix='') -> str:
        return None


class IDiv2(MathNode):
    """Integer division of the first operand by the second one: ``a // b``

    .. warning::

        This is an unprotected division, i.e. division by zero is not handled in
        this node.
    """
    INFIX_FMT = '({0} // {1})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data['name'] = '//'

    def get_arity(self=None):
        return 2

    def operation(self, *args):
        return numpy.floor_divide(args[0], args[1])

    def infix(self, **kwargs) -> str:
        return IDiv2.INFIX_FMT.format(self.children[0].infix(**kwargs),
                                      self.children[1].infix(**kwargs))

    def to_matlab_expr(self, data_name='X', function_name_prefix='') -> str:
        c1 = self.children[0].to_matlab_expr(data_name, function_name_prefix)
        c2 = self.children[1].to_matlab_expr(data_name, function_name_prefix)
        return 'idivide({arg1}, {arg2}, \'floor\')'.format(arg1=c1, arg2=c2)

    def to_matlab_def(self, argname='X', outname='Y',
                      function_name_prefix='') -> str:
        return None


class PDiv2(MathNode):
    """Protected division of the first operand by the second one, returns 1 if
    the second operand is zero: ``1 if b == 0 else a / b``
    """
    INFIX_FMT = '({0} {{/}} {1})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data['name'] = '{/}'

    def get_arity(self=None):
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

    def to_matlab_expr(self, data_name='X', function_name_prefix='') -> str:
        c1 = self.children[0].to_matlab_expr(data_name, function_name_prefix)
        c2 = self.children[1].to_matlab_expr(data_name, function_name_prefix)
        return '{pfx}{fn}({arg1}, {arg2})'.format(
            arg1=c1, arg2=c2, pfx=function_name_prefix,
            fn=self.__class__.__name__)

    def to_matlab_def(self, argname='X', outname='Y',
                      function_name_prefix='') -> str:
        return textwrap.dedent('''
        function {out} = {prefix}{name}({arg}1, {arg}2)
        {out} = {arg}1 ./ {arg}2;
        {out}({arg}2 == 0) = 1;
        end
        '''.format(arg=argname, out=outname, prefix=function_name_prefix,
                   name=self.__class__.__name__)).strip()


class PIDiv2(MathNode):
    """Protected integer division of the first operand by the second one,
    returns 1 if the second operand is zero: ``1 if b == 0 else a // b``
    """
    INFIX_FMT = '({0} {{//}} {1})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data['name'] = '{//}'

    def get_arity(self=None):
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

    def to_matlab_expr(self, data_name='X', function_name_prefix='') -> str:
        c1 = self.children[0].to_matlab_expr(data_name, function_name_prefix)
        c2 = self.children[1].to_matlab_expr(data_name, function_name_prefix)
        return '{pfx}{fn}({arg1}, {arg2})'.format(
            arg1=c1, arg2=c2, pfx=function_name_prefix,
            fn=self.__class__.__name__)

    def to_matlab_def(self, argname='X', outname='Y',
                      function_name_prefix='') -> str:
        return textwrap.dedent('''
        function {out} = {prefix}{name}({arg}1, {arg}2)
        {out} = idivide({arg}1, {arg}2, 'floor');
        {out}({arg}2 == 0) = 1;
        end
        '''.format(arg=argname, out=outname, prefix=function_name_prefix,
                   name=self.__class__.__name__)).strip()


class Sin(MathNode):
    """The sine function.
    """
    INFIX_FMT = 'sin({0})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data['name'] = 'sin'

    def get_arity(self=None):
        return 1

    def operation(self, *args):
        return numpy.sin(args[0])

    def infix(self, **kwargs):
        return Sin.INFIX_FMT.format(self.children[0].infix(**kwargs))

    def to_matlab_expr(self, data_name='X', function_name_prefix='') -> str:
        c = self.children[0].to_matlab_expr(data_name, function_name_prefix)
        return 'sin({arg})'.format(arg=c)

    def to_matlab_def(self, argname='X', outname='Y',
                      function_name_prefix='') -> str:
        return None


class Cos(MathNode):
    """The cosine function.
    """
    INFIX_FMT = 'cos({0})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data['name'] = 'cos'

    def get_arity(self=None):
        return 1

    def operation(self, *args):
        return numpy.cos(args[0])

    def infix(self, **kwargs):
        return Cos.INFIX_FMT.format(self.children[0].infix(**kwargs))

    def to_matlab_expr(self, data_name='X', function_name_prefix='') -> str:
        c = self.children[0].to_matlab_expr(data_name, function_name_prefix)
        return 'cos({arg})'.format(arg=c)

    def to_matlab_def(self, argname='X', outname='Y',
                      function_name_prefix='') -> str:
        return None


class Exp(MathNode):
    """The natural exponential function.
    """
    INFIX_FMT = 'exp({0})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data['name'] = 'exp'

    def get_arity(self=None):
        return 1

    def operation(self, *args):
        return numpy.exp(args[0])

    def infix(self, **kwargs):
        return Exp.INFIX_FMT.format(self.children[0].infix(**kwargs))

    def to_matlab_expr(self, data_name='X', function_name_prefix='') -> str:
        c = self.children[0].to_matlab_expr(data_name, function_name_prefix)
        return 'exp({arg})'.format(arg=c)

    def to_matlab_def(self, argname='X', outname='Y',
                      function_name_prefix='') -> str:
        return None


class Abs(MathNode):
    """Absolute value.
    """
    INFIX_FMT = 'abs({0})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data['name'] = 'abs'

    def get_arity(self=None):
        return 1

    def operation(self, *args):
        return numpy.abs(args[0])

    def infix(self, **kwargs):
        return Abs.INFIX_FMT.format(self.children[0].infix(**kwargs))

    def to_matlab_expr(self, data_name='X', function_name_prefix='') -> str:
        c = self.children[0].to_matlab_expr(data_name, function_name_prefix)
        return 'abs({arg})'.format(arg=c)

    def to_matlab_def(self, argname='X', outname='Y',
                      function_name_prefix='') -> str:
        return None


class Power(MathNode):
    """Power function, i.e. the argument raised to the power given in
    constructor.

    .. warning::

        This is an unprotected power, i.e. undefined powers (e.g. negative power
        of zero or half power of negative number) are not handled in this node.
    """
    INFIX_FMT = '({0}^{1})'

    def __init__(self, power=None, **kwargs):
        super().__init__(**kwargs)
        self.data['name'] = 'pow' + str(power)
        self.power = power

    def get_arity(self=None):
        return 1

    def operation(self, *args):
        return args[0] ** self.power

    def infix(self, **kwargs):
        return Power.INFIX_FMT.format(self.children[0].infix(**kwargs),
                                      self.power)

    def copy_contents(self, dest):
        super().copy_contents(dest)
        dest.power = self.power

    def to_matlab_expr(self, data_name='X', function_name_prefix='') -> str:
        c = self.children[0].to_matlab_expr(data_name, function_name_prefix)
        return '({arg} .^ {exp})'.format(arg=c, exp=repr(self.power))

    def to_matlab_def(self, argname='X', outname='Y',
                      function_name_prefix='') -> str:
        return None


class Sqrt(MathNode):
    """Square root.

    .. warning::

        This is an unprotected square root, i.e. the square root of negative
        numbers is not handled in this node.
    """
    INFIX_FMT = 'sqrt({0})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data['name'] = 'sqrt'

    def get_arity(self=None):
        return 1

    def operation(self, *args):
        return numpy.sqrt(args[0])

    def infix(self, **kwargs) -> str:
        return Sqrt.INFIX_FMT.format(self.children[0].infix(**kwargs))

    def to_matlab_expr(self, data_name='X', function_name_prefix='') -> str:
        c = self.children[0].to_matlab_expr(data_name, function_name_prefix)
        return 'sqrt({arg})'.format(arg=c)

    def to_matlab_def(self, argname='X', outname='Y',
                      function_name_prefix='') -> str:
        return None


class PSqrt(MathNode):
    """Protected square root, returns the square root of the absolute value of
    the argument.
    """
    INFIX_FMT = 'psqrt({0})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data['name'] = 'psqrt'

    def get_arity(self=None):
        return 1

    def operation(self, *args):
        return numpy.sqrt(numpy.abs(args[0]))

    def infix(self, **kwargs) -> str:
        return PSqrt.INFIX_FMT.format(self.children[0].infix(**kwargs))

    def to_matlab_expr(self, data_name='X', function_name_prefix='') -> str:
        c = self.children[0].to_matlab_expr(data_name, function_name_prefix)
        return '{pfx}{fn}({arg})'.format(arg=c, pfx=function_name_prefix,
                                         fn=self.__class__.__name__)

    def to_matlab_def(self, argname='X', outname='Y',
                      function_name_prefix='') -> str:
        return textwrap.dedent('''
        function {out} = {prefix}{name}({arg})
        {out} = sqrt(abs({arg}));
        end
        '''.format(arg=argname, out=outname, prefix=function_name_prefix,
                   name=self.__class__.__name__)).strip()


class Sigmoid(MathNode):
    """Sigmoid function: :math:`sigmoid(x) = \\frac{1}{1 + \\mathrm{e}^{-x}}`
    """
    INFIX_FMT = 'sigmoid({0})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data['name'] = 'sigm'

    def get_arity(self=None):
        return 1

    def operation(self, *args):
        return 1 / (1 + numpy.exp(-args[0]))

    def infix(self, **kwargs):
        return Sigmoid.INFIX_FMT.format(self.children[0].infix(**kwargs))

    def to_matlab_expr(self, data_name='X', function_name_prefix='') -> str:
        c = self.children[0].to_matlab_expr(data_name, function_name_prefix)
        return '{pfx}{fn}({arg})'.format(arg=c, pfx=function_name_prefix,
                                      fn=self.__class__.__name__)

    def to_matlab_def(self, argname='X', outname='Y',
                      function_name_prefix='') -> str:
        return textwrap.dedent('''
        function {out} = {prefix}{name}({arg})
        {out} = 1 ./ (1 + exp(-{arg}));
        end
        '''.format(arg=argname, out=outname, prefix=function_name_prefix,
                   name=self.__class__.__name__)).strip()


class Tanh(MathNode):
    """Hyperbolic tangent
    """
    INFIX_FMT = 'tanh({0})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data['name'] = 'tanh'

    def get_arity(self=None):
        return 1

    def operation(self, *args):
        return numpy.tanh(args[0])

    def infix(self, **kwargs):
        return Tanh.INFIX_FMT.format(self.children[0].infix(**kwargs))

    def to_matlab_expr(self, data_name='X', function_name_prefix='') -> str:
        c = self.children[0].to_matlab_expr(data_name, function_name_prefix)
        return 'tanh({arg})'.format(arg=c)

    def to_matlab_def(self, argname='X', outname='Y',
                      function_name_prefix='') -> str:
        return None


class Sinc(MathNode):
    """The sinc function: :math:`sinc(x) = \\frac{\\sin{\\pi x}}{\\pi x}`,
    :math:`sinc(0) = 1`.
    """
    INFIX_FMT = 'sinc({0})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data['name'] = 'sinc'

    def get_arity(self=None):
        return 1

    def operation(self, *args):
        return numpy.sinc(args[0])

    def infix(self, **kwargs):
        return Sinc.INFIX_FMT.format(self.children[0].infix(**kwargs))

    def to_matlab_expr(self, data_name='X', function_name_prefix='') -> str:
        c = self.children[0].to_matlab_expr(data_name, function_name_prefix)
        return 'sinc({arg})'.format(arg=c)

    def to_matlab_def(self, argname='X', outname='Y',
                      function_name_prefix='') -> str:
        return None


class Softplus(MathNode):
    """The softplus or rectifier function:
    :math:`softplus(x) = \\ln(1 + \\mathrm{e}^{x})`.
    """
    INFIX_FMT = 'softplus({0})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data['name'] = 'softplus'

    def get_arity(self=None):
        return 1

    def operation(self, *args):
        return numpy.log1p(numpy.exp(args[0]))

    def infix(self, **kwargs):
        return Softplus.INFIX_FMT.format(self.children[0].infix(**kwargs))

    def to_matlab_expr(self, data_name='X', function_name_prefix='') -> str:
        c = self.children[0].to_matlab_expr(data_name, function_name_prefix)
        return '{pfx}{fn}({arg})'.format(arg=c, pfx=function_name_prefix,
                                         fn=self.__class__.__name__)

    def to_matlab_def(self, argname='X', outname='Y',
                      function_name_prefix='') -> str:
        return textwrap.dedent('''
        function {out} = {prefix}{name}({arg})
        {out} = log1p(exp({arg}));
        end
        '''.format(arg=argname, out=outname, prefix=function_name_prefix,
                   name=self.__class__.__name__)).strip()


class Gauss(MathNode):
    """The Gauss-function simplified to the core structural form:
    :math:`gauss(x) = \\mathrm{e}^{-x^2}`.
    """
    INFIX_FMT = 'gauss({0})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data['name'] = 'gauss'

    def get_arity(self=None):
        return 1

    def operation(self, *args):
        return numpy.exp(-(args[0] ** 2))

    def infix(self, **kwargs):
        return Softplus.INFIX_FMT.format(self.children[0].infix(**kwargs))

    def to_matlab_expr(self, data_name='X', function_name_prefix='') -> str:
        c = self.children[0].to_matlab_expr(data_name, function_name_prefix)
        return '{pfx}{fn}({arg})'.format(arg=c, pfx=function_name_prefix,
                                         fn=self.__class__.__name__)

    def to_matlab_def(self, argname='X', outname='Y',
                      function_name_prefix='') -> str:
        return textwrap.dedent('''
        function {out} = {prefix}{name}({arg})
        {out} = exp(-({arg} .^ 2));
        end
        '''.format(arg=argname, out=outname, prefix=function_name_prefix,
                   name=self.__class__.__name__)).strip()


class BentIdentity(MathNode):
    """A non-linear strictly increasing function without limits in +- infinity.

    The function is: :math:`bid(x) = \\frac{\\sqrt{x^2 + 1} - 1}{2} + x`.
    """
    INFIX_FMT = 'bentid({0})'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data['name'] = 'bentid'

    def get_arity(self=None):
        return 1

    def operation(self, *args):
        return 0.5 * (numpy.sqrt(args[0] ** 2 + 1) - 1) + args[0]

    def infix(self, **kwargs):
        return BentIdentity.INFIX_FMT.format(self.children[0].infix(**kwargs))

    def to_matlab_expr(self, data_name='X', function_name_prefix='') -> str:
        c = self.children[0].to_matlab_expr(data_name, function_name_prefix)
        return '{pfx}{fn}({arg})'.format(arg=c, pfx=function_name_prefix,
                                         fn=self.__class__.__name__)

    def to_matlab_def(self, argname='X', outname='Y',
                      function_name_prefix='') -> str:
        return textwrap.dedent('''
        function {out} = {prefix}{name}({arg})
        {out} = 0.5 * (sqrt({arg} .^ 2 + 1) - 1) + {arg}
        end
        ''')


class Const(MathNode):
    """A constant.
    """
    INFIX_FMT = '{0}'

    def __init__(self, val=None, **kwargs):
        super().__init__(**kwargs)
        self.data['name'] = str(val)
        self.value = val

    def get_arity(self=None):
        return 0

    def eval(self, args):
        return self.value

    def infix(self, **kwargs):
        return Const.INFIX_FMT.format(self.data['name'])

    def copy_contents(self, dest):
        super().copy_contents(dest)
        dest.value = self.value

    def to_matlab_expr(self, data_name='X', function_name_prefix='') -> str:
        return repr(self.value)

    def to_matlab_def(self, argname='X', outname='Y',
                      function_name_prefix='') -> str:
        return None


class Variable(MathNode):
    """A variable.
    """

    def __init__(self, index=None, name=None, **kwargs):
        super().__init__(**kwargs)
        self.index = index
        if name is None:
            self.data['name'] = 'x{}'.format(index)
        else:
            self.data['name'] = name

    def get_arity(self=None):
        return 0

    def eval(self, args):
        return args[:, self.index]

    def infix(self, **kwargs):
        return self.data['name']

    def operation(self, *args):
        raise NotImplementedError('Variable does no operation.')

    def copy_contents(self, dest):
        super().copy_contents(dest)
        dest.index = self.index

    def to_matlab_expr(self, data_name='X', function_name_prefix='') -> str:
        return '{var}(:, {idx})'.format(var=data_name, idx=self.index + 1)

    def to_matlab_def(self, argname='X', outname='Y',
                      function_name_prefix='') -> str:
        return None
