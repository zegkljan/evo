"""This module contains algorithms and data structures for backpropagation-based
learning of tree expressions.
"""

import numpy

import evo.sr


def identity(x):
    """Returns the argument."""
    return x


def identity_d(_):
    """Returns 1 (derivative of :meth:`identity`\ )."""
    return 1


class WeightedNode(evo.sr.MathNode):
    """An abstract class for nodes that have a weight and/or bias assigned for
    their children, effectively carrying constants all along.
    """
    def __init__(self, tune_bias=True, tune_weights=True, **kwargs):
        """
        ``tune_bias`` specifies whether the bias in this node is subject to
        optimisation. Plain boolean value (e.g. ``True``\ ) specifies this for
        all biases. List of boolean values (e.g. ``[True, False``\ ) specifies
        this per argument on the corresponding positions.

        ``tune_weights`` works in exactly the same way as ``tune_bias``
        described above.

        :param cache: specifies whether the output of the node is going to be
            cached
        :type tune_bias: :class:`bool` or :class:`list` of bools
        :param tune_bias: specifies whether or which biases are allowed to be
            tuned
        :type tune_weights: :class:`bool` or :class:`list` of bools
        :param tune_weights: specifies whether or which weights are allowed to
            be tuned
        """
        super().__init__(**kwargs)
        self.bias = numpy.zeros(self.get_arity())
        self.weights = numpy.ones(self.get_arity())
        self.argument = None

        self.tune_bias = tune_bias
        self.tune_weights = tune_weights

    def copy_contents(self, dest):
        super().copy_contents(dest)
        dest.bias = self.bias.copy()
        dest.weights = self.weights.copy()
        if self.argument is None:
            dest.argument = None
        else:
            dest.argument = self.argument.copy()
        dest.tune_bias = self.tune_bias
        dest.tune_weights = self.tune_weights

    def eval_child(self, child_no: int, args: dict = None):
        val = super().eval_child(child_no, args)
        return self.weights[child_no] * val + self.bias[child_no]

    def operation(self, *args):
        l = max(map(lambda a: numpy.shape([a])[-1], args))
        self.argument = numpy.empty((l, self.get_arity()))
        for i in range(self.get_arity()):
            self.argument[:, i] = args[i]
        return super().operation(*args)

    def derivative(self, arg_no: int, x):
        """Returns the value of the derivative of the node's function, related
        to the given argument, at ``x``.

        :param arg_no: number of argument the derivative is related to, counted
            from 0
        :param x: point the derivative is to be computed at
        """
        raise NotImplementedError()

    def infix(self, **kwargs) -> str:
        return self.full_infix(**kwargs)

    def full_infix(self, **kwargs) -> str:
        """Returns the string representation of the tree in an infix form with
        all the weights and biases.

        :keyword num_format: format for weights and biases
        :return: the string representation of the tree
        """
        raise NotImplementedError()

    def backpropagate(self, args, datapts_no: int,
                      cost_derivative: callable=None, true_output=None,
                      output_transform: callable=None,
                      output_transform_derivative: callable=None):
        if output_transform is None:
            output_transform = identity
        if output_transform_derivative is None:
            output_transform_derivative = identity_d

        # bias derivative
        self.data['d_bias'] = numpy.empty((datapts_no, self.bias.size))
        # if this is the root call, do it differently
        if cost_derivative is not None and true_output is not None:
            for i in range(self.bias.size):
                cd = cost_derivative(output_transform(self.eval(args)),
                                     true_output)
                otfd = output_transform_derivative(self.eval(args))
                nd = self.derivative(i, self.argument)
                self.data['d_bias'][:, i] = cd * otfd * nd
        else:
            for i in range(self.bias.size):
                pd = self.parent.data['d_bias'][:, self.parent_index]
                pw = self.parent.weights[self.parent_index]
                sd = self.derivative(i, self.argument)
                self.data['d_bias'][:, i] = pd * pw * sd

        # weights derivative
        inputs = None
        if self.tune_weights is True:
            raw = [c.eval(args) for c in self.children]
            if len(raw) == 1:
                inputs = numpy.array(raw)
            else:
                inputs = numpy.column_stack(numpy.broadcast(*raw))
        elif self.tune_weights:
            raw = [self.children[i].eval(args) if self.tune_weights[i] else 0
                   for i in range(len(self.children))]
            inputs = numpy.column_stack(numpy.broadcast(*raw))
        if inputs is not None and len(inputs) > 0:
            self.data['d_weights'] = self.data['d_bias'] * inputs.T

        for c in self.children:
            if isinstance(c, WeightedNode):
                c.backpropagate(args, datapts_no)


class Add2(WeightedNode, evo.sr.Add2):
    """Weighted version of :class:`evo.sr.Add2`\ .

    .. seealso:: :class:`evo.sr.Add2`
    """
    def __init__(self, **kwargs):
        """Add2 does never use bias (only weights), therefore ``tune_bias``
        argument has no effect and is always overridden to ``False``\ .
        """
        kwargs['tune_bias'] = False
        super().__init__(**kwargs)

    def eval_child(self, child_no: int, args: dict = None):
        val = super(evo.sr.Add2, self).eval_child(child_no, args)
        return self.weights[child_no] * val

    def derivative(self, arg_no: int, x):
        return 1

    def full_infix(self, **kwargs):
        num_format = kwargs.get('num_format', '.3f')
        if num_format == 'repr':
            return '({0} * {2} + {1} * {3})'.format(
                repr(self.weights[0]), repr(self.weights[1]),
                self.children[0].infix(**kwargs),
                self.children[1].infix(**kwargs))
        return ('({0:' + num_format + '} * {2} + '
                '{1:' + num_format + '} * {3})').format(
            self.weights[0], self.weights[1],
            self.children[0].infix(**kwargs),
            self.children[1].infix(**kwargs))


class Sub2(WeightedNode, evo.sr.Sub2):
    """Weighted version of :class:`evo.sr.Sub2`\ .

    .. seealso:: :class:`evo.sr.Sub2`
    """
    def __init__(self, **kwargs):
        """Sub2 does never use bias (only weights), therefore ``tune_bias``
        argument has no effect and is always overridden to ``False``\ .
        """
        kwargs['tune_bias'] = False
        super().__init__(**kwargs)

    def eval_child(self, child_no: int, args: dict = None):
        val = super(evo.sr.Sub2, self).eval_child(child_no, args)
        return self.weights[child_no] * val

    def derivative(self, arg_no: int, x):
        if arg_no == 0:
            return 1
        elif arg_no == 1:
            return -1
        raise ValueError('Invalid argument number.')

    def full_infix(self, **kwargs):
        num_format = kwargs.get('num_format', '.3f')
        if num_format == 'repr':
            return '({0} * {2} - {1} * {3})'.format(
                repr(self.weights[0]), repr(self.weights[1]),
                self.children[0].infix(**kwargs),
                self.children[1].infix(**kwargs))
        return ('({0:' + num_format + '} * {2} - '
                '{1:' + num_format + '} * {3})').format(
            self.weights[0], self.weights[1],
            self.children[0].infix(**kwargs),
            self.children[1].infix(**kwargs))


class Mul2(WeightedNode, evo.sr.Mul2):
    """Weighted version of :class:`evo.sr.Mul2`\ .

    .. seealso:: :class:`evo.sr.Mul2`
    """
    def __init__(self, **kwargs):
        """Mul2 does never use weights (only biases), therefore ``tune_weights``
        argument has no effect and is always overridden to ``False``\ .
        """
        kwargs['tune_weights'] = False
        super().__init__(**kwargs)

    def eval_child(self, child_no: int, args: dict = None):
        val = super(evo.sr.Mul2, self).eval_child(child_no, args)
        return self.bias[child_no] + val

    def derivative(self, arg_no: int, x):
        if arg_no == 0:
            return x[:, 1]
        if arg_no == 1:
            return x[:, 0]
        raise ValueError('Invalid arg_no.')

    def full_infix(self, **kwargs):
        num_format = kwargs.get('num_format', '.3f')
        if num_format == 'repr':
            return '(({0} + {2}) * ({1} + {3}))'.format(
                repr(self.bias[0]), repr(self.bias[1]),
                self.children[0].infix(**kwargs),
                self.children[1].infix(**kwargs))
        return ('(({0:' + num_format + '} + {2}) * '
                '({1:' + num_format + '} + {3}))').format(
            self.bias[0], self.bias[1],
            self.children[0].infix(**kwargs),
            self.children[1].infix(**kwargs))


class Div2(WeightedNode, evo.sr.Div2):
    """Weighted version of :class:`evo.sr.Div2`\ .

    .. seealso:: :class:`evo.sr.Div2`
    """
    def __init__(self, **kwargs):
        """Div2 does never use weights (only biases), therefore ``tune_weights``
        argument has no effect and is always overridden to ``False``\ .
        """
        kwargs['tune_weights'] = False
        super().__init__(**kwargs)

    def eval_child(self, child_no: int, args: dict = None):
        val = super(evo.sr.Div2, self).eval_child(child_no, args)
        return self.bias[child_no] + val

    def derivative(self, arg_no: int, x):
        if arg_no == 0:
            return numpy.true_divide(1.0, numpy.square(x[:, 1]))
        if arg_no == 1:
            return -numpy.true_divide(x[:, 0], numpy.square(x[:, 1]))
        raise ValueError('Invalid arg_no.')

    def full_infix(self, **kwargs):
        num_format = kwargs.get('num_format', '.3f')
        if num_format == 'repr':
            return '(({0} + {2}) / ({1} + {3}))'.format(
                repr(self.bias[0]), repr(self.bias[1]),
                self.children[0].infix(**kwargs),
                self.children[1].infix(**kwargs))
        return ('(({0:' + num_format + '} + {2}) / '
                '({1:' + num_format + '} + {3}))').format(
            self.bias[0], self.bias[1],
            self.children[0].infix(**kwargs),
            self.children[1].infix(**kwargs))


class Sin(WeightedNode, evo.sr.Sin):
    """Weighted version of :class:`evo.sr.Sin`\ .

    .. seealso:: :class:`evo.sr.Sin`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def derivative(self, arg_no: int, x):
        return numpy.cos(x[:, 0])

    def full_infix(self, **kwargs):
        num_format = kwargs.get('num_format', '.3f')
        if num_format == 'repr':
            return 'sin({0} + {1} * {2})'.format(
                repr(self.bias[0]), repr(self.weights[0]),
                self.children[0].infix(**kwargs))
        return ('sin({0:' + num_format + '} + '
                '{1:' + num_format + '} * {2})').format(
            self.bias[0], self.weights[0],
            self.children[0].infix(**kwargs))


class Cos(WeightedNode, evo.sr.Cos):
    """Weighted version of :class:`evo.sr.Cos`\ .

    .. seealso:: :class:`evo.sr.Cos`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def derivative(self, arg_no: int, x):
        return -numpy.sin(x[:, 0])

    def full_infix(self, **kwargs):
        num_format = kwargs.get('num_format', '.3f')
        if num_format == 'repr':
            return 'cos({0} + {1} * {2})'.format(
                repr(self.bias[0]), repr(self.weights[0]),
                self.children[0].infix(**kwargs))
        return ('cos({0:' + num_format + '} + '
                '{1:' + num_format + '} * {2})').format(
            self.bias[0], self.weights[0],
            self.children[0].infix(**kwargs))


class Exp(WeightedNode, evo.sr.Exp):
    """Weighted version of :class:`evo.sr.Exp`\ .

    .. seealso:: :class:`evo.sr.Exp`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def derivative(self, arg_no: int, x):
        return numpy.exp(x[:, 0])

    def full_infix(self, **kwargs):
        num_format = kwargs.get('num_format', '.3f')
        if num_format == 'repr':
            return 'exp({0} + {1} * {2})'.format(
                repr(self.bias[0]), repr(self.weights[0]),
                self.children[0].infix(**kwargs))
        return ('exp({0:' + num_format + '} + '
                '{1:' + num_format + '} * {2})').format(
            self.bias[0], self.weights[0],
            self.children[0].infix(**kwargs))


class Abs(WeightedNode, evo.sr.Abs):
    """Weighted version of :class:`evo.sr.Abs`\ .

    .. seealso:: :class:`evo.sr.Abs`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def derivative(self, arg_no: int, x):
        return numpy.sign(x[:, 0])

    def full_infix(self, **kwargs):
        num_format = kwargs.get('num_format', '.3f')
        if num_format == 'repr':
            return 'abs({0} + {1} * {2})'.format(
                repr(self.bias[0]), repr(self.weights[0]),
                self.children[0].infix(**kwargs))
        return ('abs({0:' + num_format + '} + '
                '{1:' + num_format + '} * {2})').format(
            self.bias[0], self.weights[0],
            self.children[0].infix(**kwargs))


class Power(WeightedNode, evo.sr.Power):
    """Weighted version of :class:`evo.sr.Power`\ .

    .. seealso:: :class:`evo.sr.Power`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def derivative(self, arg_no: int, x):
        return self.power * (x[:, 0] ** (self.power - 1))

    def full_infix(self, **kwargs):
        num_format = kwargs.get('num_format', '.3f')
        if num_format == 'repr':
            return '(({0} + {1} * {3})^{2})'.format(
                repr(self.bias[0]), repr(self.weights[0]), self.power,
                self.children[0].infix(**kwargs))
        base = '(({0:' + num_format + '} + {1:' + num_format + '} * {3})^{2})'
        return base.format(
            self.bias[0], self.weights[0], self.power,
            self.children[0].infix(**kwargs))


class Sigmoid(WeightedNode, evo.sr.Sigmoid):
    """Weighted version of :class:`evo.sr.Sigmoid`\ .

    .. seealso:: :class:`evo.sr.Sigmoid`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def derivative(self, arg_no: int, x):
        a = 1 / (1 + numpy.exp(-x[:, 0]))
        return a * (1 - a)

    def full_infix(self, **kwargs):
        num_format = kwargs.get('num_format', '.3f')
        if num_format == 'repr':
            return 'sig({0} + {1} * {2})'.format(
                repr(self.bias[0]), repr(self.weights[0]),
                self.children[0].infix(**kwargs))
        return ('sig({0:' + num_format + '} + '
                '{1:' + num_format + '} * {2})').format(
            self.bias[0], self.weights[0],
            self.children[0].infix(**kwargs))


class Tanh(WeightedNode, evo.sr.Tanh):
    """Weighted version of :class:`evo.sr.Tanh`\ .

    .. seealso:: :class:`evo.sr.Tanh`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def derivative(self, arg_no: int, x):
        a = numpy.tanh(x[:, 0])
        return 1 - a ** 2

    def full_infix(self, **kwargs):
        num_format = kwargs.get('num_format', '.3f')
        if num_format == 'repr':
            return 'tanh({0} + {1} * {2})'.format(
                repr(self.bias[0]), repr(self.weights[0]),
                self.children[0].infix(**kwargs))
        return ('tanh({0:' + num_format + '} + '
                '{1:' + num_format + '} * {2})').format(
            self.bias[0], self.weights[0],
            self.children[0].infix(**kwargs))


class Sinc(WeightedNode, evo.sr.Sinc):
    """Weighted version of :class:`evo.sr.Sinc`\ .

    .. seealso:: :class:`evo.sr.Sinc`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def derivative(self, arg_no: int, x):
        return numpy.true_divide(
            x[:, 0] * numpy.cos(x[:, 0]) - numpy.sin(x[:, 0]),
            x[:, 0]**2)

    def full_infix(self, **kwargs):
        num_format = kwargs.get('num_format', '.3f')
        if num_format == 'repr':
            return 'sinc({0} + {1} * {2})'.format(
                repr(self.bias[0]), repr(self.weights[0]),
                self.children[0].infix(**kwargs))
        return ('sinc({0:' + num_format + '} + '
                '{1:' + num_format + '} * {2})').format(
            self.bias[0], self.weights[0],
            self.children[0].infix(**kwargs))


class Softplus(WeightedNode, evo.sr.Softplus):
    """Weighted version of :class:`evo.sr.Softplus`\ .

    .. seealso:: :class:`evo.sr.Softplus`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def derivative(self, arg_no: int, x):
        a = numpy.exp(x[:, 0])
        return a / (a + 1)

    def full_infix(self, **kwargs):
        num_format = kwargs.get('num_format', '.3f')
        if num_format == 'repr':
            return 'softplus({0} + {1} * {2})'.format(
                repr(self.bias[0]), repr(self.weights[0]),
                self.children[0].infix(**kwargs))
        return ('softplus({0:' + num_format + '} + '
                '{1:' + num_format + '} * {2})').format(
            self.bias[0], self.weights[0],
            self.children[0].infix(**kwargs))


class Gauss(WeightedNode, evo.sr.Gauss):
    """Weighted version of :class:`evo.sr.Gauss`\ .

    .. seealso:: :class:`evo.sr.Gauss`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def derivative(self, arg_no: int, x):
        a = self.operation(x[:, 0])
        return -2 * a * x[:, 0]

    def full_infix(self, **kwargs):
        num_format = kwargs.get('num_format', '.3f')
        if num_format == 'repr':
            return 'gauss({0} + {1} * {2})'.format(
                repr(self.bias[0]), repr(self.weights[0]),
                self.children[0].infix(**kwargs))
        return ('gauss({0:' + num_format + '} + '
                '{1:' + num_format + '} * {2})').format(
            self.bias[0], self.weights[0],
            self.children[0].infix(**kwargs))


class WeightsUpdater(object):
    """This is a base class for algorithms for updating weights on trees.
    """
    def __init__(self):
        self.updated = False

    def update(self, root: WeightedNode, error, prev_error) -> bool:
        self.updated = False
        root.preorder(self.update_node)
        return self.updated

    def update_node(self, node):
        raise NotImplementedError()


class RpropBase(WeightsUpdater):
    """This is a base class for Rprop algorithm variants.
    """
    def __init__(self, delta_init=0.1, delta_min=1e-6, delta_max=50,
                 eta_minus=0.5, eta_plus=1.2):
        super().__init__()
        self.delta_init = delta_init
        self.eta_plus = eta_plus
        self.delta_max = delta_max
        self.eta_minus = eta_minus
        self.delta_min = delta_min

    def update_node(self, node):
        raise NotImplementedError()


class RpropPlus(RpropBase):
    """This is the original Rprop algorithm [Riedmiller1993] (with
    weight-backtracking), also called Rprop\ :sup:`+` [Igel2000].

    .. [Riedmiller1993] Martin Riedmiller and Heinrich Braun. `A Direct Adaptive
        Method for Faster Backpropagation Learning: The RPROP Algorithm .`
        IEEE International Conference on Neural Networks. 1993. pp. 586-591.

    .. [Igel2000] Christian Igel and Michael Hüsken. `Improving the Rprop
        Learning Algorithm.` Proceedings of the Second International Symposium
        on Neural Computation (NC 2000). 2000.
    """

    def upd(self, w, d, prev_d, delta, prev_update, arity):
        s_prev = numpy.sign(prev_d)
        s = numpy.sign(d)
        for i in range(arity):
            if s_prev[i] * s[i] > 0:
                delta[i] *= self.eta_plus
                if delta[i] > self.delta_max:
                    delta[i] = self.delta_max
                prev_update[i] = -s[i] * delta[i]
                w[i] += prev_update[i]
                prev_d[i] = d[i]
            elif s_prev[i] * s[i] < 0:
                delta[i] *= self.eta_minus
                if delta[i] < self.delta_min:
                    delta[i] = self.delta_min
                w[i] -= prev_update[i]
                prev_d[i] = 0
            else:
                prev_update[i] = -s[i] * delta[i]
                w[i] += prev_update[i]
                prev_d[i] = d[i]

    def update_node(self, node):
        if not isinstance(node, WeightedNode):
            return
        upd = False
        if node.tune_bias and 'd_bias' in node.data:
            upd = True
            d_bias = numpy.sum(node.data['d_bias'], axis=0)
            if 'prev_d_bias' not in node.data:
                node.data['prev_d_bias'] = numpy.zeros(d_bias.size)

            if 'delta_bias' not in node.data:
                node.data['delta_bias'] = (numpy.ones(node.bias.shape) *
                                           self.delta_init)

            if 'prev_bias_update' not in node.data:
                node.data['prev_bias_update'] = numpy.zeros(node.bias.shape)

            self.upd(node.bias, d_bias, node.data['prev_d_bias'],
                     node.data['delta_bias'], node.data['prev_bias_update'],
                     node.bias.size)

        if node.tune_weights and 'd_weights' in node.data:
            upd = True
            d_weights = numpy.sum(node.data['d_weights'], axis=0)
            if 'prev_d_weights' not in node.data:
                node.data['prev_d_weights'] = numpy.zeros(d_weights.size)

            if 'delta_weights' not in node.data:
                node.data['delta_weights'] = (numpy.ones(node.weights.shape) *
                                              self.delta_init)

            if 'prev_weight_update' not in node.data:
                node.data['prev_weight_update'] = numpy.zeros(node.weights.shape)

            self.upd(node.weights, d_weights, node.data['prev_d_weights'],
                     node.data['delta_weights'],
                     node.data['prev_weight_update'], node.weights.size)

        if upd:
            self.updated = True
            node.notify_change()


class RpropMinus(RpropBase):
    """This is the Rprop algorithm without weight-backtracking, also called
    Rprop\ :sup:`-` [Igel2000].

    .. [Igel2000] Christian Igel and Michael Hüsken. `Improving the Rprop
        Learning Algorithm.` Proceedings of the Second International Symposium
        on Neural Computation (NC 2000). 2000.
    """

    def upd(self, w, d, prev_d, delta, arity):
        s_prev = numpy.sign(prev_d)
        s = numpy.sign(d)
        for i in range(arity):
            if s_prev[i] * s[i] > 0:
                delta[i] *= self.eta_plus
                if delta[i] > self.delta_max:
                    delta[i] = self.delta_max
            elif s_prev[i] * s[i] < 0:
                delta[i] *= self.eta_minus
                if delta[i] < self.delta_min:
                    delta[i] = self.delta_min
            w[i] -= s[i] * delta[i]
            prev_d[i] = d[i]

    def update_node(self, node):
        if not isinstance(node, WeightedNode):
            return
        upd = False
        if node.tune_bias and 'd_bias' in node.data:
            upd = True
            d_bias = numpy.sum(node.d_bias, axis=0)
            if 'prev_d_bias' not in node.data:
                node.data['prev_d_bias'] = numpy.zeros(d_bias.size)

            if 'delta_bias' not in node.data:
                node.data['delta_bias'] = (numpy.ones(node.bias.shape) *
                                           self.delta_init)

            self.upd(node.bias, d_bias, node.data['prev_d_bias'],
                     node.data['delta_bias'], node.bias.size)

        if node.tune_weights and 'd_weights' in node.data:
            upd = True
            d_weights = numpy.sum(node.data['d_weights'], axis=0)
            if 'prev_d_weights' not in node.data:
                node.data['prev_d_weights'] = numpy.zeros(d_weights.size)

            if 'delta_weights' not in node.data:
                node.data['delta_weights'] = (numpy.ones(node.weights.shape) *
                                              self.delta_init)

            self.upd(node.weights, d_weights, node.data['prev_d_weights'],
                     node.data['delta_weights'], node.weights.size)

        if upd:
            self.updated = True
            node.notify_change()


class IRpropPlus(RpropBase):
    """This is the improved Rprop algorithm with weight-backtracking, also called
    iRprop\ :sup:`+` [Igel2000].

    .. [Igel2000] Christian Igel and Michael Hüsken. `Improving the Rprop
        Learning Algorithm.` Proceedings of the Second International Symposium
        on Neural Computation (NC 2000). 2000.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_worsened = False

    def upd(self, w, d, prev_d, delta, prev_update, arity):
        s_prev = numpy.sign(prev_d)
        s = numpy.sign(d)
        for i in range(arity):
            if s_prev[i] * s[i] > 0:
                delta[i] *= self.eta_plus
                if delta[i] > self.delta_max:
                    delta[i] = self.delta_max
                prev_update[i] = -s[i] * delta[i]
                w[i] += prev_update[i]
                prev_d[i] = d[i]
            elif s_prev[i] * s[i] < 0:
                delta[i] *= self.eta_minus
                if delta[i] < self.delta_min:
                    delta[i] = self.delta_min
                if self.error_worsened:
                    w[i] -= prev_update[i]
                prev_d[i] = 0
            else:
                prev_update[i] = -s[i] * delta[i]
                w[i] += prev_update[i]
                prev_d[i] = d[i]

    def update_node(self, node):
        if not isinstance(node, WeightedNode):
            return
        upd = False
        if node.tune_bias and 'd_bias' in node.data:
            upd = True
            d_bias = numpy.sum(node.data['d_bias'], axis=0)
            if 'prev_d_bias' not in node.data:
                node.data['prev_d_bias'] = numpy.zeros(d_bias.size)

            if 'delta_bias' not in node.data:
                node.data['delta_bias'] = (numpy.ones(node.bias.shape) *
                                           self.delta_init)

            if 'prev_bias_update' not in node.data:
                node.data['prev_bias_update'] = numpy.zeros(node.bias.shape)

            self.upd(node.bias, d_bias, node.data['prev_d_bias'],
                     node.data['delta_bias'], node.data['prev_bias_update'],
                     node.bias.size)

        if node.tune_weights and 'd_weights' in node.data:
            upd = True
            d_weights = numpy.sum(node.data['d_weights'], axis=0)
            if 'prev_d_weights' not in node.data:
                node.data['prev_d_weights'] = numpy.zeros(d_weights.size)

            if 'delta_weights' not in node.data:
                node.data['delta_weights'] = numpy.ones(node.weights.shape) *\
                                     self.delta_init

            if 'prev_weight_update' not in node.data:
                node.data['prev_weight_update'] = numpy.zeros(node.weights.shape)

            self.upd(node.weights, d_weights, node.data['prev_d_weights'],
                     node.data['delta_weights'],
                     node.data['prev_weight_update'], node.weights.size)

        if upd:
            self.updated = True
            node.notify_change()

    def update(self, root: WeightedNode, error, prev_error):
        self.error_worsened = error > prev_error
        super().update(root, error, prev_error)


class IRpropMinus(RpropBase):
    """This is the improved Rprop algorithm without weight-backtracking, also
    called iRprop\ :sup:`-` [Igel2000].

    .. [Igel2000] Christian Igel and Michael Hüsken. `Improving the Rprop
        Learning Algorithm.` Proceedings of the Second International Symposium
        on Neural Computation (NC 2000). 2000.
    """

    def upd(self, w, d, prev_d, delta, arity):
        s_prev = numpy.sign(prev_d)
        s = numpy.sign(d)
        for i in range(arity):
            if s_prev[i] * s[i] > 0:
                delta[i] *= self.eta_plus
                if delta[i] > self.delta_max:
                    delta[i] = self.delta_max
            elif s_prev[i] * s[i] < 0:
                delta[i] *= self.eta_minus
                if delta[i] < self.delta_min:
                    delta[i] = self.delta_min
                prev_d[i] = 0
            w[i] -= s[i] * delta[i]
            prev_d[i] = d[i]

    def update_node(self, node):
        if not isinstance(node, WeightedNode):
            return
        upd = False
        if node.tune_bias and 'd_bias' in node.data:
            upd = True
            d_bias = numpy.sum(node.data['d_bias'], axis=0)
            if 'prev_d_bias' not in node.data:
                node.data['prev_d_bias'] = numpy.zeros(d_bias.size)

            if 'delta_bias' not in node.data:
                node.data['delta_bias'] = (numpy.ones(node.bias.shape) *
                                           self.delta_init)

            self.upd(node.bias, d_bias, node.data['prev_d_bias'],
                     node.data['delta_bias'], node.bias.size)

        if node.tune_weights and 'd_weights' in node.data:
            upd = True
            d_weights = numpy.sum(node.data['d_weights'], axis=0)
            if 'prev_d_weights' not in node.data:
                node.data['prev_d_weights'] = numpy.zeros(d_weights.size)

            if 'delta_weights' not in node.data:
                node.data['delta_weights'] = numpy.ones(node.weights.shape) *\
                                     self.delta_init

            self.upd(node.weights, d_weights, node.data['prev_d_weights'],
                     node.data['delta_weights'], node.weights.size)

        if upd:
            self.updated = True
            node.notify_change()
