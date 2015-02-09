# -*- coding: utf8 -*-
"""Contains classes which can form a BNF grammar.
"""
import evo.support.tree


class Rule(object):
    """Represents a rule in a BNF grammar.

    Contains name of the rule (i.e. the non-terminal) and a list of choices for
    this rule to expand.
    """

    def __init__(self, name):
        """Initializes the Rule object with the name of the rule and with an
        empty list of choices. This list is meant to be initialized externally.

        :param str name: Name of the rule. Names of rules must be unique inside
            a :class:`.Grammar`.
        """

        self.name = name
        self._choices = []
        self._terminal_choices = []

    def __repr__(self):
        return "Rule<%s>" % self.name

    def __str__(self):
        return "<%s>" % self.name

    def _collect_terminal_choices(self):
        if len(self._choices) == 1:
            if self._terminal_choices:
                return
            self._terminal_choices.append(0)
            return
        changed = False
        done = set(self._terminal_choices)
        for individual in range(len(self._choices)):
            if individual in done:
                continue
            terminal = True
            for symbol in self._choices[individual]:
                if isinstance(symbol, Rule):
                    if not symbol._is_terminal_reachable({self.name}):
                        terminal = False
                        break
            if terminal:
                self._terminal_choices.append(individual)
                changed = True
        if changed:
            self._terminal_choices.sort()

    def _is_terminal_reachable(self, closed):
        if self.name in closed:
            return False
        closed.add(self.name)
        reachable = False
        for choice in self._choices:
            choice_reachable = True
            for symbol in choice:
                if isinstance(symbol, Terminal):
                    continue
                # noinspection PyProtectedMember
                choice_reachable = symbol._is_terminal_reachable(closed)
                if not choice_reachable:
                    break
            if choice_reachable:
                reachable = True
                break
        closed.remove(self.name)
        return reachable

    def get_choices_num(self):
        """Returns the number of choices available for expansion of this rule.
        """
        return len(self._choices)

    def get_choices(self):
        """Returns a list of choices for this rule.

        :return: A list of possible choices for expansion of this rule.

            .. note::

                The returned list is a copy of the internal list so it can be
                modified, but the elements inside the list are not a copy.
        :rtype: list of lists of :class:`.Rule`\\ s and/or :class:`.Terminal`\\
            s
        """
        return list(self._choices)

    def get_choice(self, i):
        """Returns an i-th choice.

        The result is equivalent to calling (supposing rule is an instance of
        Rule)::

            list(rule.get_choices()[individual])

        :return: Returns a list of terms resulting by expansion of the i-th
            choice of this rule.

            .. note::

                The list is a copy of the internal list so it can be modified
                but the elements inside the list are not a copy.
        :rtype: list of :class:`.Rule`\\ s and/or :class:`.Terminal`\\ s
        """
        return list(self._choices[i])

    def get_terminal_choices_num(self):
        """Returns the number of choices for this rule that result in
        terminal-only symbols.
        """
        return len(self._terminal_choices)

    def get_terminal_choices(self):
        """Returns a list of choices for this rule that result in terminal-only
        symbols.

        :return: A list of possible choices for expansion of this rule.

            .. note::

                The returned list is a copy of the internal list so it can be
                modified, but the elements inside the list are not a copy.
        :rtype: list of lists of :class:`.Rule`\\ s and/or :class:`.Terminal`\\
            s
        """
        return [self._choices[i] for i in self._terminal_choices]

    def get_terminal_choice_index(self, i):
        """Returns the full index of the i-th terminal choice.
        """
        return self._terminal_choices[i]

    def get_terminal_choice(self, i):
        """Returns an i-th choice that results in terminal-only symbols.

        The result is equivalent to calling (supposing rule is an instance of
        Rule)::

            list(rule.get_terminal_choices()[individual])

        :return: Returns a list of terms resulting by expansion of the i-th
            choice of this rule that results in terminal-only symbols.

            .. note::

                The list is a copy of the internal list so it can be modified
                but the elements inside the list are not a copy.
        :rtype: list of :class:`.Rule`\\ s and/or :class:`.Terminal`\\ s
        """
        return list(self._choices[self._terminal_choices[i]])

    def to_dict(self):
        """Returns a dictionary representation of the rule.

        The representation is identical to an item in the "rules" dictionary
        described in :class:`.Grammar`.

        .. seealso::

            Constructor :class:`.Grammar`.
        """
        choices = []
        for c in self._choices:
            choice = []
            for term in c:
                if type(term) is Rule:
                    choice.append(("N", term.name))
                elif type(term) is Terminal:
                    choice.append(("T", term.text))
            choices.append(choice)
        return {self.name: choices}


class Terminal(object):
    """Represents a terminal in a BNF grammar."""

    def __init__(self, text):
        """Constructs the terminal with the given value.

        :param str text: The value of the terminal.
        """
        self.text = text
        """(:class:`str`) Holds the actual terminal value."""

    def __repr__(self):
        return "Terminal({0})".format(self.text)

    def __str__(self):
        return self.text

    def __eq__(self, other):
        if self is other:
            return True
        if type(other) is not type(self):
            return False
        return self.text == other.text

    def __hash__(self):
        return 31 * self.text.__hash__()


class Grammar(object):
    """Contains a simple BNF grammar for generating purposes."""

    def __init__(self, grammar):
        """Constructs a grammar object from the passed dictionary describing
        the grammar structure.

        Orphan rules (individual.e. rules that cannot be reached by expanding
        rules starting from the starting rule) are ignored.

        :param grammar: A dictionary with the structure of the rules and
            terminals. The structure of the dictionary is required to be the
            following:

            .. code-block:: python

                {
                    "start-rule" : <name of the start rule>,
                    "rules" : {
                        <name of the rule> : [
                            <first choice>,
                            <second choice>,
                            ...
                        ],
                        <name of another rule> : [
                            ...
                        ],
                        ...
                    }
                }

            where <Xth choice> is a list of tuples where each tuple has
            exactly two elements. The first one is a string "T" (meaning a
            terminal) or "N" (meaning a non-terminal). The second element
            is a string containing either a rule name (in case of
            non-terminal) or an arbitrary string (in case of terminal).

        :raise GrammarBuildingError: An error occurred during building of the
            grammar. Some of the possible causes are:

                * bad structure of the argument
                * a rule definition or the start rule references an
                  undefined rule
                * there are no terminals
        """
        self._start_rule = None
        self._rules = None
        self._rules_dict = None
        self._terminals = None
        self._terminals_dict = None

        if "start-rule" not in grammar:
            raise GrammarBuildingError("No \"start-rule\" present in input "
                                       "dictionary")

        if "rules" not in grammar:
            raise GrammarBuildingError("No \"rules\" present in input "
                                       "dictionary")

        rules_dict = {name: Rule(name) for name in grammar["rules"]}
        used_rules = set()
        used_rules.add(rules_dict[grammar["start-rule"]].name)
        used_terminals = set()
        terminals = dict()
        for _, choices in grammar["rules"].items():
            for choice in choices:
                for term in choice:
                    if len(term) == 2 and term[0] == "T":
                        terminals[term[1]] = Terminal(term[1])
        for rule_name, rule_def in grammar["rules"].items():
            choices = []
            for ci, choice in enumerate(rule_def):
                terms = []
                for ti, term in enumerate(choice):
                    if len(term) != 2 or not (term[0] == "T" or term[0] == "N"):
                        raise GrammarBuildingError('Term No. {0} in choice No.'
                                                   ' {1} in rule {2} is '
                                                   ' invalid  (the tuple is '
                                                   'not of two elements or the'
                                                   ' first element is neither '
                                                   '"T" nor "N".'.
                                                   format(ti, ci, rule_name))
                    if term[0] == "T":
                        terms.append(terminals[term[1]])
                        used_terminals.add(terminals[term[1]])
                    else:
                        if term[1] not in rules_dict:
                            raise GrammarBuildingError('Term No. {0} in choice'
                                                       ' No. {1} in rule {2} '
                                                       'references an '
                                                       'undefined rule.'.
                                                       format(ti, ci,
                                                              rule_name))
                        terms.append(rules_dict[term[1]])
                        used_rules.add(rules_dict[term[1]].name)
                choices.append(terms)
            # noinspection PyProtectedMember
            rules_dict[rule_name]._choices = choices

        self._rules = []
        self._rules_dict = {}

        used_rules = list(used_rules)
        used_rules.sort()
        for rule_name in used_rules:
            rule = rules_dict[rule_name]
            # noinspection PyProtectedMember
            rule._collect_terminal_choices()
            self._rules.append(rule)
            self._rules_dict[rule.name] = rule

        self._terminals = []
        self._terminals_dict = {}

        used_terminals = list(used_terminals)
        used_terminals.sort(key=lambda t: t.text)
        self._terminals = used_terminals
        for term in used_terminals:
            self._terminals_dict[term.text] = term

        self._start_rule = rules_dict[grammar["start-rule"]]

    def get_start_rule(self):
        """Returns the starting rule of the grammar.

        :return: The starting rule of the grammar.
        :rtype: :class:`.Rule`
        """
        return self._start_rule

    def to_dict(self):
        """Returns a dictionary representation of the grammar.

        The representation is identical to the one expected in the constructor
        :class:`.Grammar`.

        Supposing ``g`` is an instance of ``Grammar``, the following always
        holds::

            >>> Grammar(g.to_dict()).to_dict() == g.to_dict()
            True

        :return: Dictionary describing the grammar.
        :rtype: :class:`dict`
        :see: Constructor :class:`.Grammar`.
        """
        d = {}
        for rule in self._rules:
            d.update(rule.to_dict())
        return {"start-rule": self._start_rule.name, "rules": d}

    def get_rules(self):
        return self._rules

    def get_rule(self, rule_name):
        if rule_name in self._rules_dict:
            return self._rules_dict[rule_name]
        return None

    def get_terminal(self, terminal_text):
        if terminal_text in self._terminals_dict:
            return self._terminals_dict[terminal_text]
        return None

    def generate(self, decisions, mode='tree', max_wraps=0,
                 max_depth=float('inf')):
        """Generates the output from the given input decisions.

        Returns a tuple ``(output, finished, used_num, wraps)``:

        * ``output`` contains the output representation which is either in the
          form of a parse tree (if ``mode == 'tree'``) or in the form of text
          (if ``mode == 'text'``).
        * ``finished`` is ``True`` if and only if all rules were expanded down
          to terminals, individual.e. it is ``False`` if the mapping was ended
          prematurely and some rules were left unexpanded
        * ``used_num`` contains the number of elements from ``decisions`` that
          were used for making expansion decision (equivalently it is the
          number of how many times the ``decisions.__iter__()``'s
          ``__next__()`` was called
        * ``wraps`` indicates how many times the decisions wrapped from end to
          start

        :param decisions: an iterable containing numbers to make decisions on
            multi-choice production rules.

            The only requirement is that the numbers are positive integers.
        :param str mode: defines the type of ``output``; can be either
            ``'tree'`` (default) or ``'text'``

            If equal to ``tree`` :meth:`.to_tree` is effectively called, if
            equal to ``text`` :meth:`.to_text` is effectively called.
        :param int max_wraps: how many times is the decisions allowed to be
            reused if its end is reached without having completed the expansion

            Default value is ``0``.
        :param max_depth: the maximum depth of the derivation tree allowed
            during generation of the output. Applicable only for
            ``mode='tree'``.

            If this depth is reached, the decisions will be made only on those
            choices that result in terminal-only symbols. If there are no such
            choices for a particular rule, normal decision is made and the
            terminas will be selected as soon as possible.

            Default value is infinity.
        :return: a 4-tuple as described above
        :see: :meth:`.to_tree`\\ , :meth:`.to_text`
        """
        if mode == 'tree':
            return self.to_tree(decisions, max_wraps, max_depth)
        elif mode == 'text':
            return self.to_text(decisions, max_wraps)

    def to_tree(self, decisions, max_wraps, max_depth, sequence=None):
        """From the given input decisions generates an output in the form of a
        derivation tree.

        The derivation tree is represented by a tree formed by
        :class:`evo.support.tree.TreeNode`s where the
        :attr:`evo.support.tree.TreeNode.data` attribute's value is set to the
        name of the corresponding rule for inner nodes and for leaves the
        attribute has the value of the corresponding terminal.

        Returns a tuple ``(output, finished, used_num, wraps)``:

        * ``output`` contains the output representation which is either in the
          form of a parse tree (if ``mode == 'tree'``) or in the form of text
          (if ``mode == 'text'``).
        * ``finished`` is ``True`` if and only if all rules were expanded down
          to terminals, individual.e. it is ``False`` if the mapping was ended
          prematurely and some rules were left unexpanded
        * ``used_num`` contains the number of elements from ``decisions`` that
          were used for making expansion decision (equivalently it is the
          number of how many times the ``decisions.__iter__()``'s
          ``__next__()`` was called
        * ``wraps`` indicates how many times the decisions wrapped from end to
          start

        :param decisions: an iterable containing numbers to make decisions on
            multi-choice production rules.

            The only requirement is that the numbers are positive integers.
        :param int max_wraps: how many times is the decisions is allowed to be
            re-used if its end is reached without having the expansion
            completed
        :param max_depth: the maximum depth of the derivation tree allowed
            during generation of the output.

            If this depth is reached, the decisions will be made only on those
            choices that result in terminal-only symbols. If there are no such
            choices for a particular rule, normal decision is made and the
            terminas will be selected as soon as possible.
        :param sequence: if not ``None`` then each generated decision (before
            taking the modulo value) will be appended (using the ``append()``
            method) to this sequence
        :return: a 4-tuple as described above
        """
        tree = None
        tree_stack = None
        rules_stack = [self.get_start_rule()]
        it = decisions.__iter__()
        n = 0
        wraps = 0
        depth = None
        while rules_stack and wraps <= max_wraps:
            rule = rules_stack.pop(0)
            if isinstance(rule, Rule):
                if tree is None:
                    node = evo.support.tree.TreeNode(None, None, [], rule.name)
                    tree = node
                    tree_stack = [[node, 0]]
                    depth = 1
                else:
                    last_node = tree_stack[-1][0]
                    idx = tree_stack[-1][1]

                    node = last_node.children[idx]

                    tree_stack.append([node, 0])
                    depth += 1

                deep = (depth >= max_depth and
                        rule.get_terminal_choices_num() > 0)
                choices_num = rule.get_choices_num()
                if choices_num == 1:
                    choice = rule.get_choice(0)
                else:
                    try:
                        decision = it.__next__()
                    except StopIteration:
                        if wraps >= max_wraps:
                            break
                        wraps += 1
                        it = decisions.__iter__()
                        decision = it.__next__()
                    n += 1
                    if deep:
                        choices_num = rule.get_terminal_choices_num()
                    choice_idx = decision % choices_num
                    if deep:
                        choice_idx = rule.get_terminal_choice_index(choice_idx)
                    choice = rule.get_choice(choice_idx)
                    if sequence is not None:
                        sequence.append(decision)
                rules_stack = choice + rules_stack

                node.children = []
                for individual in range(len(choice)):
                    c = choice[individual]
                    if isinstance(c, Rule):
                        node.children.append(evo.support.tree.TreeNode(
                            node, individual, [], c.name))
                    else:
                        assert isinstance(c, Terminal)
                        node.children.append(evo.support.tree.TreeNode(
                            node, individual, None, c.text))
            else:
                assert isinstance(rule, Terminal)
                tree_stack[-1][1] += 1

                while (tree_stack and
                       tree_stack[-1][1] == len(tree_stack[-1][0].children)):
                    tree_stack.pop()
                    depth -= 1
                    if tree_stack:
                        tree_stack[-1][1] += 1

        # empty tree_stack <=> empty rules_stack
        return tree, len(tree_stack) == 0, n, wraps

    def to_text(self, decisions, max_wraps):
        """From the given input decisions generates an output in the form of a
        text.

        The output is in the form of a text defined by the grammar and the
        codon.

        Returns a tuple ``(output, finished, used_num, wraps)``:

        * ``output`` contains the output representation which is either in the
          form of a parse tree (if ``mode == 'tree'``) or in the form of text
          (if ``mode == 'text'``).
        * ``finished`` is ``True`` if and only if all rules were expanded down
          to terminals, individual.e. it is ``False`` if the mapping was ended
          prematurely and some rules were left unexpanded
        * ``used_num`` contains the number of elements from ``decisions`` that
          were used for making expansion decision (equivalently it is the
          number of how many times the ``decisions.__iter__()``'s
          ``__next__()`` was called
        * ``wraps`` indicates how many times the decisions wrapped from end to
          start

        :param decisions: an iterable containing numbers to make decisions on
            multi-choice production rules.

            The only requirement is that the numbers are positive integers.
        :param int max_wraps: how many times is the decisions is allowed to be
            re-used if its end is reached without having the expansion
            completed
        :return: a 4-tuple as described above
        """
        text = []
        rules_stack = [self.get_start_rule()]
        it = decisions.__iter__()
        n = 0
        wraps = 0
        while rules_stack and wraps <= max_wraps:
            rule = rules_stack.pop(0)
            if isinstance(rule, Rule):
                if rule.get_choices_num() == 1:
                    choice = rule.get_choice(0)
                else:
                    try:
                        decision = it.__next__()
                    except StopIteration:
                        if wraps >= max_wraps:
                            break
                        wraps += 1
                        it = decisions.__iter__()
                        decision = it.__next__()
                    n += 1
                    choice_idx = decision % rule.get_choices_num()
                    choice = rule.get_choice(choice_idx)
                rules_stack = choice + rules_stack
            else:
                assert isinstance(rule, Terminal)
                text.append(rule.text)

        return ''.join(text), len(rules_stack) == 0, n, wraps

    def derivation_tree_to_choice_sequence(self, tree_root):
        """Converts the given derivation tree on this grammar to a list of
        numbers representing the sequence of choices made at each point.

        Choices that were the only possible choices (i.e. the corresponding
        non-terminal symbol has only one possible expansion) are omitted.

        The produced sequence of choices with respect to the ``tree_root``
        always behaves in the following way::

            >>>g = ... # some Grammar
            >>>choices = ... # sequence of choices producing a valid derivation
            ...              # tree on ``g`` with no extra integers left
            >>>(tree, _, _, _) = g.to_tree(decisions=choices, max_wraps=0,
            ...                                      max_depth=float('inf'))
            >>>choices2 , _= g.derivation_tree_to_choice_squence(tree)
            >>>choices == choices2
            True

        The second returned value is a list of numbers of the same length as
        the list of choices. The numbers indicate the maximum number of choices
        available at the same choice points as in the list of choices. Because
        single choice points are omitted, this list contains only numbers
        greater than 1.

        :param tree_root: the root node of the derivation tree
        :type tree_root: :class:`evo.support.tree.TreeNode`
        """
        choices = []
        max_choices = []

        def process(node):
            if node.is_leaf():
                return
            rule = self.get_rule(node.data)
            assert rule is not None
            i = None
            for i in range(rule.get_choices_num()):
                choice = rule.get_choice(i)
                if len(choice) != len(node.children):
                    continue
                fit = True
                for child, term in zip(node.children, choice):
                    if isinstance(term, Rule):
                        data = term.name
                    elif isinstance(term, Terminal):
                        data = term.text
                    else:
                        assert False

                    if child.data != data:
                        fit = False
                        break
                if fit:
                    break
            assert i is not None
            choices.append(i)
            max_choices.append(rule.get_choices_num())

        tree_root.preorder(process)

        ch = []
        max_ch = []
        for c, mc in zip(choices, max_choices):
            if mc > 1:
                ch.append(c)
                max_ch.append(mc)
        return ch, max_ch


class GrammarBuildingError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


def derivation_tree_to_text(root):
    """Takes a derivation tree and returns the corresponding text.

    The derivation tree is expected to be a :class:`evo.support.tree.TreeNode`
    where the inner nodes represent the rules and their
    :attr:`evo.support.tree.TreeNode.data` attribute is set to the name of the rule
    and the leaves represent the terminals and their
    :attr:`evo.support.tree.TreeNode.data` attribute is set to the value of the
    terminal.
    """
    assert isinstance(root, evo.support.tree.TreeNode)
    terminals = []

    def extractor(node):
        if node.is_leaf():
            terminals.append(node.data)
    root.preorder(extractor)
    return "".join(terminals)