Prerequisites
=============

* Python 3.4+ - You can download from https://www.python.org/downloads/
  or, using Anaconda/Miniconda, from https://www.continuum.io/downloads.
* NumPy - already contained in anaconda, in miniconda run
  ``conda install numpy``
* PyYAML - already contained in anaconda, in miniconda run
  ``conda install PyYAML``

  * You can also install both prerequisites in single command using
    ``conda install numpy PyYAML``.

Installation
============

1. Download the whole package (its root directory).
2. Extract it to arbitrary directory.
3. In the extracted package, find the file ``setup.py`` and note in which
   directory is it located. Let's call this directory ``DIR``.
4. Run ``pip install DIR``.

   * **NOTE**: the prerequisites (see above) should be installed before running
     this command.

5. Done, you can even delete the downloaded package (it is now
   installed in the conda environment).
   Run with ``evo [arguments]`` (or "manually" with python by
   ``python -m evo [arguments]`` if needed).

If you have git access to the repository, you can replace the steps 1 to 4
with::

    pip install -e git+git@<repository-address>/evo.git#egg=evo

Alternative approach (no install)
---------------------------------

You can use the package "as is" without any installation:

#. Download the whole package (its root directory).
#. Copy its contents (should contain directory evo, files setup.py, README.rst,
   DESCRIPTION.rst and other files) to arbitrary location (here called
   ``DIR``).
#. Done. To run, ``cd`` to ``DIR`` and from there run
   ``python -m evo [arguments]``.

How to upgrade
--------------

If you used the no-install approach, you can simply replace the old version with
a new one.

If you installed properly, you can use one of the two ways:

1. Uninstall (see below) and install again.
2. Upgrade directly - follow the same steps as for installation but insert these
   arguments right after the ``install`` command:
   ``--upgrade --upgrade-strategy only-if-needed``.

How to uninstall
----------------

To uninstall the package, run ``pip uninstall evo``.
This works for both installation methods.
However, for the no install method this may give some errors but it seems they
are safe to ignore.

Usage
=====

The executable thing is the ``evo`` module.
To run it, enter ``python -m evo`` to the command line.
Usage is following::

    python -m evo [common options] algorithm [algorithm options]

All algorithms have a ``-h``/``--help`` options to show the reference of their
options.

Common options
--------------

``-h`` or ``--help``
    prints the basic usage and exits

``--version``
    prints version and exits (quite useless now)

``--logconf CONFFILE``
    sets the logging configuration file in yaml format (optional, there is a
    default logging set up)

Algorithms
----------

``bpgp`` - BackPropagation Genetic Programming
    MGGP-like algorithm with nodes that apply additive and multiplicative
    constants at their inputs, use linear combinations of features (LCF) as
    leaf nodes, etc.

``bpgp`` algorithm options
--------------------------

All options are optional unless stated by **REQUIRED**.

``-h`` or ``--help``
    prints help and exits

``--training-data file[:x-columns:y-column]``
    **REQUIRED** Specification of the training data in the format
    ``file[:x-columns:y-column]``.

    ``file`` is path to a CSV file to be loaded.

    ``x-columns`` is a comma-separated (only comma, without whitespaces) list
    of numbers of columns (zero-based) to be used as the features.

    ``y-column`` is a number of column that will be used as the target. You can
    use negative numbers to count from back, i.e. -1 is the last column, -2 is
    the second to the last column, etc.

    The bracketed part is optional and can be left out. If it is left out, all
    columns except the last one are used as features and the last one is used as
    target.

``--testing-data file[:x-columns:y-column]``
    Specification of the testing data. The format is identical to the one of
    ``--training-data``. The testing data must have the same number of columns
    as ``--training-data``.

    The testing data are evaluated with the best individual after the evolution
    finishes. If, for some reason, the individual fails to evaluate (e.g. due to
    numerical errors like division by zero), the second best individual is tried
    and so forth until the individual evaluates or there is no individual left.
    If the testing data is not specified, no testing is done (only training
    measures are reported).

``--delimiter DELIMITER``
    Field delimiter of the CSV files specified in ``--training-data`` and
    ``--testing-data``.

    Default is ``,`` (comma).

``-o OUTPUT_DIRECTORY`` or ``--output-directory OUTPUT_DIRECTORY``
    Directory to which the output files will be written.

    Default is current directory.

``--m-fun M_FUN``
    Name of the matlab function the model will be written to (without
    extension).

    Default is ``func``.

``--output-string-template OUTPUT_STRING_TEMPLATE```
    Template for the string that will be printed to the standard output at the
    very end of the algorithm. This can be used to report algorithm performance
    to tuners such as SMAC. Default is no string (nothing is printed).

    The string can contain any of the following placeholders: ``{tst_r2}``,
    ``{trn_r2}``, ``{tst_mse}``, ``{trn_mse}``, ``{tst_mae}``, ``{trn_mae}``,
    ``{runtime}``, ``{seed}``, ``{iterations}``.

``--seed n``
    Seed for random number generator.

    If not specified, current time will be used.

``--generations GENERATIONS``
    The maximum number of generations to run for.

    Default is infinity (i.e. until stopped externally or with some other
    stopping condition).

``--time TIME``
    The maximum number of seconds to run for.

    Default is infinity (i.e. until stopped externally or with some other
    stopping condition).

``--generation-time-combinator {any,all}``
    If both ``--generations`` and ``--time`` are specified, this determines how
    are the two conditions combined.

    The value of ``any`` causes termination when any of the two conditions is
    met.

    The value of ``all`` causes termination only after both conditions are met.

    Default is ``any``.

``--pop-size POP_SIZE``
    Population size.

    Default is 100.

``--elitism ELITISM``
    Number of elites as a fraction (float between 0 and 1) of the population
    size.

    Default is 0.15.

``--tournament-size TOURNAMENT_SIZE``
    Number of individuals competing in a tournament selection as a fraction
    (float between 0 and 1) of the population size.

    Default is 0.1.

``--max-genes MAX_GENES``
    Maximum number of genes.

    Default is 4.

``--max-depth MAX_DEPTH``
    Maximum depth of a gene.

    Default is 5.

``--max-nodes MAX_NODES``
    Maximum number of nodes in a gene.

    Default is infinity (i.e. unbounded).

``--functions Function[,Function ...]``
    A comma-separated (without whitespaces) list of functions available to the
    algorithm.

    Available functions are:

    * ``Add2`` - binary addition
    * ``Sub2`` - binary subtraction
    * ``Mul2`` - binary multiplication
    * ``Div2`` - binary division
    * ``Sin`` - sine
    * ``Cos`` - cosine
    * ``Exp`` - e^x
    * ``Abs`` - absolute value
    * ``Sqrt`` - square root (unprotected, i.e. negative argument wil produce an
      error and the individual will be assigned the worst fitness)
    * ``Sigmoid`` - 1 / (1 + e^-x)`
    * ``Tanh`` - hyperbolic tangent
    * ``Sinc`` - sin x / x (defined as 1 at x = 0)
    * ``Softplus`` - ln(1 + e^x)
    * ``Gauss`` - e^-(x^2)
    * ``BentIdentity`` - 1 / 2 * (sqrt(x + 1) - 1) + x
    * ``Pow(n)`` - x^n (``n`` must be positive integer)

    Default is ``Add2,Sub2,Mul2,Sin,Cos,Exp,Sigmoid,Tanh,Sinc,Softplus,Gauss,Pow(2),Pow(3),Pow(4),Pow(5),Pow(6)``.

``--crossover-prob CROSSOVER_PROB``
    Probability of crossover.

    Default is 0.84

``--highlevel-crossover-prob HIGHLEVEL_CROSSOVER_PROB``
    Probability of choosing a high-level crossover as a crossover operation.

    The complement to 1 is then the probability of subtree crossover. If
    ``--max-genes`` is 1, this parameter is ignored (even if not specified) and
    set to 0.

    Default is 0.2.

``--highlevel-crossover-rate HIGHLEVEL_CROSSOVER_RATE``
    Probability that a gene is chosen for crossover in high-level crossover.

    Default is 0.5.

``--mutation-prob MUTATION_PROB``
    Probability of mutation.

    Default is 0.14.

``--constant-mutation-prob CONSTANT_MUTATION_PROB``
    Probability of choosing mutation of constants as a mutation operation.

    The complement to 1 of this parameter and of ``--weights-muatation-prob`` is
    then the probability of subtree mutation. To turn this mutation off, set the
    parameter to 0.

    Default is 0.05.

``--constant-mutation-sigma CONSTANT_MUTATION_SIGMA``
    Standard deviation of the normal distribution used to mutate the constant
    values.

    Default is 0.1.

``--weights-mutation-prob WEIGHTS_MUTATION_PROB``
    Probability of choosing mutation of weights as a mutation operation.

    The complement to 1 of this parameter and of ``--constant-muatation-prob``
    is then the probability of subtree mutation. To turn this mutation off, set
    the parameter to 0.

    Default is 0.05.

``--weights-mutation-sigma WEIGHTS_MUTATION_SIGMA``
    Standard deviation of the normal distribution used to mutate the weights.

    Default is 3.

``--backpropagation-mode {none,raw,nodes,depth}``
    How is backpropagation used.

    Mode ``none`` turns the backpropagation off completely.

    Mode ``raw`` means that the number of steps is always the number specified
    in ``--backpropagation-steps`` (and hence ``--min-backpropagation-steps`` is
    ignored).

    Modes ``nodes`` and ``depth`` mean that the number of steps is the number
    specified in ``--backpropagation-steps`` minus the total number of nodes of
    the individual (for ``nodes``) or the maximum depth of the genes (for
    ``depth``).

    Default is ``none``, i.e. no backpropagation.

``--backpropagation-steps BACKPROPAGATION_STEPS``
    How many backpropagation steps are performed per evaluation.

    The actual number is computed based on the value of
    ``--backpropagation-mode``.

    Default is 25.

``--min-backpropagation-steps MIN_BACKPROPAGATION_STEPS``
    At least this number of backpropagation steps is always performed, no matter
    what ``--backpropagation-steps`` and ``--backpropagation-mode` are set to
    (except for ``none`` mode).

    Default is 2.

``--weighted``
    If specified, the inner nodes will be weighted, i.e. with multiplicative and
    additive weights, tunable by backpropagation and weights mutation.

``--lcf-mode {none,unsynced,synced,global}``
    How the LCFs are used.

    Mode ``none`` turns the LCFs off completely.

    Mode ``unsynced`` means that each LCF is free to change on its own (by
    backpropagation and/or mutation).

    Mode ``synced`` means that the LCFs are synchronized across the individual.

    Mode ``global`` means that the LCFs are synchronized across the whole
    population.

    Default is ``none``, i.e. no LCFs.

``--weight-init {latent,random}``
    How are weights in weighted nodes and LCFs (if they are turned on)
    initialized.

    Mode ``latent`` means that the initial values of weights are such that they
    play no role, i.e. additive weights set to zero, multiplicative weights set
    to one (or only one of them in case of LCFs).

    Mode ``random`` means that the values of weights are chosen randomly (see
    option ``--random-init-bounds``).

    Default is ``latent``.

``--weight-init-bounds lb ub``
    Bounds of the range the weights are sampled from when ``--weight-init`` is
    set to ``random``.

    Default is -10 and 10.

``--const-init-bounds lb ub``
    Bounds of the range the constants (leaf nodes) are sampled from.

    Default is -10 and 10.
