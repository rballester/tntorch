import torch
import tntorch as tn
import numpy as np


def true(N):
    """
    Create a formula (N-dimensional tensor) that is always true.

    :param N: an integer

    :return: a :math:`2^N` :class:`Tensor`
    """

    return tn.Tensor([torch.ones([1, 2, 1]) for n in range(N)])


def false(N):
    """
    Create a formula (N-dimensional tensor) that is always false.

    :param N: an integer

    :return: a :math:`2^N` :class:`Tensor`
    """

    return tn.Tensor([torch.zeros([1, 2, 1]) for n in range(N)])


def all(N, which=None):
    """
    Create a formula (N-dimensional tensor) that is satisfied iff all symbols are true.

    :param N: an integer
    :param which: list of integers to consider (default: all)

    :return: a :math:`2^N` :class:`Tensor`
    """

    if which is None:
        which = list(range(N))

    cores = []
    for n in range(N):
        if n in which:
            cores.append(torch.cat([torch.zeros(1, 1, 1), torch.ones(1, 1, 1)], dim=1))
        else:
            cores.append(torch.ones(1, 2, 1))
    return tn.Tensor(cores)


def none(N, which=None):
    """
    Create a formula (N-dimensional tensor) that is satisfied iff all symbols are false.

    :param N: an integer
    :param which: list of integers to consider (default: all)

    :return: a :math:`2^N` :class:`Tensor`
    """

    if which is None:
        which = list(range(N))

    cores = []
    for n in range(N):
        if n in which:
            cores.append(torch.cat([torch.ones(1, 1, 1), torch.zeros(1, 1, 1)], dim=1))
        else:
            cores.append(torch.ones(1, 2, 1))
    return tn.Tensor(cores)


def any(N, which=None):
    """
    Create a formula (N-dimensional tensor) that is satisfied iff at least one symbol is true.

    :param N: an integer
    :param which: list of integers to consider (default: all)

    :return: a :math:`2^N` :class:`Tensor`
    """

    return ~none(N, which)


def one(N, which=None):
    """
    Create a formula (N-dimensional tensor) that is satisfied iff one and only one input is true.

    Also known as "n-ary exclusive or".

    :param N: an integer
    :param which: list of integers to consider (default: all)

    :return: a :math:`2^N` :class:`Tensor`
    """

    if which is None:
        return tn.automata.weight_mask(N, 1)
    else:
        return tn.automata.weight_mask(N, 1) & tn.any(N, which)


def symbols(N):
    """
    Generate N Boolean symbols (each represented as an N-dimensional tensor).

    :param N: an integer

    :return: a list of N :math:`2^N` :class:`Tensor`
    """

    return [presence(N, n) for n in range(N)]


def relevant_symbols(t):
    """
    Finds all variables whose values affect the formula's output in at least one case.

    :param t: a :math:`2^N` :class:`Tensor`

    :return: a list of integers
    """

    cores = [torch.cat((c[:, 1:2, :]-c[:, 0:1, :], c), dim=1) for c in t.cores]
    t2 = tn.Tensor(cores)
    return [n for n in range(t.dim()) if tn.norm(t2[[slice(1, 3)]*n + [0] + [slice(1, 3)]*(t.dim()-n-1)]) > 1e-10]


def irrelevant_symbols(t):
    """
    Finds all variables whose values never affect the formula's output.

    :param t: a :math:`2^N` :class:`Tensor`

    :return: a list of integers
    """

    rel = relevant_symbols(t)
    return [n for n in range(t.dim()) if n not in rel]


def only(t):
    """
    Forces all irrelevant symbols to be zero.

    :Example:

    >>> x, y = tn.symbols(2)
    >>> tn.sum(x)  # Result: 2 (x = True, y = False, and x = True, y = True)
    >>> tn.sum(tn.only(x))  # Result: 1 (x = True, y = False)

    :param: a :math:`2^N` :class:`Tensor`

    :return: a masked :class:`Tensor`
    """

    return tn.mask(t, absence(t.dim(), irrelevant_symbols(t)))


def presence(N, which):
    """
    True iff all symbols in `which` are present.

    :param N: int
    :param which: a list of ints

    :return: a masked :class:`Tensor`
    """

    which = np.atleast_1d(which)
    cores = [torch.ones([1, 2, 1]) for n in range(N)]
    for w in which:
        cores[w][0, 0, 0] = 0
    return tn.Tensor(cores)


def absence(N, which):
    """
    True iff all symbols in `which` are absent.

    :param N: int
    :param which: a list of ints

    :return: a masked :class:`Tensor`
    """

    which = np.atleast_1d(which)
    cores = [torch.ones([1, 2, 1]) for n in range(N)]
    for w in which:
        cores[w][0, 1, 0] = 0
    return tn.Tensor(cores)


def is_tautology(t):
    """
    Checks if a formula is always satisfied.

    :param t: a :math:`2^N` :class:`Tensor`

    :return: True if `t` is a tautology; False otherwise
    """

    return bool(tn.norm(~t) <= 1e-6)


def is_contradiction(t):
    """
    Checks if a formula is never satisfied.

    :param t: a :math:`2^N` tensor

    :return: True if `t` is a contradiction; False otherwise
    """

    return bool(tn.norm(t) <= 1e-6)


def is_satisfiable(t):
    """
    Checks if a formula can be satisfied.

    :param t: a :math:`2^N` :class:`Tensor`

    :return: True if `t` is satisfiable; False otherwise
    """

    return bool(tn.sum(t) >= 1e-6)


def implies(t1, t2):
    """
    Checks if a formula implies another one (i.e. is a sufficient condition).

    :param t1: a :math:`2^N` :class:`Tensor`
    :param t2: a :math:`2^N` :class:`Tensor`

    :return: True if `t1` implies `t2`; False otherwise
    """

    return bool(is_contradiction(t1 & ~t2))


def equiv(t1, t2):
    """
    Checks if two formulas are logically equivalent.

    :param t1: a :math:`2^N` :class:`Tensor`
    :param t2: a :math:`2^N` :class:`Tensor`

    :return: True if `t1` implies `t2` and vice versa; False otherwise
    """

    return implies(t1, t2) & implies(t2, t1)
