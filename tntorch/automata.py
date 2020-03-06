import torch
import tntorch as tn


def weight_mask(N, weight, nsymbols=2):
    """
    Accepts a string iff its number of 1's equals (or is in) `weight`

    :param N: number of dimensions
    :param weight: an integer (or list thereof): recognized weight(s)
    :param nsymbols: slices per core (default is 2)

    :return: a mask tensor
    """

    if not hasattr(weight, '__len__'):
        weight = [weight]
    weight = torch.tensor(weight).long()
    assert weight[0] >= 0
    t = tn.weight_one_hot(N, int(max(weight) + 1), nsymbols)
    t.cores[-1] = torch.sum(t.cores[-1][:, :, weight], dim=2, keepdim=True)
    return t


def weight_one_hot(N, r=None, nsymbols=2):
    """
    Given a string with :math:`k` 1's, it produces a vector that represents :math:`k` in `one hot encoding <https://en.wikipedia.org/wiki/One-hot>`_

    :param N: number of dimensions
    :param r:
    :param nsymbols:

    :return: a vector of N zeros, except its :math:`k`-th element which is a 1
    """

    if not hasattr(nsymbols, '__len__'):
        nsymbols = [nsymbols]*N
    assert len(nsymbols) == N
    if r is None:
        r = N+1

    cores = []
    for n in range(N):
        core = torch.zeros([r, nsymbols[n], r])
        core[:, 0, :] = torch.eye(r)
        for s in range(1, nsymbols[n]):
            core[:, s, s:] = torch.eye(r)[:, :-s]
        cores.append(core)
    cores[0] = cores[0][0:1, :, :]
    return tn.Tensor(cores)


def weight(N, nsymbols=2):
    """
    For any string, counts how many 1's it has

    :param N: number of dimensions
    :param nsymbols: slices per core (default is 2)

    :return: a mask tensor
    """

    cores = []
    for n in range(N):
        core = torch.eye(2)[:, None, :].repeat(1, nsymbols, 1)
        core[1, :, 0] = torch.arange(nsymbols)
        cores.append(core)
    cores[0] = cores[0][1:2, :, :]
    cores[-1] = cores[-1][:, :, 0:1]
    return tn.Tensor(cores)


def length(N):  # TODO
    """
    :todo:

    :param N:
    :return:
    """
    raise NotImplementedError


def accepted_inputs(t):
    """
    Returns all strings accepted by an automaton, in alphabetical order.

    Note: each string s will appear as many times as the value t[s]

    :param t: a :class:`Tensor`

    :return Xs: a Torch matrix, each row is one string
    """

    def recursion(Xs, left, rights, bound, mu):
        if mu == t.dim():
            return

        if t.batch:
            fiber = torch.einsum('bijk,bk->bij', (t.cores[mu], rights[mu + 1]))
        else:
            fiber = torch.einsum('ijk,k->ij', (t.cores[mu], rights[mu + 1]))

        per_point = torch.matmul(left, fiber).round()

        if t.batch:
            c = torch.cat((torch.tensor([0], dtype=per_point.dtype), per_point.cumsum(dim=1))).long()
        else:
            c = torch.cat((torch.tensor([0], dtype=per_point.dtype), per_point.cumsum(dim=0))).long()

        for i, p in enumerate(per_point):
            if c[i] == c[i+1]:  # Improductive prefix, don't go further
                continue
            Xs[bound+c[i]:bound+c[i+1], mu] = i
            recursion(Xs, torch.matmul(left, t.cores[mu][..., i, :]), rights, bound + c[i], mu+1)

    Xs = torch.zeros([round(tn.sum(t).item()), t.dim()], dtype=torch.long)
    rights = [torch.ones(1)]  # Precomputed right-product chains
    for core in t.cores[::-1]:
        rights.append(torch.matmul(torch.sum(core, dim=1), rights[-1]))
    rights = rights[::-1]
    recursion(Xs, torch.ones(1), rights, 0, 0)
    return Xs
