import numpy as np
import tntorch as tn


def random_format(shape):
    """
    Generate a random tensor of random format (often hybrid) with the given shape

    :param shape:
    :return: a tensor
    """

    N = len(shape)
    if np.random.randint(4) == 0:
        ranks_tucker = None
    else:
        ranks_tucker= [None]*N
        for n in sorted(np.random.choice(N, np.random.randint(N+1), replace=False)):
            ranks_tucker[n] = np.random.randint(1, 5)
    if np.random.randint(4) == 0:
        ranks_tt = None
        ranks_cp = np.random.randint(1, 5)
    elif np.random.randint(4) == 0:
        ranks_cp = None
        ranks_tt = np.random.randint(1, 5, N-1)
    else:
        ranks_tt = list(np.random.randint(1, 5, N-1))
        ranks_cp = [None]*N
        for n in sorted(np.random.choice(N, np.random.randint(N+1), replace=False)):
            if n > 0 and ranks_cp[n-1] is not None:
                r = ranks_cp[n-1]
            else:
                r = np.random.randint(1, 5)
            ranks_cp[n] = r
            if n > 0:
                ranks_tt[n-1] = None
            if n < N-1:
                ranks_tt[n] = None
    return tn.randn(shape, ranks_tt=ranks_tt, ranks_cp=ranks_cp, ranks_tucker=ranks_tucker)