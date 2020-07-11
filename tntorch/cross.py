import tntorch as tn
import torch
import sys
import time
import numpy as np
import logging


def minimum(tensors=None, function=lambda x: x, rmax=10, max_iter=10, verbose=False, batch=False, **kwargs):
    """
    Estimate the minimal element of a tensor (or a function of one or several tensors)

    :param t: input :class:`Tensor`
    :param rmax: used for :func:`cross.cross()`. Lower is faster; higher is more accurate (default is 10)
    :param max_iter: used for :func:`cross.cross()`. Lower is faster; higher is more accurate (default is 10)
    :param verbose: default is False
    :param batch: Boolean
    :param **kwargs: passed to :func:`cross.cross()`

    :return: a scalar
    """

    _, info = cross(**kwargs, tensors=tensors, function=function, rmax=rmax, max_iter=max_iter, verbose=verbose, return_info=True, _minimize=True, batch=batch)
    return info['min']
    # return t[argmin(rmax=rmax, max_iter=max_iter, verbose=verbose, **kwargs)]


def argmin(tensors=None, function=lambda x: x, rmax=10, max_iter=10, verbose=False, batch=False, **kwargs):
    """
    Estimate the minimizer of a tensor (position where its minimum is located).

    For arguments, see :func:`cross.minimum()`

    :return: a tuple
    """

    _, info = cross(**kwargs, tensors=tensors, function=function, rmax=rmax, max_iter=max_iter, verbose=verbose, return_info=True, _minimize=True)
    return info['argmin']


def maximum(tensors=None, function=lambda x: x, rmax=10, max_iter=10, verbose=False, batch=False, **kwargs):
    """
    Estimate the maximal element of a tensor.

    For arguments, see :func:`cross.minimum()`

    :return: a scalar
    """

    _, info = cross(**kwargs, function=lambda *x: -function(*x), tensors=tensors, rmax=rmax, max_iter=max_iter, verbose=verbose, return_info=True, _minimize=True)
    return -info['min']


def argmax(tensors=None, function=lambda x: x, rmax=10, max_iter=10, verbose=False, batch=False, **kwargs):
    """
    Estimate the maximizer of a tensor (position where its maximum is located).

    For arguments, see :func:`cross.minimum()`

    :return: a tuple
    """

    _, info = cross(**kwargs, tensors=tensors, function=lambda *x: -function(*x), rmax=rmax, max_iter=max_iter, verbose=verbose, return_info=True, _minimize=True)
    return info['argmin']


def cross(function=lambda x: x, domain=None, tensors=None, function_arg='vectors', ranks_tt=None, kickrank=3, rmax=100, eps=1e-6, max_iter=25, val_size=1000, verbose=True, return_info=False, record_samples=False, _minimize=False, device=None, batch=False, suppress_warnings=False, detach_evaluations=False):
    """
    Cross-approximation routine that samples a black-box function and returns an N-dimensional tensor train approximating it. It accepts either:

    - A domain (tensor product of :math:`N` given arrays) and a function :math:`\\mathbb{R}^N \\to \\mathbb{R}`
    - A list of :math:`K` tensors of dimension :math:`N` and equal shape and a function :math:`\\mathbb{R}^K \\to \\mathbb{R}`

    :Examples:

    >>> tn.cross(function=lambda x: x**2, tensors=[t])  # Compute the element-wise square of `t` using 5 TT-ranks

    >>> domain = [torch.linspace(-1, 1, 32)]*5
    >>> tn.cross(function=lambda x, y, z, t, w: x**2 + y*z + torch.cos(t + w), domain=domain)  # Approximate a function over the rectangle :math:`[-1, 1]^5`

    >>> tn.cross(function=lambda x: torch.sum(x**2, dim=1), domain=domain, function_arg='matrix')  # An example where the function accepts a matrix

    References:

    - I. Oseledets, E. Tyrtyshnikov: `"TT-cross Approximation for Multidimensional Arrays" (2009) <http://www.mat.uniroma2.it/~tvmsscho/papers/Tyrtyshnikov5.pdf>`_
    - D. Savostyanov, I. Oseledets: `"Fast Adaptive Interpolation of Multi-dimensional Arrays in Tensor Train Format" (2011) <https://ieeexplore.ieee.org/document/6076873>`_
    - S. Dolgov, R. Scheichl: `"A Hybrid Alternating Least Squares - TT Cross Algorithm for Parametric PDEs" (2018) <https://arxiv.org/pdf/1707.04562.pdf>`_
    - A. Mikhalev's `maxvolpy package <https://bitbucket.org/muxas/maxvolpy>`_
    - I. Oseledets (and others)'s `ttpy package <https://github.com/oseledets/ttpy>`_

    :param function: should produce a vector of :math:`P` elements. Accepts either :math:`N` comma-separated vectors, or a matrix (see `function_arg`)
    :param domain: a list of :math:`N` vectors (incompatible with `tensors`)
    :param tensors: a :class:`Tensor` or list thereof (incompatible with `domain`)
    :param function_arg: if 'vectors', `function` accepts :math:`N` vectors of length :math:`P` each. If 'matrix', a matrix of shape :math:`P \\times N`.
    :param ranks_tt: int or list of :math:`N-1` ints. If None, will be determined adaptively
    :param kickrank: when adaptively found, ranks will be increased by this amount after every iteration (full sweep left-to-right and right-to-left)
    :param rmax: this rank will not be surpassed
    :param eps: the procedure will stop after this validation error is met (as measured after each iteration)
    :param max_iter: int
    :param val_size: size of the validation set
    :param verbose: default is True
    :param return_info: if True, will also return a dictionary with informative metrics about the algorithm's outcome
    :param device: PyTorch device
    :param batch: Boolean
    :param suppress_warnings: Boolean, if True, will hide the message about insufficient accuracy
    :param detach_evaluations: Boolean, if True, will remove gradient buffers for the function

    :return: an N-dimensional TT :class:`Tensor` (if `return_info`=True, also a dictionary)
    """
    if device is None and tensors is not None: 
        if type(tensors) == list:
            device = tensors[0].cores[0].device
        else:
            device = tensors.cores[0].device
            
    if verbose:
        print('cross device is', device)

    try:
        import maxvolpy.maxvol
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Functions that require cross-approximation require the optional maxvolpy package, which can be installed by 'pip install maxvolpy'. More info is available at https://bitbucket.org/muxas/maxvolpy")

    assert domain is not None or tensors is not None
    assert function_arg in ('vectors', 'matrix')
    if function_arg == 'matrix':
        def f(*args):
            return function(torch.cat([arg[:, None] for arg in args], dim=1))
    else:
        f = function

    if detach_evaluations:
        def build_function_wrapper(func):
            def g(*args):
                res = func(*args)
                if hasattr(res, '__len__') and not isinstance(res, torch.Tensor):
                    for i in range(len(res)):
                        if isinstance(res[i], torch.Tensor):
                            res[i] = res[i].detach()
                else:
                    if isinstance(res, torch.Tensor):
                        res = res.detach()
                return res
            return g

        f = build_function_wrapper(f)

    if tensors is None:
        tensors = tn.meshgrid(domain, batch=batch)

    if not hasattr(tensors, '__len__'):
        tensors = [tensors]
    tensors = [t.decompress_tucker_factors(_clone=False) for t in tensors]
    Is = list(tensors[0].shape)
    N = len(Is)

    # Process ranks and cap them, if needed
    if ranks_tt is None:
        ranks_tt = 1
    else:
        kickrank = None
    if not hasattr(ranks_tt, '__len__'):
        ranks_tt = [ranks_tt]*(N-1)
    ranks_tt = [1] + list(ranks_tt) + [1]
    Rs = np.array(ranks_tt)
    for n in list(range(1, N)) + list(range(N-1, -1, -1)):
        Rs[n] = min(Rs[n-1]*Is[n-1], Rs[n], Is[n]*Rs[n+1])

    # Initialize cores at random
    cores = [torch.randn(Rs[n], Is[n], Rs[n+1]).to(device) for n in range(N)]

    # Prepare left and right sets
    lsets = [np.array([[0]])] + [None]*(N-1)
    randint = np.hstack([np.random.randint(0, Is[n+1], [max(Rs), 1]) for n in range(N-1)] + [np.zeros([max(Rs), 1], dtype=np.int)])
    rsets = [randint[:Rs[n+1], n:] for n in range(N-1)] + [np.array([[0]])]

    # Initialize left and right interfaces for `tensors`
    def init_interfaces():
        t_linterfaces = []
        t_rinterfaces = []
        for t in tensors:
            linterfaces = [torch.ones(1, t.ranks_tt[0]).to(device)] + [None]*(N-1)
            rinterfaces = [None]*(N-1) + [torch.ones(t.ranks_tt[t.dim()], 1).to(device)]
            for j in range(N-1):
                M = torch.ones(t.cores[-1].shape[-1], len(rsets[j])).to(device)
                for n in range(N-1, j, -1):
                    if t.cores[n].dim() == 3:  # TT core
                        M = torch.einsum('iaj,ja->ia', [t.cores[n][:, rsets[j][:, n-1-j], :].to(device), M])
                    else:  # CP factor
                        M = torch.einsum('ai,ia->ia', [t.cores[n][rsets[j][:, n-1-j], :].to(device), M])
                rinterfaces[j] = M
            t_linterfaces.append(linterfaces)
            t_rinterfaces.append(rinterfaces)
        return t_linterfaces, t_rinterfaces
    t_linterfaces, t_rinterfaces = init_interfaces()

    # Create a validation set
    Xs_val = [torch.as_tensor(np.random.choice(I, int(val_size))).to(device) for I in Is]
    ys_val = f(*[t[Xs_val].torch() for t in tensors])
    if ys_val.dim() > 1:
        assert ys_val.dim() == 2
        assert ys_val.shape[1] == 1
        ys_val = ys_val[:, 0]
    assert len(ys_val) == val_size
    norm_ys_val = torch.norm(ys_val)

    if verbose:
        print('Cross-approximation over a {}D domain containing {:g} grid points:'.format(N, tensors[0].numel()))
    start = time.time()
    converged = False

    info = {
        'nsamples': 0,
        'eval_time': 0,
        'val_epss': [],
        'min': 0,
        'argmin': None
    }
    if record_samples:
        info['sample_positions'] = torch.zeros(0, N).to(device)
        info['sample_values'] = torch.zeros(0).to(device)

    def evaluate_function(j):  # Evaluate function over Rs[j] x Rs[j+1] fibers, each of size I[j]
        Xs = []
        for k, t in enumerate(tensors):
            if tensors[k].cores[j].dim() == 3:  # TT core
                V = torch.einsum('ai,ibj,jc->abc', [t_linterfaces[k][j], tensors[k].cores[j], t_rinterfaces[k][j]])
            else:  # CP factor
                V = torch.einsum('ai,bi,ic->abc', [t_linterfaces[k][j], tensors[k].cores[j], t_rinterfaces[k][j]])
            Xs.append(V.flatten())

        eval_start = time.time()
        evaluation = f(*Xs)
        if record_samples:
            info['sample_positions'] = torch.cat((info['sample_positions'], torch.cat([x[:, None] for x in Xs], dim=1)), dim=0)
            info['sample_values'] = torch.cat((info['sample_values'], evaluation))
        info['eval_time'] += time.time() - eval_start
        if _minimize:
            evaluation = np.pi/2 - torch.atan((evaluation - info['min']))  # Function used by I. Oseledets for TT minimization in ttpy
            evaluation_argmax = torch.argmax(evaluation)
            eval_min = torch.tan(np.pi/2 - evaluation[evaluation_argmax]) + info['min']
            if info['min'] == 0 or eval_min < info['min']:
                coords = np.unravel_index(evaluation_argmax, [Rs[j], Is[j], Rs[j + 1]])
                info['min'] = eval_min
                info['argmin'] = tuple(lsets[j][coords[0]][1:]) + tuple([coords[1]]) + tuple(rsets[j][coords[2]][:-1])

        # Check for nan/inf values
        if evaluation.dim() == 2:
            evaluation = evaluation[:, 0]
        invalid = (torch.isnan(evaluation) | torch.isinf(evaluation)).nonzero()
        if len(invalid) > 0:
            invalid = invalid[0].item()
            raise ValueError('Invalid return value for function {}: f({}) = {}'.format(function, ', '.join('{:g}'.format(x[invalid].detach().cpu().numpy()) for x in Xs),
                                                                         f(*[x[invalid:invalid+1][:, None] for x in Xs]).item()))

        V = torch.reshape(evaluation, [Rs[j], Is[j], Rs[j + 1]])
        info['nsamples'] += V.numel()
        return V

    # Sweeps
    for i in range(max_iter):

        if verbose:
            print('iter: {: <{}}'.format(i, len('{}'.format(max_iter))+1), end='')
            sys.stdout.flush()

        left_locals = []

        # Left-to-right
        for j in range(N-1):

            # Update tensors for current indices
            V = evaluate_function(j)

            # QR + maxvol towards the right
            V = torch.reshape(V, [-1, V.shape[2]])  # Left unfolding
            Q, R = torch.qr(V)
            if _minimize:
                local, _ = maxvolpy.maxvol.rect_maxvol(Q.detach().cpu().numpy(), maxK=Q.shape[1])
            else:
                local, _ = maxvolpy.maxvol.maxvol(Q.detach().cpu().numpy())
            V = torch.lstsq(Q.t(), Q[local, :].t())[0].t()
            cores[j] = torch.reshape(V, [Rs[j], Is[j], Rs[j+1]])
            left_locals.append(local)

            # Map local indices to global ones
            local_r, local_i = np.unravel_index(local, [Rs[j], Is[j]])
            lsets[j+1] = np.c_[lsets[j][local_r, :], local_i]
            for k, t in enumerate(tensors):
                if t.cores[j].dim() == 3:  # TT core
                    t_linterfaces[k][j+1] = torch.einsum('ai,iaj->aj', [t_linterfaces[k][j][local_r, :], t.cores[j][:, local_i, :]])
                else:  # CP factor
                    t_linterfaces[k][j+1] = torch.einsum('ai,ai->ai', [t_linterfaces[k][j][local_r, :], t.cores[j][local_i, :]])

        # Right-to-left sweep
        for j in range(N-1, 0, -1):

            # Update tensors for current indices
            V = evaluate_function(j)

            # QR + maxvol towards the left
            V = torch.reshape(V, [Rs[j], -1])  # Right unfolding
            Q, R = torch.qr(V.t())
            if _minimize:
                local, _ = maxvolpy.maxvol.rect_maxvol(Q.detach().cpu().numpy(), maxK=Q.shape[1])
            else:
                local, _ = maxvolpy.maxvol.maxvol(Q.detach().cpu().numpy())
            V = torch.lstsq(Q.t(), Q[local, :].t())[0]
            cores[j] = torch.reshape(torch.as_tensor(V), [Rs[j], Is[j], Rs[j+1]])

            # Map local indices to global ones
            local_i, local_r = np.unravel_index(local, [Is[j], Rs[j+1]])
            rsets[j-1] = np.c_[local_i, rsets[j][local_r, :]]
            for k, t in enumerate(tensors):
                if t.cores[j].dim() == 3:  # TT core
                    t_rinterfaces[k][j-1] = torch.einsum('iaj,ja->ia', [t.cores[j][:, local_i, :], t_rinterfaces[k][j][:, local_r]])
                else:  # CP factor
                    t_rinterfaces[k][j-1] = torch.einsum('ai,ia->ia', [t.cores[j][local_i, :], t_rinterfaces[k][j][:, local_r]])

        # Leave the first core ready
        V = evaluate_function(0)
        cores[0] = V

        # Evaluate validation error
        val_eps = torch.norm(ys_val - tn.Tensor(cores)[Xs_val].torch()) / norm_ys_val
        info['val_epss'].append(val_eps)
        if val_eps < eps:
            converged = True

        if verbose:  # Print status
            if _minimize:
                print('| best: {:.8g}'.format(info['min']), end='')
            else:
                print('| eps: {:.3e}'.format(val_eps), end='')
            print(' | total time: {:8.4f} | largest rank: {:3d}'.format(time.time() - start, max(Rs)), end='')
            if converged:
                print(' <- converged: eps < {}'.format(eps))
            elif i == max_iter-1:
                print(' <- max_iter was reached: {}'.format(max_iter))
            else:
                print()
        if converged:
            break
        elif i < max_iter-1 and kickrank is not None:  # Augment ranks
            newRs = Rs.copy()
            newRs[1:-1] = np.minimum(rmax, newRs[1:-1]+kickrank)
            for n in list(range(1, N)) + list(range(N-1, 0, -1)):
                newRs[n] = min(newRs[n-1]*Is[n-1], newRs[n], Is[n]*newRs[n+1])
            extra = np.hstack([np.random.randint(0, Is[n+1], [max(newRs), 1]) for n in range(N-1)] + [np.zeros([max(newRs), 1], dtype=np.int)])
            for n in range(N-1):
                if newRs[n+1] > Rs[n+1]:
                    rsets[n] = np.vstack([rsets[n], extra[:newRs[n+1]-Rs[n+1], n:]])
            Rs = newRs
            t_linterfaces, t_rinterfaces = init_interfaces()  # Recompute interfaces

    if val_eps > eps and not _minimize and not suppress_warnings:
        logging.warning('eps={:g} (larger than {}) when cross-approximating {}'.format(val_eps, eps, function))

    if verbose:
        print('Did {} function evaluations, which took {:.4g}s ({:.4g} evals/s)'.format(info['nsamples'], info['eval_time'], info['nsamples'] / info['eval_time']))
        print()

    if return_info:
        info['lsets'] = lsets
        info['rsets'] = rsets
        info['Rs'] = Rs
        info['left_locals'] = left_locals
        info['total_time'] = time.time()-start
        info['val_eps'] = val_eps
        return tn.Tensor([c if isinstance(c, torch.Tensor) else torch.tensor(c) for c in cores], batch=batch), info
    else:
        return tn.Tensor([c if isinstance(c, torch.Tensor) else torch.tensor(c) for c in cores], batch=batch)
