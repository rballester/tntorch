import tntorch as tn
import torch
import numpy as np
import time
from functools import reduce


def optimize(tensors, loss_function, optimizer=torch.optim.Adam, tol=1e-4, max_iter=1e4, print_freq=500, verbose=True):
    """
    High-level wrapper for iterative learning.

    Default stopping criterion: either the absolute (or relative) loss improvement must fall below `tol`.
    In addition, the rate loss improvement must be slowing down.

    :param tensors: one or several tensors; will be fed to `loss_function` and optimized in place
    :param loss_function: must take `tensors` and return a scalar (or tuple thereof)
    :param optimizer: one from https://pytorch.org/docs/stable/optim.html. Default is torch.optim.Adam
    :param tol: stopping criterion
    :param max_iter: default is 1e4
    :param print_freq: progress will be printed every this many iterations
    :param verbose:
    """

    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    parameters = []
    for t in tensors:
        if isinstance(t, tn.Tensor):
            parameters.extend([c for c in t.cores if c.requires_grad])
            parameters.extend([U for U in t.Us if U is not None and U.requires_grad])
        elif t.requires_grad:
            parameters.append(t)
    if len(parameters) == 0:
        raise ValueError("There are no parameters to optimize. Did you forget a requires_grad=True somewhere?")

    optimizer = optimizer(parameters)
    losses = []
    converged = False
    start = time.time()
    iter = 0
    while True:
        optimizer.zero_grad()
        loss = loss_function(*tensors)
        if not isinstance(loss, (tuple, list)):
            loss = [loss]
        total_loss = sum(loss)
        total_loss.backward(retain_graph=True)
        optimizer.step()

        losses.append(total_loss.detach())
        

        if len(losses) >= 2:
            delta_loss = (losses[-1] - losses[-2])
        else:
            delta_loss = float('-inf')
        if iter >= 2 and tol is not None and (losses[-1] <= tol or -delta_loss / losses[-1] <= tol) and losses[-2] - \
                losses[-1] < losses[-3] - losses[-2]:
            converged = True
            break
        if iter == max_iter:
            break
        if verbose and iter % print_freq == 0:
            print('iter: {: <{}} | loss: '.format(iter, len('{}'.format(max_iter))), end='')
            print(' + '.join(['{:10.6f}'.format(l.item()) for l in loss]), end='')
            if len(loss) > 1:
                print(' = {:10.4}'.format(losses[-1].item()), end='')
            print(' | total time: {:9.4f}'.format(time.time() - start))

        iter += 1
    if verbose:
        print('iter: {: <{}} | loss: '.format(iter, len('{}'.format(max_iter))), end='')
        print(' + '.join(['{:10.6f}'.format(l.item()) for l in loss]), end='')
        if len(loss) > 1:
            print(' = {:10.4}'.format(losses[-1].item()), end='')
        print(' | total time: {:9.4f}'.format(time.time() - start), end='')
        if converged:
            print(' <- converged (tol={})'.format(tol))
        else:
            print(' <- max_iter was reached: {}'.format(max_iter))


def dof(t):
    """
    Compute the number of degrees of freedom of a tensor network.

    It is the sum of sizes of all its tensor nodes that have the requires_grad=True flag.

    :param t: input tensor

    :return: an integer
    """

    result = 0
    for n in range(t.dim()):
        if t.cores[n].requires_grad:
            result += t.cores[n].numel()
        if t.Us[n] is not None and t.Us[n].requires_grad:
            result += t.Us[n].numel()
    return result