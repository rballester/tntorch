import tntorch as tn
import torch


def cumsum(t, dim=None):
    """
    Computes the cumulative sum of a tensor along one or several dims, similarly to PyTorch's `cumsum()`.

    :param t: input :class:`Tensor`
    :param dim: an int or list of ints (default: all)

    :return: a :class:`Tensor` of the same shape
    """

    if dim is None:
        dim = range(t.dim())
    if not hasattr(dim, '__len__'):
        dim = [dim]

    t = t.clone()
    for n in dim:
        if t.Us[n] is None:
            t.cores[n] = torch.cumsum(t.cores[n], dim=-2)
        else:
            t.Us[n] = torch.cumsum(t.Us[n], dim=0)
    return t


def cumprod(t, dim=None):
    """
    Computes the cumulative sum of a tensor along one or several dims, similarly to PyTorch's `cumprod()`.

    Note: this function is approximate and uses cross-approximation (:func:`tntorch.cross()`)

    :param t: input :class:`Tensor`
    :param dim: an int or list of ints (default: all)

    :return: a :class:`Tensor` of the same shape
    """

    return tn.exp(tn.cumsum(tn.log(t), dim=dim))


"""
Unary operations (using cross-approximation)
"""


def abs(t):
    """
    Element-wise absolute value computed using cross-approximation; see PyTorch's `abs()`.

    :param t: input :class:`Tensor`

    :return: a :class:`Tensor`
    """

    return tn.cross(lambda x: torch.abs(x), tensors=t, verbose=False)


def acos(t):
    """
    Element-wise arccosine computed using cross-approximation; see PyTorch's `acos()`.

    :param t: input :class:`Tensor`s

    :return: a :class:`Tensor`
    """

    return tn.cross(lambda x: torch.acos(x), tensors=t, verbose=False)


def asin(t):
    """
    Element-wise arcsine computed using cross-approximation; see PyTorch's `asin()`.

    :param t: input :class:`Tensor`

    :return: a :class:`Tensor`
    """

    return tn.cross(lambda x: torch.asin(x), tensors=t, verbose=False)


def cos(t):
    """
    Element-wise cosine computed using cross-approximation; see PyTorch's `cos()`.

    :param t: input :class:`Tensor`

    :return: a :class:`Tensor`
    """

    return tn.cross(lambda x: torch.cos(x), tensors=t, verbose=False)


def cosh(t):
    """
    Element-wise hyperbolic cosine computed using cross-approximation; see PyTorch's `cosh()`.

    :param t: input :class:`Tensor`

    :return: a :class:`Tensor`
    """

    return tn.cross(lambda x: torch.cosh(x), tensors=t, verbose=False)


def erf(t):
    """
    Element-wise error function computed using cross-approximation; see PyTorch's `erf()`.

    :param t: input :class:`Tensor`

    :return: a :class:`Tensor`
    """

    return tn.cross(lambda x: torch.erf(x), tensors=t, verbose=False)


def erfinv(t):
    """
    Element-wise inverse error function computed using cross-approximation; see PyTorch's `erfinv()`.

    :param t: input :class:`Tensor`

    :return: a :class:`Tensor`
    """

    return tn.cross(lambda x: torch.erfinv(x), tensors=t, verbose=False)


def exp(t):
    """
    Element-wise exponentiation computed using cross-approximation; see PyTorch's `exp()`.

    :param t: input :class:`Tensor`

    :return: a :class:`Tensor`
    """

    return tn.cross(lambda x: torch.exp(x), tensors=t, verbose=False)


def log(t):
    """
    Element-wise natural logarithm computed using cross-approximation; see PyTorch's `log()`.

    :param t: input :class:`Tensor`

    :return: a :class:`Tensor`
    """

    return tn.cross(lambda x: torch.log(x), tensors=t, verbose=False)


def log10(t):
    """
    Element-wise base-10 logarithm computed using cross-approximation; see PyTorch's `log10()`.

    :param t: input :class:`Tensor`

    :return: a :class:`Tensor`
    """

    return tn.cross(lambda x: torch.log10(x), tensors=t, verbose=False)


def log2(t):
    """
    Element-wise base-2 logarithm computed using cross-approximation; see PyTorch's `log2()`.

    :param t: input :class:`Tensor`

    :return: a :class:`Tensor`
    """

    return tn.cross(lambda x: torch.log2(x), tensors=t, verbose=False)


def reciprocal(t):
    """
    Element-wise reciprocal computed using cross-approximation; see PyTorch's `reciprocal()`.

    :param t: input :class:`Tensor`

    :return: a :class:`Tensor`
    """

    return tn.cross(lambda x: torch.reciprocal(x), tensors=t, verbose=False)


def rsqrt(t):
    """
    Element-wise square-root reciprocal computed using cross-approximation; see PyTorch's `rsqrt()`.

    :param t: input :class:`Tensor`

    :return: a :class:`Tensor`
    """

    return tn.cross(lambda x: torch.rsqrt(x), tensors=t, verbose=False)


def sigmoid(t):
    """
    Element-wise sigmoid computed using cross-approximation; see PyTorch's `igmoid()`.

    :param t: input :class:`Tensor`

    :return: a :class:`Tensor`
    """

    return tn.cross(lambda x: torch.sigmoid(x), tensors=t, verbose=False)


def sin(t):
    """
    Element-wise sine computed using cross-approximation; see PyTorch's `in()`.

    :param t: input :class:`Tensor`

    :return: a :class:`Tensor`
    """

    return tn.cross(lambda x: torch.sin(x), tensors=t, verbose=False)


def sinh(t):
    """
    Element-wise hyperbolic sine computed using cross-approximation; see PyTorch's `inh()`.

    :param t: input :class:`Tensor`

    :return: a :class:`Tensor`
    """

    return tn.cross(lambda x: torch.sinh(x), tensors=t, verbose=False)


def sqrt(t):
    """
    Element-wise square root computed using cross-approximation; see PyTorch's `qrt()`.

    :param t: input :class:`Tensor`

    :return: a :class:`Tensor`
    """

    return tn.cross(lambda x: torch.sqrt(x), tensors=t, verbose=False)


def tan(t):
    """
    Element-wise tangent computed using cross-approximation; see PyTorch's `tan()`.

    :param t: input :class:`Tensor`

    :return: a :class:`Tensor`
    """

    return tn.cross(lambda x: torch.tan(x), tensors=t, verbose=False)


def tanh(t):
    """
    Element-wise hyperbolic tangent computed using cross-approximation; see PyTorch's `tanh()`.

    :param t: input :class:`Tensor`

    :return: a :class:`Tensor`
    """

    return tn.cross(lambda x: torch.tanh(x), tensors=t, verbose=False)


"""
Binary operations (using cross-approximation)
"""


def add(t1, t2):
    """
    Element-wise addition computed using cross-approximation; see PyTorch's `add()`.

    :param t1: input :class:`Tensor`
    :param t2: input :class:`Tensor`

    :return: a :class:`Tensor`
    """

    return tn.cross(lambda x, y: torch.add(x, y), tensors=[t1, t2], verbose=False)


def atan2(t1, t2):
    """
    Element-wise arctangent computed using cross-approximation; see PyTorch's `atan2()`.

    :param t1: input :class:`Tensor`
    :param t2: input :class:`Tensor`

    :return: a :class:`Tensor`
    """

    return tn.cross(lambda x, y: torch.atan2(x, y), tensors=[t1, t2], verbose=False)


def div(t1, t2):
    """
    Element-wise division computed using cross-approximation; see PyTorch's `div()`.

    :param t1: input :class:`Tensor`
    :param t2: input :class:`Tensor`

    :return: a :class:`Tensor`
    """

    return t1/t2


def mul(t1, t2):
    """
    Element-wise product computed using cross-approximation; see PyTorch's `mul()`.

    :param t1: input :class:`Tensor`
    :param t2: input :class:`Tensor`

    :return: a :class:`Tensor`
    """

    return tn.cross(lambda x, y: torch.mul(x, y), tensors=[t1, t2], verbose=False)


def pow(t1, t2):
    """
    Element-wise power operation computed using cross-approximation; see PyTorch's `pow()`.

    :param t1: input :class:`Tensor`
    :param t2: input :class:`Tensor`

    :return: a :class:`Tensor`
    """

    return t1**t2
