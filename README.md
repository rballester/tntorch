# tntorch - Tensor Network Learning with PyTorch

[Welcome to *tntorch*](https://github.com/rballester/tntorch/blob/master/tutorials/introduction.ipynb), a PyTorch-powered learning library using tensor networks. Such networks are unique in that they use *multilinear* neural units (instead of non-linear activation units).

The tensor networks currently supported are those in the *Tensor Train - Tucker* format. It is a flexible model that generalizes both the [Tucker](https://epubs.siam.org/doi/pdf/10.1137/S0895479898346995) and [tensor train (TT)](https://epubs.siam.org/doi/pdf/10.1137/090752286) formats --two of the most popular tensor decompositions. For example, the following networks both represent a 4D tensor (i.e. a real function that can take I1 x I2 x I3 x I4 possible values):

<p align="center"><img src="https://github.com/rballester/tntorch/blob/master/images/tensors.jpg" width="600" title="TT-Tucker"></p>

With *tntorch* you can handle tensors in a transparent, yet super-compressed form as if they were plain NumPy arrays or PyTorch tensors:

```
> import tntorch as tn
> t = tn.randn([32]*4, ranks_tt=5)  # Random 4D TT tensor of shape 32 x 32 x 32 x 32 and rank 5
> print(t)

4D TT tensor:

 32  32  32  32
  |   |   |   |
 (0) (1) (2) (3)
 / \ / \ / \ / \
1   5   5   5   1

> print(tn.mean(t))

tensor(8.0388)

> print(tn.norm(t))

tensor(9632.3726)
```

Decompressing tensors is easy:  

```
> print(t.full().shape)
torch.Size([32, 32, 32, 32])
```

Thanks to PyTorch's automatic differentiation, you can easily define all sorts of loss functions on tensors:

```
def loss(t):
    return torch.norm(t[:, 0, 10:, [3, 4]].full())  # NumPy-like "fancy indexing" for arrays
```

Most importantly, loss functions can be defined on **compressed** tensors as well:

```
def loss(t):
    return tn.norm(t[:3, :3, :3, :3] - t[-3:, -3:, -3:, -3:])
```

Check out the [introductory notebook](https://github.com/rballester/tntorch/blob/master/tutorials/introduction.ipynb) for all the details on the basics.

## Tutorial Notebooks

- [Introduction](https://github.com/rballester/tntorch/blob/master/tutorials/introduction.ipynb)
- [Active subspaces](https://github.com/rballester/tntorch/blob/master/tutorials/active_subspaces.ipynb)
- [ANOVA decomposition](https://github.com/rballester/tntorch/blob/master/tutorials/anova.ipynb)
- [Boolean logic](https://github.com/rballester/tntorch/blob/master/tutorials/logic.ipynb)
- [Discrete/weighted finite automata](https://github.com/rballester/tntorch/blob/master/tutorials/automata.ipynb)
- [Polynomial chaos expansions](https://github.com/rballester/tntorch/blob/master/tutorials/pce.ipynb)
- [Tensor completion](https://github.com/rballester/tntorch/blob/master/tutorials/completion.ipynb)
- [Tensor decomposition](https://github.com/rballester/tntorch/blob/master/tutorials/decompositions.ipynb)
- [Sensitivity analysis](https://github.com/rballester/tntorch/blob/master/tutorials/sobol.ipynb)

## Planned

- Classification
- Dynamical systems
- Gibbs sampling
- Hidden Markov models
- Polyharmonic regression
- Ridge regression
- Tensor weight regression/classification