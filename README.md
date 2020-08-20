[![Documentation Status](https://readthedocs.org/projects/tntorch/badge/?version=latest)](https://tntorch.readthedocs.io/en/latest/?badge=latest)

# tntorch - Tensor Network Learning with PyTorch

**[Read the Docs site: *http://tntorch.readthedocs.io/*](http://tntorch.readthedocs.io/)**

[Welcome to *tntorch*](https://github.com/VMML/tntorch/blob/master/docs/tutorials/introduction.ipynb), a PyTorch-powered modeling and learning library using tensor networks. Such networks are unique in that [they use *multilinear* neural units](https://arxiv.org/abs/1711.00811) (instead of non-linear activation units). Features include:

- Basic and fancy **indexing** of tensors, **broadcasting**, **assignment**, etc.
- Tensor **decomposition** and **reconstruction**
- Element-wise and tensor-tensor **arithmetics**
- Building tensors from black-box functions using **cross-approximation**
- Finding global **maxima** and **minima** from tensors
- **Statistics** and **sensitivity analysis**
- **Optimization** using autodifferentiation
- **Misc. operations** on tensors: stacking, unfolding, sampling, derivating, etc.
- **Batch operations** (work in progress)


Available [tensor formats](https://github.com/rballester/tntorch/blob/master/docs/tutorials/main_formats.ipynb) include:

- [CANDECOMP/PARAFAC (CP)](https://epubs.siam.org/doi/pdf/10.1137/07070111X)
- [Tucker](https://epubs.siam.org/doi/pdf/10.1137/S0895479898346995)
- [Tensor train (TT)](https://epubs.siam.org/doi/abs/10.1137/090752286?journalCode=sjoce3)
- Hybrids: CP-Tucker, TT-Tucker, etc. 
- [Partial support](https://github.com/rballester/tntorch/blob/master/docs/tutorials/other_formats.ipynb) for other decompositions such as [INDSCAL, CANDELINC, DEDICOM, PARATUCK2](https://epubs.siam.org/doi/pdf/10.1137/07070111X), and custom formats

For example, the following networks both represent a 4D tensor (i.e. a real function that can take I1 x I2 x I3 x I4 possible values) in the TT and TT-Tucker formats:

<p align="center"><img src="https://github.com/rballester/tntorch/blob/master/images/tensors.jpg" width="600" title="TT-Tucker"></p>

In *tntorch*, **all tensor decompositions share the same interface**. You can handle them in a transparent form, as if they were plain NumPy arrays or PyTorch tensors:

```
> import tntorch as tn
> t = tn.randn(32, 32, 32, 32, ranks_tt=5)  # Random 4D TT tensor of shape 32 x 32 x 32 x 32 and TT-rank 5
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
> print(t.torch().shape)
torch.Size([32, 32, 32, 32])
```

Thanks to PyTorch's automatic differentiation, you can easily define all sorts of loss functions on tensors:

```
def loss(t):
    return torch.norm(t[:, 0, 10:, [3, 4]].torch())  # NumPy-like "fancy indexing" for arrays
```

Most importantly, loss functions can be defined on **compressed** tensors as well:

```
def loss(t):
    return tn.norm(t[:3, :3, :3, :3] - t[-3:, -3:, -3:, -3:])
```

Check out the [introductory notebook](https://github.com/rballester/tntorch/blob/master/docs/tutorials/introduction.ipynb) for all the details on the basics.

## Tutorial Notebooks

- [Introduction](https://github.com/rballester/tntorch/blob/master/docs/tutorials/introduction.ipynb)
- [Active subspaces](https://github.com/rballester/tntorch/blob/master/docs/tutorials/active_subspaces.ipynb)
- [ANOVA decomposition](https://github.com/rballester/tntorch/blob/master/docs/tutorials/anova.ipynb)
- [Boolean logic](https://github.com/rballester/tntorch/blob/master/docs/tutorials/logic.ipynb)
- [Classification](https://github.com/rballester/tntorch/blob/master/docs/tutorials/classification.ipynb)
- [Cross-approximation](https://github.com/rballester/tntorch/blob/master/docs/tutorials/cross.ipynb)
- [Differentiation](https://github.com/rballester/tntorch/blob/master/docs/tutorials/derivatives.ipynb)
- [Discrete/weighted finite automata](https://github.com/rballester/tntorch/blob/master/docs/tutorials/automata.ipynb)
- [Exponential machines](https://github.com/rballester/tntorch/blob/master/docs/tutorials/exponential_machines.ipynb)
- [Main tensor formats available](https://github.com/rballester/tntorch/blob/master/docs/tutorials/main_formats.ipynb)
- [Other custom formats](https://github.com/rballester/tntorch/blob/master/docs/tutorials/other_formats.ipynb)
- [Polynomial chaos expansions](https://github.com/rballester/tntorch/blob/master/docs/tutorials/pce.ipynb)
- [Tensor arithmetics](https://github.com/rballester/tntorch/blob/master/docs/tutorials/arithmetics.ipynb)
- [Tensor completion and regression](https://github.com/rballester/tntorch/blob/master/docs/tutorials/completion.ipynb)
- [Tensor decomposition](https://github.com/rballester/tntorch/blob/master/docs/tutorials/decompositions.ipynb)
- [Sensitivity analysis](https://github.com/rballester/tntorch/blob/master/docs/tutorials/sobol.ipynb)
- [Vector field data](https://github.com/rballester/tntorch/blob/master/docs/tutorials/vector_fields.ipynb)

## Installation

You can install *tntorch* using *pip*:

```
pip install tntorch
```

Alternatively, you can install from the source:

```
git clone https://github.com/rballester/tntorch.git
cd tntorch
pip install .
```

For functions that use cross-approximation, the optional package [*maxvolpy*](https://bitbucket.org/muxas/maxvolpy) is required (it can be installed via `pip install maxvolpy`).

## Testing

We use [*pytest*](https://docs.pytest.org/en/latest/). Simply run:

```
cd tests/
pytest
```

## Contributing

Pull requests are welcome! 

Besides using the [issue tracker](https://github.com/rballester/tntorch/issues), feel also free to contact me at <rafael.ballester@ie.edu>.
