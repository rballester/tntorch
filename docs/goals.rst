Project Goals
=============

This package was born to bring together some of the most popular tensor decomposition models (including CP, Tucker, and the tensor train) under a common interface. Thus, we use *one class* for all those models. They are all particular cases of `tensor networks <https://arxiv.org/abs/1609.00893>`_, and the idea is that decomposing, manipulating, and reconstructing tensors can be (to some extent) abstracted away from the particular decomposition format.

Building on top of `PyTorch <http://pytorch.org/>`_'s flexibility and built-in automatic differentiation, the overall goal is to exploit those features and allow users to quickly develop, model, and fit various tensor decompositions in a range of data science applications.
