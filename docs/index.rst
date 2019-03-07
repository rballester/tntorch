tntorch -- Tensor Network Learning with PyTorch
===============================================

.. image:: tntorch.svg
   :width: 300 px
   :align: center

This is a `PyTorch <http://pytorch.org/>`__-powered library for tensor modeling and learning that features transparent support for the `tensor train (TT) model <https://epubs.siam.org/doi/pdf/10.1137/090752286>`_, `CANDECOMP/PARAFAC (CP) <https://epubs.siam.org/doi/pdf/10.1137/07070111X>`_, the `Tucker model <https://epubs.siam.org/doi/pdf/10.1137/S0895479898346995>`_, and more. Supported operations (CPU and GPU) include:

- Basic and fancy **indexing** of tensors, **broadcasting**, **assignment**, etc.
- Tensor **decomposition** and **reconstruction**
- Element-wise and tensor-tensor **arithmetics**
- Building tensors from black-box functions using **cross-approximation**
- **Statistics** and **sensitivity analysis**
- **Optimization** using autodifferentiation
- **Misc. operations** on tensors: stacking, unfolding, sampling, derivating, etc.

Get the Code
------------

You can clone the project from `tntorch's GitHub page <https://github.com/rballester/tntorch>`_:

.. code-block:: bash

    git clone https://github.com/rballester/tntorch.git

or get it as a `zip file <https://github.com/rballester/tntorch/archive/master.zip>`_.
    
Installation
------------

The main dependencies are `NumPy <http://www.numpy.org/>`_ and `PyTorch <https://pytorch.org/>`_ (we recommend to install those with `Conda <https://conda.io/en/latest/>`_ or `Miniconda <https://conda.io/en/latest/miniconda.html>`_). To install *tntorch*, run:

.. code-block:: bash

   cd tntorch
   pip install .

First Steps
-----------

Some basic tensor manipulation:

.. code-block:: python

   import tntorch as tn
   
   t = tn.ones(64, 64)  # 64 x 64 tensor, filled with ones
   t = t[:, :, None] + 2*t[:, None, :]  # Singleton dimensions, broadcasting, and arithmetics
   print(tn.mean(t))  # Result: 3

Decomposing a tensor:
   
.. code-block:: python

   import tntorch as tn
   
   data = ...  # A NumPy or PyTorch tensor
   t1 = tn.Tensor(data, ranks_cp=5)  # A CP decomposition
   t2 = tn.Tensor(data, ranks_tucker=5)  # A Tucker decomposition
   t3 = tn.Tensor(data, ranks_tt=5)  # A tensor train decomposition

To get fully on board, check out the complete documentation:

.. toctree::
   :hidden:

   Welcome <self>

.. toctree::
   :maxdepth: 1

   goals
   api
   tutorial-notebooks
   contact
