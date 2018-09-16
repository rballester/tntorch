import numpy as np
import torch
import tntorch as tn
torch.set_default_dtype(torch.float64)
import time


class Tensor(object):

    def __init__(self, data, Us=None, idxs=None, eps=None):  # TODO requires_grad
        if isinstance(data, (list, tuple)):
            # data = [torch.Tensor(d) for d in data]
            # TODO accept ndarrays
            if not all([d.dim() == 3 for d in data]):
                raise ValueError('All TT cores must have 3 dimensions')
            if not all([data[i].shape[-1] == data[i+1].shape[0] for i in range(len(data)-1)]):
                str = ' '.join(['({},{},{})'.format(d.shape[0], d.shape[1], d.shape[2]) for d in data])
                raise ValueError('Core ranks do not match: {}'.format(str))
            self.cores = data
        elif isinstance(data, np.ndarray):
            data = torch.Tensor(data)
        if isinstance(data, torch.Tensor):
            if eps is None:  # Naive TT formatting, don't even attempt to compress
                self.cores = []
                N = data.dim()
                shape = data.shape
                data = torch.reshape(torch.Tensor(data), [shape[0], -1])
                for n in range(1, N):
                    self.cores.append(torch.reshape(torch.eye(data.shape[0]), [data.shape[0] // shape[n-1],
                                                                               shape[n - 1], data.shape[0]]))
                    data = torch.reshape(data, (data.shape[0] * shape[n], data.shape[1] // shape[n]))
                self.cores.append(torch.reshape(data, [data.shape[0] // shape[N - 1], shape[N-1], 1]))
            else:  # TT-SVD (or TT-EIG) algorithm
                raise NotImplementedError
        if Us is not None:
            for n in range(self.ndim):
                if Us[n] is None:
                    continue
                assert Us[n].dim() == 2
                assert self.cores[n].shape[1] == Us[n].shape[1]
            self.Us = Us
        else:
            self.Us = [None]*self.ndim
        if idxs is not None:
            self.idxs = idxs
        else:
            self.idxs = [torch.arange(sh) for sh in self.shape]

    def __add__(self, other):
        if not isinstance(other, Tensor):
            factor = other
            other = Tensor([torch.ones([1, self.shape[n], 1]) for n in range(self.ndim)])
            other.cores[0].data *= factor
        if not np.array_equal(self.shape, other.shape):
            raise ValueError("Element-wise addition requires tensors to have equal shape")
        if self.ndim == 1:  # Special case
            return Tensor([self.full_tucker().cores[0] + other.full_tucker().cores[0]])
        cores = []
        Us = []
        for n in range(self.ndim):
            if self.Us[n] is not None and other.Us[n] is not None:
                # assert 0
                slice1 = torch.cat([core1, torch.zeros([core2.shape[0], core1.shape[1], core1.shape[2]])], dim=0)
                slice1 = torch.cat([slice1, torch.zeros(core1.shape[0]+core2.shape[0], core1.shape[1], core2.shape[2])], dim=2)
                slice2 = torch.cat([torch.zeros([core1.shape[0], core2.shape[1], core2.shape[2]]), core2], dim=0)
                slice2 = torch.cat([torch.zeros(core1.shape[0]+core2.shape[0], core2.shape[1], core1.shape[2]), slice2], dim=2)
                c = torch.cat([slice1, slice2], dim=1)
                cores.append(c)
                Us.append(torch.cat((self.Us[n], other.Us[n]), dim=1))
                continue
            core1 = self.cores[n]
            if self.Us[n] is not None:
                core1 = torch.einsum('ijk,aj->iak', (core1, self.Us[n]))
            core2 = other.cores[n]
            if other.Us[n] is not None:
                core2 = torch.einsum('ijk,aj->iak', (core2, other.Us[n]))
            column1 = torch.cat([core1, torch.zeros([core2.shape[0], self.shape[n], core1.shape[2]])], dim=0)
            column2 = torch.cat([torch.zeros([core1.shape[0], self.shape[n], core2.shape[2]]), core2], dim=0)
            c = torch.cat([column1, column2], dim=2)
            cores.append(c)
            Us.append(None)
        cores[0] = torch.sum(cores[0], dim=0, keepdim=True)
        cores[-1] = torch.sum(cores[-1], dim=2, keepdim=True)
        return Tensor(cores, Us=Us)

    def __radd__(self, other):
        if other is None:
            return self
        return other + self

    def __sub__(self, other):
        return self + -1*other

    def __rsub__(self, other):
        return -1*self + other

    def __neg__(self):
        return -1*self

    def __mul__(self, other):
        if not isinstance(other, Tensor):  # A scalar
            result = self.clone()
            result.cores[0].data *= other
            return result
        if not np.array_equal(self.shape, other.shape):
            raise ValueError("Element-wise multiplication requires tensors to have equal shape")
        cores = []
        Us = []
        for n in range(self.ndim):
            if self.Us[n] is not None and other.Us[n] is not None and self.cores[n].shape[1]*other.cores[n].shape[1] < self.shape[n]:
                cores.append(torch.reshape(torch.einsum('ijk,abc->iajbkc', (self.cores[n], other.cores[n])), 
                                           (self.cores[n].shape[0]*other.cores[n].shape[0],
                                            self.cores[n].shape[1]*other.cores[n].shape[1],
                                            self.cores[n].shape[2]*other.cores[n].shape[2])))
                Us.append(torch.reshape(torch.einsum('ij,ik->ijk', (self.Us[n], other.Us[n])),
                         (self.Us[n].shape[0], -1)))
                print(self.cores[n].shape, other.cores[n].shape, cores[-1].shape)
                print(self.Us[n].shape, other.Us[n].shape, Us[-1].shape)
            else:
                core1 = self.cores[n]
                core2 = other.cores[n]
                if self.Us[n] is not None:
                    core1 = torch.einsum('ijk,aj->iak', (core1, self.Us[n]))
                if other.Us[n] is not None:
                    core2 = torch.einsum('ijk,aj->iak', (core2, other.Us[n]))
                cores.append(tn.core_kron(core1, core2))
                Us.append(None)
                
        return tn.Tensor(cores, Us=Us)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (1./other)

    def __invert__(self):
        return 1 - self

    def __and__(self, other):
        return self*other

    def __or__(self, other):
        return self+other - self*other

    def __xor__(self, other):
        return self+other - 2*self*other

    def __eq__(self, other):
        return (self-other).norm() <= 1e-14

    def __ne__(self, other):
        return not self == other

    @property
    def shape(self):
        shape = []
        for n in range(self.ndim):
            if self.Us[n] is None:
                shape.append(self.cores[n].shape[1])
            else:
                shape.append(self.Us[n].shape[0])
        return torch.Size(shape)

    @property
    def ranks_tt(self):
        return np.array([c.shape[0] for c in self.cores] + [self.cores[-1].shape[-1]])

    @property
    def ranks_tucker(self):
        return np.array([c.shape[1] for c in self.cores])

    @property
    def ndim(self):
        return len(self.cores)

    @property
    def size(self):
        return torch.prod(torch.Tensor(list(self.shape)))

    def __repr__(self):

        if any([U is not None for U in self.Us]):
            format = 'TT-Tucker'
        else:
            format = 'TT'
        s = '{}D {} tensor:\n'.format(self.ndim, format)
        s += '\n'
        ttr = self.ranks_tt
        tuckerr = self.ranks_tucker

        if any([U is not None for U in self.Us]):

            # Shape
            row = [' ']*(4*self.ndim-1)
            shape = self.shape
            for n in range(self.ndim):
                if self.Us[n] is None:
                    continue
                lenn = len('{}'.format(shape[n]))
                row[n*4-lenn//2+2:n*4-lenn//2+lenn+2] = '{}'.format(shape[n])
            s += ''.join(row)
            s += '\n'

        # Tucker ranks
        row = [' ']*(4*self.ndim-1)
        for n in range(self.ndim):
            if self.Us[n] is None:
                lenr = len('{}'.format(tuckerr[n]))
                row[n*4-lenr//2+2:n*4-lenr//2+lenr+2] = '{}'.format(tuckerr[n])
            else:
                row[n*4+2:n*4+3] = '|'
        s += ''.join(row)
        s += '\n'

        row = [' ']*(4*self.ndim-1)
        for n in range(self.ndim):
            if self.Us[n] is None:
                row[n*4+2:n*4+3] = '|'
            else:
                lenr = len('{}'.format(tuckerr[n]))
                row[n*4-lenr//2+2:n*4-lenr//2+lenr+2] = '{}'.format(tuckerr[n])
        s += ''.join(row)
        s += '\n'

        # Nodes
        row = [' ']*(4*self.ndim-1)
        for n in range(self.ndim):
            lenn = len('({})'.format(n))
            row[(n+1)*4-(lenn-1)//2:(n+1)*4-(lenn-1)//2+lenn] = '({})'.format(n)
        s += ''.join(row[2:])
        s += '\n'

        # TT rank bars
        s += ' / \\'*self.ndim
        s += '\n'

        # Bottom: TT ranks
        row = [' ']*(4*self.ndim)
        for n in range(self.ndim+1):
            lenr = len('{}'.format(ttr[n]))
            row[n*4:n*4+lenr] = '{}'.format(ttr[n])
        s += ''.join(row)
        s += '\n'

        return s

    def dot(self, other, k=None):  # TODO support partial dot products
        assert np.array_equal(self.shape, other.shape)
        if k is None:
            k = min(self.ndim, other.ndim)
        Lprod = torch.ones([1, 1])
        for mu in range(self.ndim-1, self.ndim-1-k, -1):
            core1 = self.cores[mu]
            if self.Us[mu] is None:
                core2 = other.cores[mu]
                if other.Us[mu] is not None:
                    core1 = torch.einsum('ijk,ja->iak', (core1, other.Us[mu]))
            elif other.Us[mu] is None:
                core2 = torch.einsum('ijk,ja->iak', (other.cores[mu], self.Us[mu]))
            else:
                core2 = torch.einsum('ijk,ar,aj->irk', (other.cores[mu], self.Us[mu], other.Us[mu]))
            Ucore = torch.einsum('ijk,ka->ija', (core1, Lprod))
            Vcore = core2
            Lprod = torch.mm(Ucore.reshape([Ucore.shape[0], -1]), torch.t(Vcore.reshape([Vcore.shape[0], -1])))
        return torch.squeeze(Lprod)

    def _process_key(self, key):
        if not hasattr(key, '__len__'):
            key = (key,)
        if isinstance(key, tuple) or any([not isinstance(k, int) for k in key]):
            key = list(key)

        # Process ellipsis, if any
        nonecount = sum(1 for k in key if k is None)
        for i in range(len(key)):
            if key[i] is Ellipsis:
                key = key[:i] + [slice(None)] * (self.ndim - (len(key) - nonecount) + 1) + key[i + 1:]
                break
        if any([k is Ellipsis for k in key]):
            raise IndexError("Only one ellipsis is allowed, at most")
        if self.ndim - (len(key) - nonecount) < 0:
            raise IndexError("Too many index entries")

        # Fill remaining unspecified dimensions with slice(None)
        key = key + [slice(None)] * (self.ndim - (len(key) - nonecount))
        return key

    def __getitem__(self, key):
        """
        NumPy-style indexing for TT tensors. There are 5 accessors supported: slices, index arrays, integers, None, or
        another Tensor (selection via binary indexing)

        - Index arrays can be lists, tuples, or vectors
        - All index arrays must have the same length P
        - In NumPy, index arrays and slices can be interleaved. We do not admit that, as it requires expensive transpose operations

        """

        # Preprocessing
        if isinstance(key, Tensor):
            if torch.abs(tn.sum(key)-1) > 1e-8:
                raise ValueError("When indexing via a mask tensor, that mask should have exactly 1 accepting string")
            s = tn.accepted_inputs(key)[0]
            slicing = []
            for n in range(self.ndim):
                idx = self.idxs[n].long()
                idx[idx > 1] = 1
                idx = np.where(idx == s[n])[0]
                sl = slice(idx[0], idx[-1]+1)
                lenidx = len(idx)
                if lenidx == 1:
                    sl = idx.item()
                slicing.append(sl)
            return self[slicing]

        if isinstance(key, torch.Tensor):
            key = np.array(key, dtype=np.int)
        if isinstance(key, np.ndarray) and key.ndim == 2:
            key = [key[:, col] for col in range(key.shape[1])]
        key = self._process_key(key)
        last_mode = None
        index_done = False
        factors = {'int': None, 'index': None}
        cores = []
        Us = []
        counter = 0

        def insert_core(factors, core=None, key=None, U=None):
            if factors['index'] is not None:
                if factors['int'] is not None:
                    factors['index'] = torch.einsum('ij,jak->iak', (factors['int'], factors['index']))
                    factors['int'] = None
                cores.append(factors['index'])
                Us.append(None)
                factors['index'] = None
            if core is not None:
                if factors['int'] is not None:
                    if U is None:
                        cores.append(torch.einsum('ij,jak->iak', (factors['int'], core[:, key, :])))
                        Us.append(None)
                    else:
                        cores.append(torch.einsum('ij,jak->iak', (factors['int'], core)))
                        Us.append(U[key, :])
                    factors['int'] = None
                else:
                    if U is None:
                        cores.append(core[:, key, :])
                        Us.append(None)
                    else:
                        cores.append(core)
                        Us.append(U[key, :])

        def get_key(counter, key):
            if self.Us[counter] is None:
                return self.cores[counter][:, key, :]
            else:
                sl = self.Us[counter][key, :]
                if sl.dim() == 1:
                    return torch.einsum('ijk,j->ik', (self.cores[counter], sl))
                return torch.einsum('ijk,aj->iak', (self.cores[counter], sl))

        for i in range(len(key)):
            if hasattr(key[i], '__len__'):
                this_mode = 'index'
            elif key[i] is None:
                this_mode = 'none'
            elif isinstance(key[i], int):
                this_mode = 'int'
            elif isinstance(key[i], slice):
                this_mode = 'slice'
            else:
                raise IndexError

            if this_mode == 'none':
                insert_core(factors, torch.eye(self.cores[counter].shape[0])[:, None, :], key=slice(None), U=None)
            elif this_mode == 'slice':
                insert_core(factors, self.cores[counter], key=key[i], U=self.Us[counter])
                counter += 1
            elif this_mode == 'index':
                if index_done:
                    raise IndexError("All index arrays must appear contiguously")
                if factors['index'] is None:
                    factors['index'] = get_key(counter, key[i])
                else:
                    if factors['index'].shape[1] != len(key[i]):
                        raise ValueError("Index arrays must have the same length")
                    a1 = factors['index']
                    a2 = get_key(counter, key[i])
                    # Until https://github.com/pytorch/pytorch/issues/10661 is resolved
                    factors['index'] = torch.sum(a1[:, :, :, None]*a2.permute(1, 0, 2)[None, :, :, :], dim=2)
                counter += 1
            elif this_mode == 'int':
                if last_mode == 'index':
                    insert_core(factors)
                if factors['int'] is None:
                    factors['int'] = get_key(counter, key[i])
                else:
                    factors['int'] = torch.einsum('ij,jk->ik', (factors['int'], get_key(counter, key[i])))
                counter += 1
            last_mode = this_mode

        if last_mode == 'index':
            cores.append(factors['index'])
            Us.append(None)
        elif last_mode == 'int':
            if len(cores) > 0:
                cores[-1] = torch.einsum('ija,ak->ijk', (cores[-1], factors['int']))
            else:
                return factors['int'][0, 0]
        return tn.Tensor(cores, Us=Us)

    def __setitem__(self, key, value):  # TODO not fully working yet
        key = self._process_key(key)

        scalar = False
        if isinstance(value, np.ndarray):
            value = tn.Tensor(torch.Tensor(value))
        elif isinstance(value, torch.Tensor):
            value = tn.Tensor(value)
        elif isinstance(value, tn.Tensor):
            pass
        else:  # It's a scalar
            scalar = True

        subtract_cores = []
        add_cores = []
        for i in range(len(key)):
            if not isinstance(key[i], slice) and not hasattr(key[i], '__len__'):
                key[i] = slice(key[i], key[i]+1)
            chunk = self.cores[i][:, key[i], :]
            subtract_core = torch.zeros_like(self.cores[i])
            subtract_core[:, key[i], :] += chunk
            subtract_cores.append(subtract_core)
            if scalar:
                add_core = torch.zeros(1, self.shape[i], 1)
                add_core[:, key[i], :] += 1
                if i == 0:
                    add_core *= value
            else:
                if chunk.shape[1] != value.shape[i]:
                    raise ValueError('{}-th dimension mismatch in tensor assignment: {} (lhs) != {} (rhs)'.format(i, chunk.shape[1], value.shape[i]))
                add_core = torch.zeros(value.cores[i].shape[0], self.shape[i], value.cores[i].shape[2])
                add_core[:, key[i], :] += value.cores[i]
            add_cores.append(add_core)
        # print(tn.Tensor(subtract_cores), tn.Tensor(add_cores))
        result = self - tn.Tensor(subtract_cores) + tn.Tensor(add_cores)
        self.__init__(result.cores, result.Us, self.idxs)

    def full_tucker(self):
        """
        Decompresses this tensor only along the Tucker factors.

        :return: a tensor in TT format

        """

        cores = []
        for n in range(self.ndim):
            if self.Us[n] is not None:
                cores.append(torch.einsum('ijk,aj->iak', (self.cores[n], self.Us[n])))
            else:
                cores.append(self.cores[n].clone())
        return tn.Tensor(cores, idxs=self.idxs)

    def full(self):
        """
        Decompresses this tensor into a torch tensor.

        :return: a torch tensor

        """

        t = self.full_tucker()
        shape = []
        factor = torch.ones([1, 1])
        for n in range(t.ndim):
            shape.append(t.cores[n].shape[1])
            factor = torch.einsum('ai,ibj->abj', (factor, t.cores[n]))
            factor = factor.reshape([-1, t.cores[n].shape[-1]])
        factor = factor[..., 0]
        factor = factor.reshape(shape)
        return factor

    def numpy(self):
        """
        Decompresses this tensor into a NumPy multiarray.

        :return: a NumPy tensor

        """

        return self.full().detach().numpy()

    def clone(self):
        """
        Creates a copy of this tensor (calls clone() on all internal tensor network nodes)

        :return: another compressed tensor

        """

        cores = [self.cores[n].clone()for n in range(self.ndim)]
        Us = []
        for n in range(self.ndim):
            if self.Us[n] is None:
                Us.append(None)
            else:
                Us.append(self.Us[n].clone())
        if hasattr(self, 'idxs'):
            return tn.Tensor(cores, Us=Us, idxs=self.idxs)
        return tn.Tensor(cores, Us=Us)

    def factor_orthogonalize(self, mu):
        """
        Pushes the factor's non-orthogonal part to its corresponding core.

        This method works in place.

        """

        if self.Us[mu] is None:
            return
        Q, R = torch.qr(self.Us[mu])
        self.Us[mu] = Q
        self.cores[mu] = torch.einsum('ijk,aj->iak', (self.cores[mu], R))

    def left_orthogonalize(self, mu):
        """
        Makes the mu-th core left-orthogonal and pushes the R factor to its right core
        Note: this may change the ranks of the cores.

        This method works in place.

        :return: the R factor
        """

        assert 0 <= mu < self.ndim-1
        self.factor_orthogonalize(mu)
        Q, R = torch.qr(tn.left_unfolding(self.cores[mu]))
        self.cores[mu] = torch.reshape(Q, self.cores[mu].shape[:-1] + (Q.shape[1], ))
        rightcoreR = tn.right_unfolding(self.cores[mu+1])
        self.cores[mu+1] = torch.reshape(torch.mm(R, rightcoreR), (R.shape[0], ) + self.cores[mu+1].shape[1:])
        return R

    def right_orthogonalize(self, mu):
        """
        Makes the mu-th core right-orthogonal and pushes the L factor to its left core
        Note: this may change the ranks of the tensor.

        This method works in place.

        :return: the L factor
        """

        assert 1 <= mu < self.ndim
        self.factor_orthogonalize(mu)
        Q, L = torch.qr(tn.right_unfolding(self.cores[mu]).permute(1, 0))  # Torch has no rq() decomposition
        L = L.permute(1, 0)
        Q = Q.permute(1, 0)
        self.cores[mu] = torch.reshape(Q, (Q.shape[0], ) + self.cores[mu].shape[1:])
        leftcoreL = tn.left_unfolding(self.cores[mu-1])
        self.cores[mu-1] = torch.reshape(torch.mm(leftcoreL, L), self.cores[mu-1].shape[:-1] + (L.shape[1], ))
        return L

    def orthogonalize(self, mu):
        """
        Apply all left and right orthogonalizations needed to make the tensor mu-orthogonal.

        This method works in place.

        :returns L, R: left and right factors
        """

        L = torch.ones(1, 1)
        R = torch.ones(1, 1)
        for i in range(0, mu):
            R = self.left_orthogonalize(i)
        for i in range(self.ndim-1, mu, -1):
            L = self.right_orthogonalize(i)
        return R, L

    def round(self, eps=0, rmax=np.iinfo(np.int32).max, algorithm='svd', verbose=False):
        """
        Tries to recompress this tensor in place by reducing its TT ranks.

        Note: this method does not reduce Tucker ranks (yet).

        :param eps: this relative error will not be exceeded. Default is 0
        :param rmax: all ranks should be rmax at most
        :param algorithm: 'svd' (default) or 'eig'. The latter can be faster, but less accurate
        :param verbose:

        """

        N = self.ndim
        shape = self.shape
        start = time.time()
        self.orthogonalize(N-1)  # Make everything left-orthogonal
        if verbose:
            print('Orthogonalization time:', time.time() - start)
        delta = eps / max(1, np.sqrt(N - 1)) * torch.norm(self.cores[-1])
        for mu in range(N - 1, 0, -1):
            M = tn.right_unfolding(self.cores[mu])
            left, M = tn.truncated_svd(M, delta=delta, rmax=rmax, left_ortho=False, algorithm=algorithm, verbose=verbose)
            self.cores[mu] = torch.reshape(M, [-1, shape[mu], self.cores[mu].shape[2]])
            self.cores[mu-1] = torch.einsum('ijk,kl', (self.cores[mu-1], left))  # Pass factor to the left

    def set_factors(self, name, modes='all', requires_grad=False):
        """
        Sets factors Us of this tensor to be of a certain family.

        :param name: See `generate_basis()`
        :param modes: list of factors to set; default is 'all'
        :param requires_grad: whether the new factors should be optimizable. Default is False

        """

        if modes == 'all':
            modes = range(self.ndim)

        for m in modes:
            if self.Us[m] is None:
                self.Us[m] = tn.generate_basis(name, (self.cores[m].shape[1], self.cores[m].shape[1]))
            else:
                self.Us[m] = tn.generate_basis(name, self.Us[m].shape)
            self.Us[m].requires_grad = requires_grad
