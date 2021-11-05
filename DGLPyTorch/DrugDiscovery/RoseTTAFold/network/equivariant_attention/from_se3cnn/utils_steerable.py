import os
import torch
import math
import numpy as np
from equivariant_attention.from_se3cnn.SO3 import irr_repr, torch_default_dtype
from equivariant_attention.from_se3cnn.cache_file import cached_dirpklgz
from equivariant_attention.from_se3cnn.representations import SphericalHarmonics

################################################################################
# Solving the constraint coming from the stabilizer of 0 and e
################################################################################

def get_matrix_kernel(A, eps=1e-10):
    '''
    Compute an orthonormal basis of the kernel (x_1, x_2, ...)
    A x_i = 0
    scalar_product(x_i, x_j) = delta_ij

    :param A: matrix
    :return: matrix where each row is a basis vector of the kernel of A
    '''
    _u, s, v = torch.svd(A)

    # A = u @ torch.diag(s) @ v.t()
    kernel = v.t()[s < eps]
    return kernel


def get_matrices_kernel(As, eps=1e-10):
    '''
    Computes the commun kernel of all the As matrices
    '''
    return get_matrix_kernel(torch.cat(As, dim=0), eps)


@cached_dirpklgz("%s/cache/trans_Q"%os.path.dirname(os.path.realpath(__file__)))
def _basis_transformation_Q_J(J, order_in, order_out, version=3):  # pylint: disable=W0613
    """
    :param J: order of the spherical harmonics
    :param order_in: order of the input representation
    :param order_out: order of the output representation
    :return: one part of the Q^-1 matrix of the article
    """
    with torch_default_dtype(torch.float64):
        def _R_tensor(a, b, c): return kron(irr_repr(order_out, a, b, c), irr_repr(order_in, a, b, c))

        def _sylvester_submatrix(J, a, b, c):
            ''' generate Kronecker product matrix for solving the Sylvester equation in subspace J '''
            R_tensor = _R_tensor(a, b, c)  # [m_out * m_in, m_out * m_in]
            R_irrep_J = irr_repr(J, a, b, c)  # [m, m]
            return kron(R_tensor, torch.eye(R_irrep_J.size(0))) - \
                kron(torch.eye(R_tensor.size(0)), R_irrep_J.t())  # [(m_out * m_in) * m, (m_out * m_in) * m]

        random_angles = [
            [4.41301023, 5.56684102, 4.59384642],
            [4.93325116, 6.12697327, 4.14574096],
            [0.53878964, 4.09050444, 5.36539036],
            [2.16017393, 3.48835314, 5.55174441],
            [2.52385107, 0.2908958, 3.90040975]
        ]
        null_space = get_matrices_kernel([_sylvester_submatrix(J, a, b, c) for a, b, c in random_angles])
        assert null_space.size(0) == 1, null_space.size()  # unique subspace solution
        Q_J = null_space[0]  # [(m_out * m_in) * m]
        Q_J = Q_J.view((2 * order_out + 1) * (2 * order_in + 1), 2 * J + 1)  # [m_out * m_in, m]
        assert all(torch.allclose(_R_tensor(a, b, c) @ Q_J, Q_J @ irr_repr(J, a, b, c)) for a, b, c in torch.rand(4, 3))

    assert Q_J.dtype == torch.float64
    return Q_J  # [m_out * m_in, m]


def get_spherical_from_cartesian_torch(cartesian, divide_radius_by=1.0):

    ###################################################################################################################
    # ON ANGLE CONVENTION
    #
    # sh has following convention for angles:
    # :param theta: the colatitude / polar angle, ranging from 0(North Pole, (X, Y, Z) = (0, 0, 1)) to pi(South Pole, (X, Y, Z) = (0, 0, -1)).
    # :param phi: the longitude / azimuthal angle, ranging from 0 to 2 pi.
    #
    # the 3D steerable CNN code therefore (probably) has the following convention for alpha and beta:
    # beta = pi - theta; ranging from 0(South Pole, (X, Y, Z) = (0, 0, -1)) to pi(North Pole, (X, Y, Z) = (0, 0, 1)).
    # alpha = phi
    #
    ###################################################################################################################

    # initialise return array
    # ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    spherical = torch.zeros_like(cartesian)

    # indices for return array
    ind_radius = 0
    ind_alpha = 1
    ind_beta = 2

    cartesian_x = 2
    cartesian_y = 0
    cartesian_z = 1

    # get projected radius in xy plane
    # xy = xyz[:,0]**2 + xyz[:,1]**2
    r_xy = cartesian[..., cartesian_x] ** 2 + cartesian[..., cartesian_y] ** 2

    # get second angle
    # version 'elevation angle defined from Z-axis down'
    spherical[..., ind_beta] = torch.atan2(torch.sqrt(r_xy), cartesian[..., cartesian_z])
    # ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2])
    # version 'elevation angle defined from XY-plane up'
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy))
    # spherical[:, ind_beta] = np.arctan2(cartesian[:, 2], np.sqrt(r_xy))

    # get angle in x-y plane
    spherical[...,ind_alpha] = torch.atan2(cartesian[...,cartesian_y], cartesian[...,cartesian_x])

    # get overall radius
    # ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    if divide_radius_by == 1.0:
        spherical[..., ind_radius] = torch.sqrt(r_xy + cartesian[...,cartesian_z]**2)
    else:
        spherical[..., ind_radius] = torch.sqrt(r_xy + cartesian[...,cartesian_z]**2)/divide_radius_by

    return spherical


def get_spherical_from_cartesian(cartesian):

    ###################################################################################################################
    # ON ANGLE CONVENTION
    #
    # sh has following convention for angles:
    # :param theta: the colatitude / polar angle, ranging from 0(North Pole, (X, Y, Z) = (0, 0, 1)) to pi(South Pole, (X, Y, Z) = (0, 0, -1)).
    # :param phi: the longitude / azimuthal angle, ranging from 0 to 2 pi.
    #
    # the 3D steerable CNN code therefore (probably) has the following convention for alpha and beta:
    # beta = pi - theta; ranging from 0(South Pole, (X, Y, Z) = (0, 0, -1)) to pi(North Pole, (X, Y, Z) = (0, 0, 1)).
    # alpha = phi
    #
    ###################################################################################################################

    if torch.is_tensor(cartesian):
        cartesian = np.array(cartesian.cpu())

    # initialise return array
    # ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    spherical = np.zeros(cartesian.shape)

    # indices for return array
    ind_radius = 0
    ind_alpha = 1
    ind_beta = 2

    cartesian_x = 2
    cartesian_y = 0
    cartesian_z = 1

    # get projected radius in xy plane
    # xy = xyz[:,0]**2 + xyz[:,1]**2
    r_xy = cartesian[..., cartesian_x] ** 2 + cartesian[..., cartesian_y] ** 2

    # get overall radius
    # ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    spherical[..., ind_radius] = np.sqrt(r_xy + cartesian[...,cartesian_z]**2)

    # get second angle
    # version 'elevation angle defined from Z-axis down'
    spherical[..., ind_beta] = np.arctan2(np.sqrt(r_xy), cartesian[..., cartesian_z])
    # ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2])
    # version 'elevation angle defined from XY-plane up'
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy))
    # spherical[:, ind_beta] = np.arctan2(cartesian[:, 2], np.sqrt(r_xy))

    # get angle in x-y plane
    spherical[...,ind_alpha] = np.arctan2(cartesian[...,cartesian_y], cartesian[...,cartesian_x])

    return spherical

def test_coordinate_conversion():
    p = np.array([0, 0, -1])
    expected = np.array([1, 0, 0])
    assert get_spherical_from_cartesian(p) == expected
    return True


def spherical_harmonics(order, alpha, beta, dtype=None):
    """
    spherical harmonics
    - compatible with irr_repr and compose

    computation time: excecuting 1000 times with array length 1 took 0.29 seconds;
    executing it once with array of length 1000 took 0.0022 seconds
    """
    #Y = [tesseral_harmonics(order, m, theta=math.pi - beta, phi=alpha) for m in range(-order, order + 1)]
    #Y = torch.stack(Y, -1)
    # Y should have dimension 2*order + 1
    return SphericalHarmonics.get(order, theta=math.pi-beta, phi=alpha) 

def kron(a, b):
    """
    A part of the pylabyk library: numpytorch.py at https://github.com/yulkang/pylabyk

    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    return res.reshape(siz0 + siz1)


def get_maximum_order_unary_only(per_layer_orders_and_multiplicities):
    """
    determine what spherical harmonics we need to pre-compute. if we have the
    unary term only, we need to compare all adjacent layers

    the spherical harmonics function depends on J (irrep order) purely, which is dedfined by
    order_irreps = list(range(abs(order_in - order_out), order_in + order_out + 1))
    simplification: we only care about the maximum (in some circumstances that means we calculate a few lower
    order spherical harmonics which we won't actually need)

    :param per_layer_orders_and_multiplicities: nested list of lists of 2-tuples
    :return: integer indicating maximum order J
    """

    n_layers = len(per_layer_orders_and_multiplicities)

    # extract orders only
    per_layer_orders = []
    for i in range(n_layers):
        cur = per_layer_orders_and_multiplicities[i]
        cur = [o for (m, o) in cur]
        per_layer_orders.append(cur)

    track_max = 0
    # compare two (adjacent) layers at a time
    for i in range(n_layers - 1):
        cur = per_layer_orders[i]
        nex = per_layer_orders[i + 1]
        track_max = max(max(cur) + max(nex), track_max)

    return track_max


def get_maximum_order_with_pairwise(per_layer_orders_and_multiplicities):
    """
    determine what spherical harmonics we need to pre-compute. for pairwise
    interactions, this will just be twice the maximum order

    the spherical harmonics function depends on J (irrep order) purely, which is defined by
    order_irreps = list(range(abs(order_in - order_out), order_in + order_out + 1))
    simplification: we only care about the maximum (in some circumstances that means we calculate a few lower
    order spherical harmonics which we won't actually need)

    :param per_layer_orders_and_multiplicities: nested list of lists of 2-tuples
    :return: integer indicating maximum order J
    """

    n_layers = len(per_layer_orders_and_multiplicities)

    track_max = 0
    for i in range(n_layers):
        cur = per_layer_orders_and_multiplicities[i]
        # extract orders only
        orders = [o for (m, o) in cur]
        track_max = max(track_max, max(orders))

    return 2*track_max


def precompute_sh(r_ij, max_J):
    """
    pre-comput spherical harmonics up to order max_J

    :param r_ij: relative positions
    :param max_J: maximum order used in entire network
    :return: dict where each entry has shape [B,N,K,2J+1]
    """
    
    i_distance = 0
    i_alpha = 1
    i_beta = 2

    Y_Js = {}
    sh = SphericalHarmonics()

    for J in range(max_J+1):
        # dimension [B,N,K,2J+1]
        #Y_Js[J] = spherical_harmonics(order=J, alpha=r_ij[...,i_alpha], beta=r_ij[...,i_beta])
        Y_Js[J] = sh.get(J, theta=math.pi-r_ij[...,i_beta], phi=r_ij[...,i_alpha], refresh=False)

    sh.clear()
    return Y_Js


class ScalarActivation3rdDim(torch.nn.Module):
    def __init__(self, n_dim, activation, bias=True):
        '''
        Can be used only with scalar fields [B, N, s] on last dimension

        :param n_dim: number of scalar fields to apply activation to
        :param bool bias: add a bias before the applying the activation
        '''
        super().__init__()

        self.activation = activation

        if bias and n_dim > 0:
            self.bias = torch.nn.Parameter(torch.zeros(n_dim))
        else:
            self.bias = None

    def forward(self, input):
        '''
        :param input: [B, N, s]
        '''

        assert len(np.array(input.shape)) == 3

        if self.bias is not None:
            x = input + self.bias.view(1, 1, -1)
        else:
            x = input
        x = self.activation(x)

        return x
