
import torch
import numpy as np
from torch import nn
from scipy import linalg
from torch.nn import functional as F
from utils.operations import squeeze_nc, unsqueeze_nc
from custom_pkg.pytorch.operations import api_empty_cache


class TransposableModule(nn.Module):
    """
    Base class for linearized transposable modules.
    """
    def forward(self, *args, **kwargs):
        raise ValueError

    def linearized_transpose(self, eps):
        """ The Jacobian of this mapping equals J^\top, where J is the Jacobian of the forward mapping. """
        raise NotImplementedError


class InvertibleModule(TransposableModule):
    """
    Base class for invertible modules.
    """
    def inverse(self, *args, **kwargs):
        """ The inverted mapping of the forward mapping. """
        raise NotImplementedError

    def inverse_lt(self, eps):
        """ The Jacobian of this mapping equals J^-\top, where J is the Jacobian of the forward mapping. """
        raise NotImplementedError


########################################################################################################################
# Convolutions, normalization & nonlinearities.
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# Convolutions
# ----------------------------------------------------------------------------------------------------------------------

class InvConv2d1x1Fixed(InvertibleModule):
    """
    Invconv1x1 with fixed rotation matrix as the weight.
    """
    def __init__(self, input_nc, output_nc, **kwargs):
        super(InvConv2d1x1Fixed, self).__init__()
        # Config.
        nc = max(input_nc, output_nc)
        # For getting conv weight.
        if output_nc < input_nc: self._get_weight = lambda _w: _w[:output_nc].unsqueeze(-1).unsqueeze(-1)
        elif output_nc > input_nc: self._get_weight = lambda _w: _w[:, :input_nc].unsqueeze(-1).unsqueeze(-1)
        else: self._get_weight = lambda _w: _w.unsqueeze(-1).unsqueeze(-1)
        # --------------------------------------------------------------------------------------------------------------
        # Architecture
        # --------------------------------------------------------------------------------------------------------------
        if 'matrix_r' in kwargs.keys():
            matrix_r = kwargs['matrix_r']
            assert len(matrix_r.size()) == 2 and matrix_r.size(0) == matrix_r.size(1) == nc
        else: matrix_r = torch.from_numpy(linalg.qr(np.random.randn(nc, nc))[0].astype('float32'))
        """ Set weight. """
        self.register_buffer("_matrix_r", matrix_r)

    def forward(self, x):
        return F.conv2d(x, self._get_weight(self._matrix_r))

    def linearized_transpose(self, eps):
        return F.conv_transpose2d(eps, self._get_weight(self._matrix_r))

    def inverse(self, x):
        """ Only applicable in the case where output_nc >= input_nc, and we ASSUME that the given x is 'on manifold'. """
        # Get weight. (output_nc, input_nc, 1, 1)
        weight = self._get_weight(self._matrix_r)
        assert weight.size(0) >= weight.size(1)
        # Get output.
        output = F.conv2d(x, weight.transpose(0, 1))
        # Return
        return output

    def inverse_lt(self, eps):
        # Get weight. (output_nc, input_nc, 1, 1)
        weight = self._get_weight(self._matrix_r)
        assert weight.size(0) >= weight.size(1)
        # Get output.
        return F.conv_transpose2d(eps, weight.transpose(0, 1))


class ConvTranspose2d(TransposableModule):
    """
    Conv transpose layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ConvTranspose2d, self).__init__()
        # Architecture.
        self._convt = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self._convt(x)

    def linearized_transpose(self, eps):
        return F.conv2d(eps, self._convt.weight, stride=self._convt.stride, padding=self._convt.padding)


# ----------------------------------------------------------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------------------------------------------------------

class ActNorm(InvertibleModule):
    """ ActNorm. """
    def __init__(self, input_nc):
        super(ActNorm, self).__init__()
        # Parameters: mean & variance.
        self._param_loc = nn.Parameter(torch.zeros(1, input_nc, 1, 1))
        self._param_log_scale = nn.Parameter(torch.zeros(1, input_nc, 1, 1))

    def forward(self, x):
        return (x + self._param_loc) * (self._param_log_scale.exp() + 1e-8)

    def linearized_transpose(self, eps):
        return self.forward(eps)

    def inverse(self, x):
        return x / (self._param_log_scale.exp() + 1e-8) - self._param_loc

    def inverse_lt(self, eps):
        return self.inverse(eps)


# ----------------------------------------------------------------------------------------------------------------------
# Nonlinearities
# ----------------------------------------------------------------------------------------------------------------------

class Nonlinearity(InvertibleModule):
    """
    Element-wise activation.
    """
    def __init__(self):
        super(Nonlinearity, self).__init__()
        # Gradients buffer.
        self._grads, self._grads_inv = None, None

    @api_empty_cache
    def _lt(self, eps, grads):
        # --------------------------------------------------------------------------------------------------------------
        """ Given x (n_x, ...) and grads (n_grads, ...), in the case of n_x > n_grads, the grads are repeatedly used.
        Namely it should be that n_x=n_grads*n_repeat, so grads should be repeated first. """
        if len(eps) > len(grads):
            grads = grads.unsqueeze(1).expand(grads.size(0), len(eps)//len(grads), *grads.size()[1:]).reshape(*eps.size())
        # --------------------------------------------------------------------------------------------------------------
        # Calculate output
        output = eps * grads
        # Return
        return output

    def forward(self, x, linearize=False):
        raise NotImplementedError

    @api_empty_cache
    def linearized_transpose(self, eps):
        return self._lt(eps, grads=self._grads)

    def inverse(self, x, linearize=False):
        raise NotImplementedError

    @api_empty_cache
    def inverse_lt(self, eps):
        return self._lt(eps, grads=self._grads_inv)


class ReLU(Nonlinearity):
    """
    ReLU activation.
    """
    def forward(self, x, linearize=False):
        # Calculate output.
        output = torch.relu(x)
        """ Linearize """
        if linearize: self._grads = torch.gt(x, torch.zeros_like(x)).to(x.dtype).detach()
        # Return
        return output

    def inverse(self, x, linearize=False):
        raise AssertionError("ReLU cannot be inverted. ")


class Tanh(Nonlinearity):
    """
    Tanh activation.
    """
    def forward(self, x, linearize=False):
        # Calculate output.
        output = torch.tanh(x)
        """ Linearize """
        if linearize: self._grads = (1.0 - torch.tanh(x)**2).detach()
        # Return
        return output

    def inverse(self, x, linearize=False):
        # Calculate output.
        output = torch.atanh(x)
        """ Linearize """
        if linearize: self._grads_inv = (1.0 / (1.0 - x**2)).detach()
        # Return
        return output


########################################################################################################################
# Utils.
########################################################################################################################

class Squeeze(InvertibleModule):
    """
    Squeeze Fn.
    """
    def __init__(self, s=2):
        super(Squeeze, self).__init__()
        # Config.
        self._s = s

    def forward(self, x):
        return squeeze_nc(x, s=self._s)

    def linearized_transpose(self, eps):
        return unsqueeze_nc(eps, s=self._s)

    def inverse(self, x):
        return unsqueeze_nc(x, s=self._s)

    def inverse_lt(self, eps):
        return squeeze_nc(eps, s=self._s)


class Unsqueeze(InvertibleModule):
    """
    Unsqueeze Fn.
    """
    def __init__(self, s=2):
        super(Unsqueeze, self).__init__()
        # Config.
        self._s = s

    def forward(self, x):
        return unsqueeze_nc(x, s=self._s)

    def linearized_transpose(self, eps):
        return squeeze_nc(eps, s=self._s)

    def inverse(self, x):
        return squeeze_nc(x, s=self._s)

    def inverse_lt(self, eps):
        return unsqueeze_nc(eps, s=self._s)


class ConcatZero(InvertibleModule):
    """
    Concat zero Fn.
    """
    def forward(self, x):
        return torch.cat([x, torch.zeros_like(x)], dim=1)

    def linearized_transpose(self, eps):
        return eps.chunk(2, dim=1)[0]

    def inverse(self, x):
        """ ASSUME that the given x is 'on manifold'. """
        return x.chunk(2, dim=1)[0]

    def inverse_lt(self, eps):
        return torch.cat([eps, torch.zeros_like(eps)], dim=1)
