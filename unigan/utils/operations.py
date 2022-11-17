
import torch
from torch import autograd
from torch.nn.functional import one_hot
from custom_pkg.basic.metrics import EMA
from custom_pkg.pytorch.operations import api_empty_cache


########################################################################################################################
# Squeeze
########################################################################################################################

def squeeze_nc(x, s=2):
    """
    :param x: (n, c, h, w)
    :param s:
    :return: (n, c*(s**2), h//s, w//s)
    """
    if s == 1: return x
    # Squeeze
    b_size, input_nc, height, width = x.shape
    # (n, c, h//s, s, w//s, s) -> (n, c, s, s, h//s, w//s)
    x = x.view(b_size, input_nc, height//s, s, width//s, s).permute(0, 1, 3, 5, 2, 4)
    # (n, c, s, s, h//s, w//s) -> (n, c*(s**2), h//s, w//s)
    x = x.contiguous().view(b_size, input_nc*s**2, height//s, width//s)
    # Return
    return x


def unsqueeze_nc(x, s=2):
    """
    :param x: (n, c, h, w)
    :param s:
    :return: (n, c//(s**2), h*s, w*s)
    """
    if s == 1: return x
    # Unsqueeze
    b_size, input_nc, height, width = x.shape
    # (n, c//(s**2), s, s, h, w) -> (n, c//(s**2), h, s, w, s)
    x = x.view(b_size, input_nc//s**2, s, s, height, width).permute(0, 1, 4, 2, 5, 3)
    # (n, c//(s**2), h, s, w, s) -> (n, c//(s**2), h*s, w*s)
    x = x.contiguous().view(b_size, input_nc//s**2, height*s, width*s)
    # Return
    return x


########################################################################################################################
# Calculations.
########################################################################################################################

def batch_diag(x):
    """
    :param x: (batch, n, n)
    :return: (batch, n)
    """
    n = x.size(1)
    indices = torch.arange(n**2, dtype=torch.int64, device=x.device).reshape(n, n)
    indices = torch.diag(indices)
    return torch.index_select(x.reshape(x.size(0), -1), dim=1, index=indices)


def l2_normalization(x):
    sizes = x.size()[1:]
    # 1. Reshape.
    x = x.reshape(x.size(0), -1)
    # 2. Normalize.
    x = x / ((x**2).sum(dim=1, keepdim=True).sqrt() + 1e-8)
    # 3. Reshape.
    x = x.reshape(x.size(0), *sizes)
    # Return
    return x


def normalized_mean_absolute_err(pred, gt):
    """ nMAE. The first axis should be batch. """
    pred, gt = map(lambda _x: _x.reshape(_x.size(0), -1), [pred, gt])
    # 1. Calculate Mean Absolute Err (MAE).
    mae = (pred - gt).abs().mean(1)
    # 2. Normalize.
    norm = gt.abs().mean(1)
    # Return
    return mae / (norm+1e-8)


def measure_ortho(matrix):
    """
    :param matrix: (batch, n, n)
    :return: Orthogonality significance. (batch, )
    """
    indicator = torch.ones_like(matrix) == torch.eye(matrix.size(1), dtype=matrix.dtype, device=matrix.device).unsqueeze(0)
    elem_diag, elem_other = map(lambda _i: matrix[_i].reshape(matrix.size(0), -1), [indicator, ~indicator])
    # Get orthogonality significance = magnitude(other_locations) / magnitude(diagonals). (batch, )
    mag_diag = elem_diag.abs().mean(dim=1)
    mag_other = elem_other.abs().mean(dim=1)
    ortho_sign = mag_other / (mag_diag + 1e-8)
    # Return
    return ortho_sign.cpu().numpy()


########################################################################################################################
# Autograd
########################################################################################################################

def autograd_proc(eps, ipt, opt, create_graph=False):
    """ Autograd procedure involved in the fast approximation of spectral norm.
    :param eps: (output_shape). Random vector.
    :param ipt: (input_shape). Input to the module whose Jacobian is targeted.
    :param opt: (output_shape). Output of the module.
    :param create_graph:
    return: (input_shape). A vector computed as J^\top(ipt)*eps.
    """
    y = (opt * eps).sum()
    grads = autograd.grad(outputs=y, inputs=ipt, grad_outputs=torch.ones_like(y), create_graph=create_graph, retain_graph=True)[0]
    # Return
    return grads if create_graph else grads.detach()


@api_empty_cache
def autograd_jacob(x, func, bsize, x_clip=None, y_clip=None):
    """
    :param x: (n, ...)
    :param func: A mapping x -> y.
    :param bsize:
    :param x_clip: indices.
    :param y_clip: indices.
    :return: The Jacobian matrix. (n, ny, nx)
    """
    ####################################################################################################################
    # 1. Get x (n, bsize, ...) & y (n, bsize, ny')
    ####################################################################################################################
    x = x.unsqueeze(dim=1).expand(x.size(0), bsize, *x.size()[1:]).requires_grad_(True)
    y = func(x.reshape(-1, *x.size()[2:])).reshape(x.size(0), bsize, -1)
    if y_clip is not None: y = y[:, :, y_clip]
    ####################################################################################################################
    # 2. Calculate.
    ####################################################################################################################
    # (1) Init results.
    results, counts, max_counts = [], 0, y.size(2)
    # (2) Each batch.
    while counts < max_counts:
        # 1. Select outputs.
        batch_size = min(bsize, max_counts-counts)
        outputs = batch_diag(y[:, 0:batch_size, counts:counts+batch_size])
        # 2. Calculate grads. (n, bsize, ...) -> (n, batch_size, -1)
        grads = autograd.grad(outputs=outputs, inputs=x, grad_outputs=torch.ones_like(outputs), create_graph=False, retain_graph=True)[0]
        grads = grads[:, 0:batch_size].reshape(grads.size(0), batch_size, -1)
        if x_clip is not None: grads = grads[:, :, x_clip]
        # Save
        results.append(grads.detach())
        # Move forward
        counts += batch_size
    ####################################################################################################################
    # 3. Get Jacobian matrix. (n, ny, nx)
    return torch.cat(results, dim=1)


def fast_approx_logsn(x, output, x_t, output_t, sn_power=3):
    """ The fast approximation algorithm. """
    # 1. Init u & v.
    u, v = torch.randn_like(output), None
    # 2. Approximation iterations.
    for index in range(sn_power):
        # (1) Update v.
        v = autograd_proc(eps=l2_normalization(u), ipt=x, opt=output)
        # (2) Update u.
        u = autograd_proc(eps=l2_normalization(v), ipt=x_t, opt=output_t, create_graph=True if index==sn_power-1 else False)
    # 3. Get result.
    logsn = (u.reshape(u.size(0), -1)**2).sum(dim=1).log() * 0.5
    # Return
    return logsn


class EMAMultiClass(EMA):
    """
    Multi-class EMA.
    """
    def __init__(self, n_classes, beta, init=None):
        super(EMAMultiClass, self).__init__(beta, init)
        # Config.
        self._n_classes = n_classes

    def update_average(self, new):
        """
        :param new: A tuple containing logsv (batch, ) & cat (batch, ), namely the logsv values and the associated categories.
        :return:
        """
        logsv, cat = new[0].detach(), new[1]
        # --------------------------------------------------------------------------------------------------------------
        # 1. Get batch avg (n_classes, ) & cat mask (n_classes, ).
        # --------------------------------------------------------------------------------------------------------------
        # Get cat onehot (batch, n_classes) & cat accumulate number (n_classe, ).
        cat_onehot = one_hot(cat, self._n_classes)
        cat_accum = cat_onehot.sum(dim=0)
        # Get batch avg (n_classes, ) & cat mask (n_classe, ).
        batch_avg = ((logsv.unsqueeze(1)*cat_onehot).sum(dim=0) + 1e-30) / cat_accum
        cat_mask = cat_accum > 0
        # --------------------------------------------------------------------------------------------------------------
        # 2. Update stale
        # --------------------------------------------------------------------------------------------------------------
        if self._stale is not None:
            self._stale[cat_mask] = self._beta*self._stale[cat_mask] + (1.0-self._beta)*batch_avg[cat_mask]
        else:
            self._stale = batch_avg
        # Return
        return self._stale
