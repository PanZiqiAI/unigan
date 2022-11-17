
import math
import torch
from torch import nn
from torch.nn import init
from ..basic.operations import chk_ns, BatchSlicerInt


########################################################################################################################
# Data
########################################################################################################################

class DataCycle(object):
    """
    Data cycle infinitely. Using next(self) to fetch batch data.
    """
    def __init__(self, dataloader):
        # Dataloader
        self._dataloader = dataloader
        # Iterator
        self._data_iterator = iter(self._dataloader)
        
    @property
    def dataset(self):
        return self._dataloader.dataset

    @property
    def num_samples(self):
        return len(self._dataloader.dataset)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._data_iterator)
        except StopIteration:
            self._data_iterator = iter(self._dataloader)
            return next(self._data_iterator)


########################################################################################################################
# Networks
########################################################################################################################

def network_param_m(network):
    return sum([param.numel() for param in network.parameters()]) / 1e6


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.
    Parameters:
        net (network)       -- network to be initialized
        init_type (str)     -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)   -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            # Weight
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            # Bias
            if chk_ns(m, 'bias', 'is not', None):
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def collect_weight_keys(module, prefix=None, destination=None):
    """
    Collecting a module's weight (linear & conv).
    """
    # Check compatibility
    if destination is None:
        assert prefix is None
        destination = []
    else:
        assert prefix is not None
    # 1. For module is an instance of nn.Linear or nn.Conv
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        # Get weight
        state_dict = module.state_dict()
        weight_key = list(filter(lambda k: k.endswith("weight"), state_dict.keys()))
        assert len(weight_key) == 1
        weight_key = weight_key[0]
        # Save to result & return
        destination.append(("%s.%s" % (prefix, weight_key)) if prefix is not None else weight_key)
        return destination
    # 2. Recursive
    else:
        for key, sub_module in module._modules.items():
            destination = collect_weight_keys(
                sub_module, prefix=("%s.%s" % (prefix, key)) if prefix is not None else key,
                destination=destination)
        return destination


def fix_grad(net):
    """
    Fix gradient.
    :param net:
    :return:
    """
    if isinstance(net, list):
        for layer in net:
            for p in layer.parameters():
                p.requires_grad = False
    else:
        for p in net.parameters():
            p.requires_grad = False


def set_requires_grad(nets, requires_grad=False, **kwargs):
    """
    :param nets: a list of networks
    :param requires_grad: whether the networks require gradients or not
    :param kwargs:
        - modes: a list of targeted mode
    """
    # Normalize input
    if not isinstance(nets, list): nets = [nets]
    if not isinstance(requires_grad, list): requires_grad = [requires_grad] * len(nets)
    if 'modes' in kwargs.keys() and not isinstance(kwargs['modes'], list): kwargs['modes'] = [kwargs['modes']] * len(nets)
    # Deploy
    for index, (n, rg) in enumerate(zip(nets, requires_grad)):
        if n is not None:
            # Gradients requiring
            for param in n.parameters(): param.requires_grad = rg
            # Others
            if 'modes' in kwargs.keys(): n.train() if kwargs['modes'][index] == 'train' else n.eval()


class EvalMode(object):
    """
    Turn network into eval mode when enter & reverse to the original mode when exiting.
    """

    def __init__(self, *nets):
        self._nets = nets
        self._training_indices = list(filter(lambda _index: self._nets[_index].training, range(len(self._nets))))

    def __enter__(self):
        for n in self._nets: n.eval()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for index in self._training_indices:
            self._nets[index].train()


class RequiresGrad(object):
    """
    Set parameters' requires_grad when enter & reverse to the original mode when exiting.
    """

    def __init__(self, nets, requires_grad=False):
        # 1. Config
        self._nets = nets if isinstance(nets, list) else [nets]
        self._req_grad = requires_grad if isinstance(requires_grad, list) else [requires_grad] * len(self._nets)
        # 2. Params to-be-reverted
        self._tbr_params = []

    def __enter__(self):
        for n_index, (net, rg) in enumerate(zip(self._nets, self._req_grad)):
            for p in net.parameters():
                if p.requires_grad != rg:
                    p.requires_grad = rg
                    self._tbr_params.append(p)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for p in self._tbr_params:
            p.requires_grad = not p.requires_grad


def api_empty_cache(func):
    @wraps(func)
    def _wrapped(*args, **kwargs):
        torch.cuda.empty_cache()
        ret = func(*args, **kwargs)
        torch.cuda.empty_cache()
        return ret

    return _wrapped


########################################################################################################################
# Criterion
########################################################################################################################

# ======================================================================================================================
# Typical usage:
#    Set lambda to positive if you want to both check(show in log) & back_prop.
#    Set lambda to 0 if you want to totally disable the loss, \ie, neither check nor back_prop.
#    Set lambda to negative (-1) if you want to only check (show in log).
# ----------------------------------------------------------------------------------------------------------------------

class TensorWrapper(object):
    """
    Tensor wrapper.
    """
    def __init__(self, tensor):
        self._tensor = tensor
        
    @property
    def tensor(self):
        return self._tensor

    def item(self):
        return None if self._tensor is None else self._tensor.item()


class LossWrapper(TensorWrapper):
    """
    Loss wrapper.
    """
    def __init__(self, _lmd, loss_tensor):
        super(LossWrapper, self).__init__(loss_tensor)
        # Lambda
        self._lmd = _lmd

    def loss_backprop(self):
        if self._lmd.hyper_param > 0.0 and self._tensor is not None:
            return self._lmd(self._tensor) * self._lmd.hyper_param
        else:
            return None


def summarize_losses_and_backward(*args, **kwargs):
    """
    Each arg should either be instance of
        - None
        - Tensor
        - LossWrapper
        - LossWrapperContainer
    """
    # 1. Init
    ret = 0.0
    # 2. Summarize to result
    for arg in args:
        if arg is None:
            continue
        elif isinstance(arg, LossWrapper):
            loss_backprop = arg.loss_backprop()
            if loss_backprop is not None: ret += loss_backprop
        elif isinstance(arg, torch.Tensor):
            ret += arg
        else:
            raise NotImplementedError
    if 'weight' in kwargs.keys(): ret = ret * kwargs.pop('weight')
    # 3. Backward
    if isinstance(ret, torch.Tensor):
        ret.backward(**kwargs)


def wraps(hyper_param):

    def _set_lmd(_lmd):
        setattr(_lmd, 'hyper_param', hyper_param)
        return _lmd

    return _set_lmd


class BaseCriterion(object):
    """
    Base criterion class.
    """
    def __init__(self, lmd=None):
        """
        Dynamic lambda if given is None.
        """
        # Config
        self._lmd = self._get_lmd(lmd) if lmd is not None else None

    @staticmethod
    def _get_lmd(_lmd):
        """
        _lmd:
            float: Will be reformed to a function with attribute "hyper_param".
            Function: If has not attribute "hyper_param", the attr will be set.
        Return: A function that maps original_tensor (produced by _call_method) to the tensor for backward (without
        multiplied the hyper_param).
        """
        def __get_single_lmd(_l):
            if isinstance(_l, float):
                _l = wraps(hyper_param=_l)(lambda x: x)
            else:
                if not hasattr(_l, "hyper_param"):
                    _l = wraps(hyper_param=1.0)(_l)
            # Return
            return _l

        # Single
        if not isinstance(_lmd, dict):
            return __get_single_lmd(_lmd)
        else:
            return {k: __get_single_lmd(_lmd[k]) for k in _lmd.keys()}

    def _call_method(self, *args, **kwargs):
        """
        For lambda is a number, return the corresponding loss tensor.
        For lambda is a dict, return a loss tensor dict that corresponds the lambda dict.
        """
        raise NotImplementedError

    @staticmethod
    def _get_loss_wrappers(_lmd, loss_tensor=None):
        # 1. For single lambda
        if not isinstance(_lmd, dict):
            # (1) Shared lambda - dict
            if isinstance(loss_tensor, dict):
                return {key: LossWrapper(_lmd, value) for key, value in loss_tensor.items()}
            # (2) Shared lambda - list
            elif isinstance(loss_tensor, list):
                return [LossWrapper(_lmd, tensor) for tensor in loss_tensor]
            # (3) Single tensor.
            else:
                return LossWrapper(_lmd, loss_tensor)
        # 2. For multi lambda
        else:
            return {key: LossWrapper(_lmd[key], loss_tensor[key] if loss_tensor is not None else None)
                    for key in _lmd.keys()}

    @staticmethod
    def _need_calculate(_lmd):
        # 1. Lmd is a scalar
        if not isinstance(_lmd, dict):
            if _lmd.hyper_param == 0.0:
                return False
            else:
                return True
        # 2. Multi lmd
        else:
            for _lmd in _lmd.values():
                if _lmd.hyper_param != 0.0:
                    return True
            return False

    def __call__(self, *args, **kwargs):
        # Get lambda & Check if not calculate
        if 'lmd' in kwargs.keys():
            _lmd = self._get_lmd(kwargs.pop('lmd'))
        else:
            assert self._lmd is not None
            _lmd = self._lmd
        if not self._need_calculate(_lmd): return self._get_loss_wrappers(_lmd)
        # 1. Calculate
        call_result = self._call_method(*args, **kwargs)
        # 2. Return
        if isinstance(call_result, tuple):
            loss_tensor, others = call_result
            return self._get_loss_wrappers(_lmd, loss_tensor), others
        else:
            return self._get_loss_wrappers(_lmd, call_result)

# ======================================================================================================================


class KLDivLoss(BaseCriterion):
    """
    KL Divergence loss.
    """
    def __init__(self, lmd=None, **random_kwargs):
        super(KLDivLoss, self).__init__(lmd=lmd)
        # Config.
        self._random_kwargs = random_kwargs

    def _call_method(self, z):
        if self._random_kwargs['random_type'] == 'gauss':
            mu, logvar = z
            return - 0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(1).mean()
        elif self._random_kwargs['random_type'] == 'uni':
            return torch.relu(z.abs() - self._random_kwargs['random_uni_radius']).sum(1).mean()
        else: raise ValueError


########################################################################################################################
# Calculations
########################################################################################################################

def nanmean(x):
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum()
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum()
    return value / num


def sampling_z(batch_size, nz, device, random_type, **kwargs):
    if random_type == 'uni':
        # [-1, 1]
        z = torch.rand(batch_size, nz).to(device) * 2.0 - 1.0
        # [-radius, radius]
        if 'random_uni_radius' in kwargs.keys():
            z = z * kwargs['random_uni_radius']
        # [-radius + c, radius + c]
        if 'random_uni_center' in kwargs.keys():
            z = z + kwargs['random_uni_center']
    elif random_type == 'gauss':
        z = torch.randn(batch_size, nz).to(device)
    else:
        raise NotImplementedError
    return z


def reparameterize(mu, logvar, n_samples=1, squeeze=True):
    """
    :param mu: (batch, nz)
    :param logvar: (batch, nz)
    :param n_samples:
    :param squeeze:
    :return:
        - For squeeze=True and n_samples=1: (batch, nz)
        - Otherwise: (batch, n_samples, nz)
    """
    mu, logvar = mu.unsqueeze(1), logvar.unsqueeze(1)
    ret = torch.randn(mu.size(0), n_samples, mu.size(1), dtype=mu.dtype, device=mu.device)*(0.5*logvar).exp() + mu
    # Return
    if squeeze: return ret.squeeze(1) if n_samples == 1 else ret.reshape(-1, ret.size(2))
    return ret


def gaussian_log_density_marginal(sample, params, mesh=False):
    """
    Estimate Gaussian log densities: - 1/2 * { dim*log(2*pi) + log det(SIGMA) + (x-u)'.det(SIGMA^{-1}).(x-u) }
        - For not mesh:   log p(sample_i|params_i), i in [batch]
        - Otherwise:      log p(sample_i|params_j), i in [num_samples], j in [num_params]
    :param sample: (num_samples, dims)
    :param params: mu, logvar. Each is (num_params, dims)
    :param mesh:
    :return:
        - For not mesh: (num_sample, dims)
        - Otherwise: (num_sample, num_params, dims)
    """

    def _process_param(_p):
        if isinstance(_p, float):
            assert not mesh
            return torch.ones_like(sample) * _p
        return _p

    # Get data
    mu, logvar = map(_process_param, params)
    # Mesh
    if mesh:
        sample = sample.unsqueeze(1)
        mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
    # 1. Calculate
    constant = math.log(2*math.pi)
    dev_invstd_dev = ((sample - mu) / (0.5*logvar).exp()) ** 2
    # 2. Get result
    log_prob_marginal = - 0.5 * (constant + logvar + dev_invstd_dev)
    # Return
    return log_prob_marginal


def gaussian_cross_entropy_marginal(params1, params2):
    """
    Analytical form is:
        0.5 * {
            log det(SIGMA_2) +                          = sum_j=1^K (logvar2_j)
            Tr( SIGMA_2^{-1} * SIGMA_1 ) +              = sum_j=1^K (sig1_j / sig2_j)^2
            (u_2-u_1)^T * SIGMA_2^{-1} * (u_2-u_1) +    = sum_j=1^K [(u2_j-u1_j) / sig2_j]^2
            K * log(2*pi)
        }
    :param params1: mu, logvar. Each is (num_params, dims)
    :param params2:
    :return: (batch, dims)
    """
    mu1, logvar1 = params1
    if params2 is None: mu2, logvar2 = torch.zeros_like(mu1), torch.ones_like(logvar1)
    elif params2 == 'entropy': mu2, logvar2 = mu1, logvar1
    elif isinstance(params2, torch.Tensor): mu2, logvar2 = params2
    else: raise ValueError
    # 1. Calculate.
    # (1) log det(SIGMA_2)
    log_det_sigma2 = logvar2
    # (2) Tr( SIGMA_2^{-1} * SIGMA_1 )
    tr = logvar1.exp() / logvar2.exp()
    # (3) (u_2-u_1)^T * SIGMA_2^{-1} * (u_2-u_1)
    dev_inv_sigma2_dev = ((mu2 - mu1) / (0.5*logvar2).exp()) ** 2
    # (4) Constant
    c = math.log(2*math.pi)
    # Get results.
    r = 0.5 * (log_det_sigma2 + tr + dev_inv_sigma2_dev + c)
    # Return
    return r


def gaussian_kl_div_marginal(params1, params2):
    return gaussian_cross_entropy_marginal(params1, params2) - gaussian_cross_entropy_marginal(params1, 'entropy')


# ----------------------------------------------------------------------------------------------------------------------
# Others
# ----------------------------------------------------------------------------------------------------------------------

def triu_vec(matrix, diagonal=1):
    return matrix[torch.ones(*matrix.size(), dtype=torch.int, device=matrix.device).triu(diagonal) == 1]


class TriuCollector(object):
    """
    Collecting triu matrix.
    """
    def __init__(self, n, batch_size, device, **kwargs):
        # Config
        self._batch_sizes = [_bs for _bs in BatchSlicerInt(n, batch_size)]
        self._prefix_batch = kwargs['prefix_batch'] if 'prefix_batch' in kwargs.keys() else None
        # Data
        # (1) Inidices
        self._indices_matrix = torch.arange(n**2, dtype=torch.int64, device=device).reshape(n, n)
        # (2) Results
        shape = (1 if self._prefix_batch is None else self._prefix_batch, n**2)
        self._results_vec = torch.zeros(*shape, dtype=torch.float32, device=device)

    def _batch_range(self, index):
        start = sum(self._batch_sizes[:index])
        end = sum(self._batch_sizes[:index+1])
        # Return
        return start, end

    def __call__(self, h_batch, w_batch, results):
        """
        :param h_batch:
        :param w_batch:
        :param results: (h, w)
        :return:
        """
        assert h_batch <= w_batch
        # 1. Get batch indices matrix. (h_size, w_size)
        h_range, w_range = map(self._batch_range, [h_batch, w_batch])
        indices_block = self._indices_matrix[h_range[0]:h_range[1], w_range[0]:w_range[1]]
        # 2. Push
        lmd_r = lambda _x: triu_vec(_x, diagonal=0) if h_batch == w_batch else _x.reshape(-1, )
        indices_block, results = lmd_r(indices_block), lmd_r(results).reshape(len(results), -1)
        self._results_vec[:, indices_block] = results

    def results(self):
        size = len(self._indices_matrix)
        # 1. Get triu matrix
        matrix = self._results_vec.reshape(-1, size, size)
        # 2. Get all matrix
        ret = matrix + matrix.triu(diagonal=1).transpose(1, 2)
        # Return
        return ret.squeeze(dim=0) if self._prefix_batch is None else ret
