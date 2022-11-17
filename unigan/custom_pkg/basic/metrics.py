
import time
import torch
import atexit
import numpy as np
from tqdm import tqdm
from functools import wraps
from sklearn.metrics import accuracy_score
from ..basic.operations import chk_d, TempKwargsManager, BatchSlicerLenObj


########################################################################################################################
# Meters
########################################################################################################################

class ResumableMeter(object):
    """
    Meter that is resumable.
    """
    def load(self, **kwargs):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError


class BestPerfMeter(ResumableMeter):
    """
    Meter to remember the best.
    """
    def __init__(self, iter_name, perf_name, lmd_ascend_perf=lambda new, stale: new > stale, early_stop_trials=-1):
        # Configuration
        self._iter_name, self._perf_name = iter_name, perf_name
        self._lmd_ascend_perf = lmd_ascend_perf
        self._early_stop_trials = early_stop_trials
        # Data
        self._best_iter = None
        self._best_perf = None
        self._trials_no_ascend = 0

    def load(self, **kwargs):
        self._best_iter = kwargs['best_%s' % self._iter_name]
        self._best_perf = kwargs['best_%s' % self._perf_name]
        self._trials_no_ascend = kwargs['trials_no_ascend']

    def save(self):
        return {
            'best_%s' % self._iter_name: self._best_iter, 'best_%s' % self._perf_name: self._best_perf,
            'trials_no_ascend': self._trials_no_ascend
        }

    @property
    def best_iter(self):
        return self._best_iter
    
    @property
    def best_perf(self):
        return self._best_perf

    @property
    def early_stop(self):
        if (self._early_stop_trials > 0) and (self._trials_no_ascend >= self._early_stop_trials):
            return True
        else:
            return False

    def update(self, iter_index, new_perf):
        # Better
        if self._best_perf is None or self._lmd_ascend_perf(new_perf, self._best_perf):
            # Ret current best iter as 'last_best_iter'
            ret = self._best_iter
            # Update
            self._best_iter = iter_index
            self._best_perf = new_perf
            self._trials_no_ascend = 0
            return ret
        # Update trials
        else:
            self._trials_no_ascend += 1
            return -1


class FreqCounter(ResumableMeter):
    """
    Handling frequency.
    """
    def __init__(self, freq, iter_fetcher=None):
        # Config
        self._freq = freq
        # 1. Values
        self._iter_last_triggered = -1
        self._status = False
        # 2. Iter fetcher
        if iter_fetcher is not None:
            self._iter_fetcher = iter_fetcher
            setattr(self, 'iter_fetcher', iter_fetcher)

    def load(self, **kwargs):
        self._iter_last_triggered, self._status = kwargs['iter_last_triggered'], kwargs['status']

    def save(self):
        return {'iter_last_triggered': self._iter_last_triggered, 'status': self._status}

    @property
    def status(self):
        return self._status

    def check(self, iteration=None, virtual=False):
        if self._freq <= 0: return False
        # Get iteration.
        if hasattr(self, '_iter_fetcher'):
            assert iteration is None
            iteration = self._iter_fetcher()
        # Update
        if (iteration+1)//self._freq > (self._iter_last_triggered+1)//self._freq:
            if virtual: return True
            self._iter_last_triggered = iteration
            self._status = True
        else:
            if virtual: return False
            self._status = False
        # Return
        return self._status


class TriggerLambda(ResumableMeter):
    """
    Triggered by a function.
    """
    def __init__(self, lmd_trigger, n_fetcher=None):
        # Config
        self._lmd_trigger = lmd_trigger
        # 1. Fetcher
        if n_fetcher is not None:
            self._n_fetcher = n_fetcher
            setattr(self, 'n_fetcher', n_fetcher)
        # 2. Status
        self._first = None

    def load(self, **kwargs):
        self._first = kwargs['first']

    def save(self):
        return {'first': self._first}
        
    @property
    def first(self):
        return self._first

    def check(self, n=None):
        if hasattr(self, '_n_fetcher'):
            assert n is None
            n = self._n_fetcher()
        # Check
        ret = self._lmd_trigger(n)
        if ret:
            if self._first is None: self._first = True
            else: self._first = False
        # Return
        return ret


class TriggerPeriod(ResumableMeter):
    """
    Trigger using period:
        For the example 'period=10, area=3', then 0,1,2 (valid), 3,4,5,6,7,8,9 (invalid).
        For the example 'period=10, area=-3', then 0,1,2,3,4,5,6 (invalid), 7,8,9 (valid).
    """
    def __init__(self, period, area):
        assert period >= 0 and period >= area
        # Get lambda & init
        self._lmd_trigger = (lambda n: n < area) if area >= 0 else (lambda n: n >= period + area)
        # Configs
        self._period = period
        # Data
        self._count = 0
        self._n_valid, self._n_invalid = 0, 0

    @property
    def n_valid(self):
        return self._n_valid

    @property
    def n_invalid(self):
        return self._n_invalid

    def load(self, **kwargs):
        self._count = kwargs['count']
        self._n_valid = kwargs['n_valid']
        self._n_invalid = kwargs['n_invalid']

    def save(self):
        return {'count': self._count, 'n_valid': self._n_valid, 'n_invalid': self._n_invalid}

    def check(self):
        # 1. Get return
        ret = self._lmd_trigger(self._count)
        # 2. Update counts
        if self._period != 0:
            self._count = (self._count + 1) % self._period
        if ret: self._n_valid += 1
        else: self._n_invalid += 1
        # Return
        return ret


class Pbar(tqdm, ResumableMeter):
    """
    Progress bar.
    """
    def __init__(self, *args, **kwargs):
        super(Pbar, self).__init__(*args, **kwargs, disable=None)
        # Setup start_t & last_print_n for disable=True
        if self.disable:
            self.start_t = time.time()
            self.last_print_n = self.n
        # Register upon exit
        atexit.register(lambda: self.close())

    def update(self, n=1):
        if self.disable:
            self.n += n
            self.last_print_n = self.n
            return
        else: super(Pbar, self).update(n)

    def load(self, **kwargs):
        self.start_t = time.time() - kwargs['elapsed']
        self.n = kwargs['n']
        self.last_print_n = kwargs['last_print_n']

    def save(self):
        return {'elapsed': time.time() - self.start_t, 'n': self.n, 'last_print_n': self.last_print_n}


# ----------------------------------------------------------------------------------------------------------------------
# Exponential Moving Average
# ----------------------------------------------------------------------------------------------------------------------

class EMA(ResumableMeter):
    """
    Exponential Moving Average.
    """
    def __init__(self, beta, init=None):
        super(EMA, self).__init__()
        # Config
        self._beta = beta
        # Data
        self._stale = init

    def load(self, **kwargs):
        self._stale = kwargs['avg']

    def save(self):
        return {'avg': self._stale}

    @property
    def avg(self):
        return self._stale

    def update_average(self, new):
        # Update stale
        if new is not None:
            self._stale = new if self._stale is None else \
                self._beta * self._stale + (1.0 - self._beta) * new
        # Return
        return self._stale


class EMAPyTorchModel(ResumableMeter):
    """
    Exponential Moving Average for PyTorch Model.
    """
    def __init__(self, beta, model, **kwargs):
        # Config
        self._beta = beta
        # 1. Data
        self._model, self._initialized = model, False
        # 2. Init
        if 'init' in kwargs.keys():
            self._model.load_state_dict(kwargs['init'].state_dict())
            self._initialized = True

    def load(self, **kwargs):
        self._model.load_state_dict(kwargs['avg_state_dict'])
        self._initialized = kwargs['initialized']

    def save(self):
        return {
            'avg_state_dict': self._model.state_dict(),
            'initialized': self._initialized
        }

    @property
    def initialized(self):
        return self._initialized

    @property
    def avg(self):
        return self._model

    def update_average(self, new):
        # Update stale
        if new is not None:
            # 1. Init
            if not self._initialized:
                self._model.load_state_dict(new.state_dict())
                self._initialized = True
            # 2. Moving average
            else:
                for stale_param, new_param in zip(self._model.parameters(), new.parameters()):
                    stale_param.data = self._beta * stale_param.data + (1.0 - self._beta) * new_param.data
        # Return
        return self._model


# ----------------------------------------------------------------------------------------------------------------------
# Timers
# ----------------------------------------------------------------------------------------------------------------------

class StopWatch(object):
    """
    Timer for recording durations.
    """
    def __init__(self):
        # Statistics - current
        self._stat = 'off'
        self._cur_duration = 0.0
        # Statistics - total
        self._total_duration = 0.0

    @property
    def stat(self):
        return self._stat

    def resume(self):
        # Record start time, switch to 'on'
        self._cur_duration = time.time()
        self._stat = 'on'

    def pause(self):
        if self._stat == 'off': return
        # Get current duration, switch to 'off'
        self._cur_duration = time.time() - self._cur_duration
        self._stat = 'off'
        # Update total duration
        self._total_duration += self._cur_duration

    def get_duration_and_reset(self):
        result = self._total_duration
        self._total_duration = 0.0
        return result


class _TimersManager(object):
    """
    Context manager for timers.
    """
    def __init__(self, timers, cache):
        # Config
        self._timers = timers
        self._cache = cache

    def __enter__(self):
        if self._cache is None: return
        # Activate
        for k in self._cache['on']:
            self._timers[k].resume()
        for k in self._cache['off']:
            self._timers[k].pause()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cache is None: return
        # Restore
        for k in self._cache['on']:
            self._timers[k].pause()
        for k in self._cache['off']:
            self._timers[k].resume()


class TimersController(object):
    """
    Controller for a bunch of timers.
    """
    def __init__(self, **kwargs):
        # Members
        self._timers = {}
        # Set timers
        for k, v in kwargs.items():
            self[k] = v

    def __contains__(self, key):
        return key in self._timers.keys()

    def __setitem__(self, key, val):
        assert key not in self._timers.keys() and isinstance(val, StopWatch) and val.stat == 'off'
        # Save
        self._timers[key] = val

    def __getitem__(self, key):
        return self._timers[key]

    def __call__(self, *args, **kwargs):
        # Calculate cache
        if not chk_d(kwargs, 'void'):
            # 1. On
            on = list(filter(lambda _k: _k in self._timers, args))
            for k in on: assert self._timers[k].stat == 'off'
            # 2. Off
            off = [k for k in filter(lambda _k: self._timers[_k].stat == 'on', self._timers.keys())]
            # Result
            cache = {'on': on, 'off': off}
        else:
            cache = None
        # Return
        return _TimersManager(self._timers, cache=cache)


########################################################################################################################
# Metrics
########################################################################################################################

def basic_stat(data, axis=-1, **kwargs):
    """
    :param data: np.array.
    :param axis:
    :param kwargs:
        - conf_percent:
        - minmax:
    :return:
    """
    # 1. Calculate avg & std.
    results = {'avg': data.mean(axis=axis), 'std': data.std(ddof=1, axis=axis)}
    # 2. Calculate CI.
    if 'conf_percent' in kwargs.keys():
        # (1) Get 'n'
        n = {90: 1.645, 95: 1.96, 99: 2.576}[kwargs['conf_percent']]
        # (2) Get interval
        results['interval'] = n * results['std']
    # 3. Min & max
    if 'minmax' in kwargs.keys() and kwargs['minmax']:
        results.update({'min': data.min(), 'max': data.max()})
    # Return
    return results


def mean_accuracy(global_gt, global_pred, num_classes=None):
    """
    Mean Accuracy for classification.
    :param global_gt: (N, )
    :param global_pred: (N, )
    :param num_classes: Int. Provided for avoiding inference.
    :return:
    """
    # Infer num_classes
    if num_classes is None: num_classes = len(set(global_gt))
    # (1) Init result
    mean_acc = 0
    classes_acc = []
    # (2) Process each class
    for i in range(num_classes):
        cur_indices = np.where(global_gt == i)[0]
        # For current class
        cur_acc = accuracy_score(global_gt[cur_indices], global_pred[cur_indices])
        # Add
        mean_acc += cur_acc
        classes_acc.append(cur_acc)
    # (3) Get result
    mean_acc = mean_acc / num_classes
    # Return
    return mean_acc, np.array(classes_acc)


def api_eval_torch(func):
    @wraps(func)
    def _api(*args, **kwargs):
        with TempKwargsManager(args[0], **kwargs):
            torch.cuda.empty_cache()
            ret = func(*args)
            torch.cuda.empty_cache()
            return ret

    return _api


# ----------------------------------------------------------------------------------------------------------------------
# Evaluating Disentanglement: Qualitative latent traversal.
# ----------------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def vis_latent_traversal(func_decoder, z, limit=3.0, n_traversals=10):
    """
    :param func_decoder: A mapping which takes z as input, and outputs reconstruction. (batch, C, H, W)
    :param z: (batch, nz)
    :param limit:
    :param n_traversals:
    :return: (batch, nz, 1+n_traversals, C, H, W), where '1' represents for reconstruction.
    """
    bsize, nz = z.size()
    # ------------------------------------------------------------------------------------------------------------------
    # Get reconstruction. (batch, C, H, W)
    # ------------------------------------------------------------------------------------------------------------------
    x_recon = func_decoder(z).cpu()
    # ------------------------------------------------------------------------------------------------------------------
    # Get latent traversal. (batch, nz, n_traversals, C, H, W)
    # ------------------------------------------------------------------------------------------------------------------
    # Interpolation. (n_traversals, )
    interp = torch.arange(-limit, limit+1e-8, step=2.0*limit/(n_traversals-1), device=z.device)
    # 1. Get interp z. (batch, nz, n_traverals, nz)
    z = z.unsqueeze(1).unsqueeze(1).expand(bsize, nz, n_traversals, nz)
    mask_interp = torch.eye(nz, dtype=z.dtype, device=z.device).unsqueeze(1).unsqueeze(0)
    z_interp = z * (1.0-mask_interp) + interp.unsqueeze(0).unsqueeze(0).unsqueeze(-1) * mask_interp
    # 2. Get decoded. (batch*nz,*n_traversals, C, H, W) -> (batch, nz, n_traversals, C, H, W)
    x_traverals = []
    for z in BatchSlicerLenObj(z_interp.reshape(-1, nz), batch_size=16):
        x_traverals.append(func_decoder(z).cpu())
    x_traverals = torch.cat(x_traverals)
    x_traverals = x_traverals.reshape(bsize, nz, n_traversals, *x_traverals.size()[1:])
    # Return
    return torch.cat([x_recon.unsqueeze(1).unsqueeze(1).expand(bsize, nz, 1, *x_recon.size()[1:]), x_traverals], dim=2)


def vis_latent_traversal_given_x(func_encoder, func_decoder, x_real, limit=3.0, n_traversals=10):
    """
    :param func_encoder: A mapping which takes x as input, and outputs z. (batch, nz)
    :param func_decoder: A mapping which takes z as input, and outputs reconstruction. (batch, C, H, W)
    :param x_real: (batch, C, H, W)
    :param limit:
    :param n_traversals:
    :return: (batch, nz, 2+n_traversals, C, H, W), where '2' represents for 'real' & 'reconstruction'.
    """
    # 1. Get (batch, nz, 1+n_traversals, C, H, W)
    z = func_encoder(x_real)
    latent_traversals = vis_latent_traversal(func_decoder, z, limit, n_traversals)
    # 2. Get (batch, nz, 2+n_traversals, C, H, W)
    ret = torch.cat([x_real.unsqueeze(1).unsqueeze(1).expand(z.size(0), z.size(1), 1, *x_real.size()[1:]).cpu(), latent_traversals], dim=2)
    # Return
    return ret
