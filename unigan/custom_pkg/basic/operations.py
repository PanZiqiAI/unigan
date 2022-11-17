
import os
import math
import torch
import shutil
import inspect
import functools
import numpy as np
from collections import OrderedDict


########################################################################################################################
# Collectors & Containers
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# Storage & Fetching
# ----------------------------------------------------------------------------------------------------------------------

class IterCollector(object):
    """
    Collecting items.
    """
    def __init__(self):
        self._dict = {}

    def __getitem__(self, key):
        return self._dict[key]

    @property
    def dict(self):
        return self._dict

    def _collect_method(self, value, self_value):
        if isinstance(value, dict):
            return {
                k: self._collect_method(v, self_value[k] if self_value is not None and k in self_value.keys() else None)
                for k, v in value.items()
            }
        else:
            if self_value is None:
                return [value]
            else:
                assert isinstance(self_value, list)
                self_value.append(value)
                return self_value

    def collect(self, items):
        # Process each item
        for key, value in items.items():
            self_value = self._dict[key] if key in self._dict.keys() else None
            self._dict[key] = self._collect_method(value, self_value)

    def _pack_method(self, value, **kwargs):
        if isinstance(value, dict): return {k: self._pack_method(v, **kwargs) for k, v in value.items()}
        # Pack method.
        assert isinstance(value, list)
        # 1. Pack
        if 'pack' not in kwargs.keys(): kwargs['pack'] = 'np_cat'
        # Pack using different methods.
        if kwargs['pack'] == 'np_cat':
            value = np.concatenate(value, axis=0)
        elif kwargs['pack'] == 'torch_cat':
            value = torch.cat(value, dim=0)
        else:
            raise ValueError
        # 2. Reduction
        if 'reduction' in kwargs.keys():
            value = kwargs['reduction'](value)
        # Return
        return value

    def pack(self, **kwargs):
        # Init result
        result = {}
        # Process each item
        for key, value in self._dict.items():
            result[key] = self._pack_method(value, **kwargs)
        # Return
        return result


class ValidDict(OrderedDict):
    """
    Dict that doesn't tolerate None.
    """
    def __init__(self, *args, **cfg_kwargs):
        # Configs
        self._update_skip_none = chk_d(cfg_kwargs, 'update_skip_none')
        # Collect dicts
        _dict = {}
        for a in args:
            assert isinstance(a, dict) and len(set(a.keys()) - set(_dict.keys())) == len(a), "Duplicated keys. "
            _dict.update(a)
        # Initialize
        super(ValidDict, self).__init__(self._process_dict(_dict))

    def __getitem__(self, key):
        return None if key not in self.keys() else super(ValidDict, self).__getitem__(key)

    def __setitem__(self, key, value):
        if value is None: 
            if key in self.keys() and not self._update_skip_none: self.pop(key)
        else:
            super(ValidDict, self).__setitem__(key, value)

    @staticmethod
    def _process_dict(_dict):
        # Init ret
        ret = OrderedDict()
        # Process
        for k, v in _dict.items():
            if v is None: continue
            ret[k] = v
        # Return
        return ret

    def update(self, _dict, **cfg_kwargs):
        """
        :param _dict:
        :param cfg_kwargs:
            - skip_none: How to handle value that is None.
        """
        # Update
        for k, v in _dict.items():
            # Process None
            if v is None:
                # Processing None value
                if k in self.keys():
                    # Whether to skip
                    skip = self._update_skip_none
                    if 'skip_none' in cfg_kwargs.keys(): skip = cfg_kwargs['skip_none']
                    # Process
                    if not skip: self.pop(k)
                # Move to next
                continue
            # Update
            self[k] = v
        # Return
        return self


def fet_d(_dict, *args, **kwargs):
    """
    :param _dict:
    :param args:
    :param kwargs:
    :return:
        - policy_on_null: How to handle key that not exists.
        - pop: If pop out from container.

        - prefix: Prefix of keys to be fetched.
        - suffix: Suffix of keys to be fetched.

        - lambda processing keys:
            - remove
            - replace
            - lmd_k
        - lambda processing values:
            - lmd_v
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Preliminary
    # ------------------------------------------------------------------------------------------------------------------
    # 1. Policy
    policy_on_null = kwargs['policy_on_null'] if 'policy_on_null' in kwargs.keys() else 'skip'
    assert policy_on_null in ['ret_none', 'skip']
    # 2. Lambdas
    # (1) Processing key
    lmd_k = None
    # 1> Remove
    if 'remove' in kwargs.keys():
        assert lmd_k is None
        tbr = kwargs['remove']
        # (1) Single str
        if isinstance(tbr, str):
            lmd_k = lambda _k: _k.replace(tbr, "")
        # (2) Multi str
        elif is_tuple_list(tbr):
            lmd_k_str = "lambda k: k"
            for t in tbr: lmd_k_str += ".replace('%s', '')" % t
            lmd_k = eval(lmd_k_str)
        # Others
        else:
            raise NotImplementedError
    # 2> Replace
    if 'replace' in kwargs.keys():
        assert lmd_k is None
        tbr = kwargs['replace']
        # (1) Prefix
        if isinstance(tbr, str):
            assert not ('prefix' in kwargs.keys() and 'suffix' in kwargs.keys()), "Can't determine use which. "
            _which = 'prefix' if 'prefix' in kwargs.keys() else 'suffix'
            lmd_k = lambda _k: _k.replace(kwargs[_which], tbr)
        # (2) Tuple
        elif is_tuple_list(tbr):
            lmd_k = lambda _k: _k.replace(tbr[0], tbr[1])
        # Others
        else:
            raise NotImplementedError
    # 3> Other
    if 'lmd_k' in kwargs.keys():
        assert lmd_k is None
        lmd_k = kwargs['lmd_k']
    # (2) Processing values
    lmd_v = None if 'lmd_v' not in kwargs.keys() else kwargs['lmd_v']
    # ------------------------------------------------------------------------------------------------------------------
    # Fetching items
    # ------------------------------------------------------------------------------------------------------------------
    # 1. Collect keys
    if (not args) and ('prefix' not in kwargs.keys()) and ('suffix' not in kwargs.keys()):
        args = list(_dict.keys())
    else:
        args = list(args)
        if 'prefix' in kwargs.keys(): args += list(filter(lambda _k: _k.startswith(kwargs['prefix']), _dict.keys()))
        if 'suffix' in kwargs.keys(): args += list(filter(lambda _k: _k.endswith(kwargs['suffix']), _dict.keys()))
    # 2. Fetching items
    # (1) Init
    ret = OrderedDict()
    # (2) Process
    for k in args:
        # Fetching
        k_fetch, k_ret = k if is_tuple_list(k) else (k, k)
        if k not in _dict.keys():
            if policy_on_null == 'skip':
                continue
            else:
                v = None
        else:
            v = _dict[k_fetch] if not chk_d(kwargs, 'pop') else _dict.pop(k_fetch)
        # Processing key & value
        if lmd_k is not None: k_ret = lmd_k(k_ret)
        if lmd_v is not None: v = lmd_v(v)
        # Set
        ret[k_ret] = v
    # (3) Return
    return ret


# ----------------------------------------------------------------------------------------------------------------------
# Checking key
# ----------------------------------------------------------------------------------------------------------------------

def check_container(lmd_check_key, lmd_fetch_key, key, operator=None, another=None):
    # 1. Return False if key not exists
    if not lmd_check_key(key): return False
    # 2. Check
    if operator is None:
        return lmd_fetch_key(key)
    elif operator == 'not':
        return not lmd_fetch_key(key)
    elif callable(operator):
        if another is None:
            return operator(lmd_fetch_key(key))
        else:
            return operator(lmd_fetch_key(key), another)
    else:
        return eval('lmd_fetch_key(key) %s another' % operator)


def chk_d(container, key, operator=None, another=None):
    lmd_check_key = lambda k: k in container.keys()
    lmd_fetch_key = lambda k: container[k]
    return check_container(lmd_check_key, lmd_fetch_key, key, operator, another)


def chk_ns(container, key, operator=None, another=None):
    lmd_check_key = lambda k: hasattr(container, k)
    lmd_fetch_key = lambda k: getattr(container, k)
    return check_container(lmd_check_key, lmd_fetch_key, key, operator, another)


########################################################################################################################
# Utils
########################################################################################################################

def is_tuple_list(val):
    return isinstance(val, tuple) or isinstance(val, list)


class TempDirManager(object):
    """
    Contextual manager. Create temporary directory for evaluations when entering, and delete when exiting.
    """
    def __init__(self, root_dir, *dir_names):
        self._root_dir = root_dir
        self._dir_names = dir_names
        self._dirs = []

    def __enter__(self):
        # Process each one
        for name in self._dir_names:
            temp_dir = os.path.join(self._root_dir, name)
            # Make directory
            assert not os.path.exists(temp_dir)
            os.makedirs(temp_dir)
            # Save
            self._dirs.append(temp_dir)
        # Return
        return self._dirs

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Delete temporary directories
        for dir_path in self._dirs:
            shutil.rmtree(dir_path)


class TempKwargsManager(object):
    """
    Config manager. Temporarily update config before an operation (when entering), and restore when exiting.
    """
    def __init__(self, instance, **kwargs):
        """
        :param instance: Should has attr _kwargs.
        :param kwargs: Configs to be updated.
        """
        # Members
        self._instance = instance
        self._kwargs = kwargs
        # Original kwargs
        self._orig_kwargs = getattr(self._instance, '_kwargs')

    def __enter__(self):
        if not self._kwargs: return
        # Get temporary cfg
        kwargs = fet_d(self._orig_kwargs, *list(self._orig_kwargs.keys()))
        kwargs.update(self._kwargs)
        # Set
        setattr(self._instance, '_kwargs', kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._kwargs: return
        # Restore cfg
        setattr(self._instance, '_kwargs', self._orig_kwargs)


class TempDelAttr(object):
    """
    Delete an instance's attribute temporarily.
    """
    def __init__(self, instance, attr, delete):
        # Config
        self._instance, self._attr, self._value = instance, attr, getattr(instance, attr)
        self._delete = delete

    def __enter__(self):
        if self._delete:
            try:
                delattr(self._instance, self._attr)
            except AttributeError:
                self._delete = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._delete:
            setattr(self._instance, self._attr, self._value)


class PathPreparation(object):
    """
    Given a path, make directories.
    """
    def __init__(self, *args, **kwargs):
        """
        :param args: Components of the path.
        :param kwargs:
        """
        args = list(args)
        # Preprocess paths
        if 'appendix' in kwargs.keys():
            args[-1] = "%s%s" % (args[-1], kwargs['appendix'])
        if 'ext' in kwargs.keys():
            args[-1] = "%s.%s" % (args[-1], kwargs['ext'])
        # Get path
        if 'ext' in kwargs.keys():
            self._file_path = os.path.join(*args)
            self._dir = os.path.split(self._file_path)[0]
        else:
            self._dir = os.path.join(*args)

    def __enter__(self):
        # Mkdir
        if not os.path.exists(self._dir): os.makedirs(self._dir)
        # Return
        if not hasattr(self, '_file_path'):
            return self._dir
        else:
            return self._dir, self._file_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# ----------------------------------------------------------------------------------------------------------------------
# Dataset-like
# ----------------------------------------------------------------------------------------------------------------------

class BatchSlicerInt(object):
    """
    Slice batch.
    """
    def __init__(self, x, batch_size, ret_mode='batch_size'):
        assert ret_mode in ['batch_size', 'values']
        # Config
        self._batch_size = batch_size
        self._lmd_ret = (lambda _count, _batch_size: _batch_size) if ret_mode == 'batch_size' else \
            (lambda _count, _batch_size: range(_count, _count+_batch_size))
        # Members
        self._x = x
        self._counts = 0

    def __len__(self):
        return int(math.ceil(self._x/self._batch_size))

    def __iter__(self):
        return self

    def __next__(self):
        if self._counts < self._x:
            # Get batch_size
            batch_size = min(self._batch_size, self._x-self._counts)
            # 1. Result
            ret = self._lmd_ret(self._counts, batch_size)
            # 2. Update
            self._counts += batch_size
            # Return
            return ret
        else:
            raise StopIteration


class BatchSlicerLenObj(object):
    """
    Slice batch.
    """
    def __init__(self, x, batch_size, max_counts=-1):
        assert hasattr(x, '__len__')
        # Config
        self._batch_size, self._max_counts = batch_size, (min(max_counts, len(x)) if max_counts > 0 else len(x))
        # Members
        self._x = x
        self._counts = 0

    @property
    def max_counts(self):
        return self._max_counts

    def __len__(self):
        return int(math.ceil(self._max_counts/self._batch_size))

    def __iter__(self):
        return self

    def __next__(self):
        if self._counts < self._max_counts:
            # Get batch_size
            batch_size = min(self._batch_size, self._max_counts-self._counts)
            # 1. Result
            ret = self._x[self._counts:self._counts+batch_size]
            # 2. Update
            self._counts += batch_size
            # Return
            return ret
        else:
            raise StopIteration


########################################################################################################################
# Magic methods.
########################################################################################################################

def modify_base_cls(cls, base_cls, orig_cls=None):
    """
    Note: The cls cannot be directly inherited from the object class.
    """
    # Directly changing
    if orig_cls is None:
        cls.__bases__ = (base_cls, )
        return cls
    # Recursively changing all

    def _assign(_cls):
        _indices = []
        # 1. Change bases
        for _i, _base in enumerate(_cls.__bases__):
            if _base is object: continue
            # Recursive
            if _base is not orig_cls:
                _assign(_base)
            # Save index
            else:
                _indices.append(_i)
        # 2. Change current
        if _indices:
            _bases = list(_cls.__bases__)
            for _i in _indices:
                _bases[_i] = base_cls
            _cls.__bases__ = tuple(_bases)

    _assign(cls)
    return cls


def get_method_cls(meth):
    if isinstance(meth, functools.partial):
        return get_method_cls(meth.func)
    if inspect.ismethod(meth) or (inspect.isbuiltin(meth) and getattr(meth, '__self__', None) is not None and getattr(meth.__self__, '__class__', None)):
        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__: return cls
        # fallback to __qualname__ parsing
        meth = getattr(meth, '__func__', meth)
    if inspect.isfunction(meth):
        cls = getattr(inspect.getmodule(meth), meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0], None)
        if isinstance(cls, type): return cls
    # handle special descriptor objects
    return getattr(meth, '__objclass__', None)
