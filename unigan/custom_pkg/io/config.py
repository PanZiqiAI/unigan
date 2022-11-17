
import os
import sys
import copy
import torch
import pickle
import random
import inspect
import argparse
from types import MethodType
from argparse import Namespace
from ..io.logger import show_arguments
from ..basic.operations import is_tuple_list, fet_d


########################################################################################################################
# Fundamental
########################################################################################################################

def str2bool(string):
    if string.lower() == 'true': return True
    elif string.lower() == 'false': return False
    else: raise ValueError


class CustomParser(object):
    """
    Argument generator.
    """
    def __init__(self, args_dict):
        """
        Param: args_dict used to record all args at current time.
        """
        self._args_dict = args_dict
        # 1. Specified args
        self._parser, self._parser_names = argparse.ArgumentParser(allow_abbrev=False), []
        self._post_procs = {}
        # 2. Settings
        self._settings = {}

    def _check_duplicate(self, key):
        assert key not in self._parser_names, \
            "Key '%s' had already been added as a user-specified arguments. " % key
        assert key not in self._settings.keys(), \
            "Key '%s' has already been added as a determined setting with value '%s'. " % (key, self._settings[key])

    def add_argument(self, key, **kwargs):
        assert key.startswith("--"), "Argument key must start with '--'. "
        if key[2:] not in self._args_dict.keys():
            # Check duplicate
            self._check_duplicate(key[2:])
            # 1. Set command
            # (1) Case: bool type
            if 'type' in kwargs.keys() and kwargs['type'] == bool:
                kwargs['type'] = str2bool
                if 'default' in kwargs.keys(): kwargs['default'] = str(kwargs['default'])
            # (2) Case: offshell for list
            if 'offshell' in kwargs.keys():
                assert 'nargs' in kwargs.keys() and kwargs['nargs'] == '+'
                if kwargs.pop('offshell'): self._post_procs[key[2:]] = lambda _v: _v[0] if len(_v) == 1 else _v
            # Add
            self._parser.add_argument(key, **kwargs)
            # 2. Save
            self._parser_names.append(key[2:])

    def set(self, key, value):
        if isinstance(key, list):
            for k, v in zip(key, value):
                self.set(k, v)
        else:
            if key not in self._args_dict.keys():
                self._check_duplicate(key)
                self._settings.update({key: value})

    def get_default(self, dest):
        return self._parser.get_default(dest)

    def get_args_dict(self):
        # 1. Get respectively
        specified_args, _ = self._parser.parse_known_args()
        provided_args_dict = self._settings
        # 2. Get result
        # (1) Specified
        result = vars(specified_args)
        for _k, _func in self._post_procs.items(): result[_k] = _func(result[_k])
        # (2) Settings
        result.update(provided_args_dict)
        # Return
        return result, provided_args_dict.keys()


class ArgvBlocker(object):
    """
    Block argv when setting up config.
    """
    def __init__(self, block):
        self._block = block

    def __enter__(self):
        if self._block:
            setattr(self, '_saved_argv', sys.argv)
            sys.argv = sys.argv[:1]

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._block:
            sys.argv = getattr(self, '_saved_argv')


class TreeConfig(object):
    """
    1. Add root args;
    2. Add tree args according to current args:
        (1) Add arguments that are specified by user;
        (2) Add arguments that are uniquely determined;
    3. Add additional args;

    For convenience (typically loading config from file), arguments can be induced via args_inductor:
    1. Inducted args will be used during the entire building process of args tree;
    2. Arguments priorities:
            specifically provided
        =   determined (if their conditions have already been updated given inducted args)
        >   inducted args
        >   default args

    Arguments must be in the form of '--args=value'.
    """
    def __init__(self, args_inductor=None, block_argv=False):
        # Initialize
        with ArgvBlocker(block_argv):
            self.__setup__(args_inductor)

    def __setup__(self, args_inductor):
        # Parsing inducted_args from inductor
        self._inducted_args_dict = self._parse_inducted_args(args_inductor)
        # Get args & default args
        self._default_args_dict = {}
        ################################################################################################################
        # Stage 1: Add root args & parsing
        ################################################################################################################
        # 1. Add root args & parsing
        self.parser = CustomParser({})
        self._add_root_args()
        root_args_dict, provided_keys = self.parser.get_args_dict()
        # 2. Merge with inducted_args in terms of root_sys_args & save default
        self._collect_default_args(root_args_dict.keys())
        args_dict = self._merge_with_inducted_args(root_args_dict, provided_keys)
        ################################################################################################################
        # Stage 2: Add tree args & parsing via loop
        ################################################################################################################
        while True:
            # 1. Add tree args according to current args & parsing
            # (1) Reset
            self.parser = CustomParser(args_dict)
            # (2) Get args
            self._add_tree_args(args_dict)
            loop_args_dict, provided_keys = self.parser.get_args_dict()
            # 2. Get incremental args
            assert len(list(filter(lambda n: n in args_dict.keys(), loop_args_dict.keys()))) == 0
            # (2) Update or break
            if loop_args_dict:
                # 1> Merge with inducted_args for incremental_args & save default
                self._collect_default_args(loop_args_dict.keys())
                loop_args_dict = self._merge_with_inducted_args(loop_args_dict, provided_keys)
                # 2> Merge with stale
                args_dict = self._check_duplicated_args_and_merge(args_dict, loop_args_dict, 'stale', 'incremental')
            else:
                break
        ################################################################################################################
        # Stage 3: Add additional args & parsing
        ################################################################################################################
        # 1. Add additional args
        self.parser = CustomParser(args_dict)
        self._add_additional_args()
        additional_args_dict, provided_keys = self.parser.get_args_dict()
        # 2. Merge with inducted & save default
        self._collect_default_args(additional_args_dict.keys())
        additional_args_dict = self._merge_with_inducted_args(additional_args_dict, provided_keys)
        # 3. Update
        args_dict = self._check_duplicated_args_and_merge(args_dict, additional_args_dict, 'tree', 'additional')
        ################################################################################################################
        # Get final args
        ################################################################################################################
        self.args = Namespace(**args_dict)

        # Delete useless members
        del self.parser

    def _volatile_args(self):
        return []

    @staticmethod
    def _check_duplicated_args_and_merge(args_dict1, args_dict2, args_name1, args_name2):
        duplicated_names = list(filter(lambda name: name in args_dict1.keys(), args_dict2.keys()))
        assert len(duplicated_names) == 0, \
            'Duplicated args between %s and %s args: %s. ' % (args_name1, args_name2, str(duplicated_names))
        args_dict1.update(args_dict2)
        return args_dict1

    ####################################################################################################################
    # Args-generate
    ####################################################################################################################

    def _parse_inducted_args(self, args_inductor):
        """
        This can be override, e.g., for loading from file to get induced_args.
        """
        return {}

    def _add_root_args(self):
        # Like: self.parser.add_argument(...)
        pass

    def _add_tree_args(self, args_dict):
        # Like: self.parser.add_argument(...) or
        #       self.parser.set(...)
        pass

    def _add_additional_args(self):
        # Like: self.parser.add_argument(...)
        pass

    @staticmethod
    def _collect_provided_sys_args_name():
        """
        Specifically provided and determined args have the highest priority.
        """
        # 1. Init result
        args_name = []
        # 2. Collect from commandline
        args_list = sys.argv[1:]
        for arg in args_list:
            if not arg.startswith("--"): continue
            # (1) Has = (typical)
            if '=' in arg: arg_name = str(arg[2:]).split("=")[0]
            # (2) Otherwise (typically the arg is list)
            else: arg_name = arg[2:]
            # Save
            assert arg_name not in args_name, "Duplicated arg_name '%s'. " % arg_name
            args_name.append(arg_name)
        # Return
        return args_name

    def _collect_default_args(self, args_keys):
        for key in args_keys:
            assert key not in self._default_args_dict.keys()
            # Only specifically provided args can be collected to default dict
            if key in self._collect_provided_sys_args_name():
                self._default_args_dict.update({key: self.parser.get_default(key)})

    def _merge_with_inducted_args(self, args_dict, provided_keys):
        if not self._inducted_args_dict: return args_dict
        # 1. Collect highest priority args: provided & determined
        highest_priority_args = {
            key: args_dict[key]
            for key in filter(lambda name: name in args_dict.keys(), self._collect_provided_sys_args_name() + list(provided_keys))
        }
        # 2. Update inducted_args into sys_args
        args_dict.update({
            key: self._inducted_args_dict[key]
            for key in filter(lambda name: name in args_dict.keys(), self._inducted_args_dict.keys())
        })
        # 3. Override highest priority args
        args_dict.update(highest_priority_args)
        # Return
        return args_dict

    ####################################################################################################################
    # Utilization
    ####################################################################################################################

    def show_arguments(self, title=None, identifier='induced'):
        return show_arguments(
            self.args, title, self._volatile_args(),
            default_args=self._default_args_dict, compared_args=self._inducted_args_dict, competitor_name=identifier)

    def fet_d(self, *args, **kwargs):
        return fet_d(vars(self.args), *args, **kwargs)

    def merge(self, others, overwrite=True):
        """
        :type others: TreeConfig or list of TreeConfig
        :param overwrite:
        """
        def _update_dict_ow(_orig, _new, _ow):
            if _ow: _orig.update(_new)
            else: _orig.update(fet_d(_new, *(set(_new.keys()) - set(_orig.keys()))))

        assert not hasattr(self, '_merged')
        if not is_tuple_list(others): others = [others]
        if not is_tuple_list(overwrite): overwrite = [overwrite] * len(others)
        # --------------------------------------------------------------------------------------------------------------
        # Merge
        # --------------------------------------------------------------------------------------------------------------
        cfg = copy.deepcopy(self)
        # 1. Init args
        args_dict = copy.deepcopy(vars(self.args))
        # 2. Update
        for other_cfg, ow in zip(others, overwrite):
            # (1) Args
            _update_dict_ow(_orig=args_dict, _new=vars(other_cfg.args), _ow=ow)
            # (2) Others
            _update_dict_ow(_orig=cfg._inducted_args_dict, _new=other_cfg._inducted_args_dict, _ow=ow)
            _update_dict_ow(_orig=cfg._default_args_dict, _new=other_cfg._default_args_dict, _ow=ow)
        # 3. Set args
        cfg.args = Namespace(**args_dict)
        # --------------------------------------------------------------------------------------------------------------
        # Set methods
        # --------------------------------------------------------------------------------------------------------------

        def _show_arguments(
                _self, title=tuple(['Config - main'] + ['Config - sub[%d]' % i for i in range(len(others))]),
                identifier=inspect.signature(cfg.show_arguments).parameters['identifier'].default):
            """
            Adapted method for 'show_arguments'.
            """
            if is_tuple_list(title):
                # Get kwargs
                if not is_tuple_list(identifier): identifier = [identifier] * (1 + len(others))
                # Get results
                echo = self.show_arguments(title[0], identifier[0])
                for _index, _other_cfg in enumerate(others):
                    echo += _other_cfg.show_arguments(title[_index + 1], identifier[_index + 1])
                # Return
                return echo
            else:
                return super(_self.__class__, _self).show_arguments(title, identifier)

        cfg.show_arguments = MethodType(_show_arguments, cfg)
        setattr(cfg, '_merged', len(others))
        # Return
        return cfg


########################################################################################################################
# Up-level
########################################################################################################################

def update_freq_args(cfg):
    default_args_dict = getattr(cfg, '_default_args_dict')
    for k, v in copy.deepcopy(vars(cfg.args)).items():
        # Replace starting with freq_counts_
        if k.startswith("freq_counts_"):
            marker = '%ss' % k[len("freq_counts_"):].split("_")[0]
            new_k = 'freq_%s' % k[len("freq_counts_"):]
            _get_freq = lambda _v: getattr(cfg.args, marker) // _v if _v > 0 else _v
            # ----------------------------------------------------------------------------------------------------------
            # 1. Replace freq_counts -> freq.
            setattr(cfg.args, new_k, _get_freq(v))
            delattr(cfg.args, k)
            # 2. Update default args.
            if k in default_args_dict:
                default_args_dict[new_k] = _get_freq(default_args_dict.pop(k))


class CanonicalConfig(TreeConfig):
    """
    Canonical Config.
    """
    def __init__(self, exp_dir, load_rel_path=None, deploy=True, block_argv=False):
        """
            Having the following built-in Arguments:
            (1) --desc:
            (2) --rand_seed
            (3) --load_from (for resuming training, typically an epoch or iteration index)
        and the following environmental arguments:
            (1) --exp_dir = given_exp_dir
            (2) --trial_dir = exp_dir/given_desc/RandSeed[%d]
            (3) --params_dir = trial_dir/params
            (4) --ana_dir = trial_dir/analyses
            (5, optional) --tfboard_dir = exp_dir/../tfboard
        """
        # Members
        self._exp_dir = exp_dir
        # 1. Generate load_rel_path from given
        if load_rel_path is None:
            load_rel_path = self._generate_load_rel_path()
        # 2. If given, generating is not allowed
        else:
            # (1) Given is list of str.
            if not isinstance(load_rel_path, str):
                desc, rand_seed, load_from = load_rel_path
                # Automatically get rand seed.
                if rand_seed is None:
                    choices = os.listdir(os.path.join(self._exp_dir, desc))
                    assert len(choices) == 1, "Two many trials so can not determine which. "
                    rand_seed = int(choices[0].split("[")[-1].split("]")[0])
                # Set load path.
                load_rel_path = "%s/RandSeed[%s]/params/config[%s].pkl" % (desc, rand_seed, load_from)
            # (2) Given is directly path
            else:
                pass
        # Super method for parsing configurations
        super(CanonicalConfig, self).__init__(args_inductor=load_rel_path, block_argv=block_argv)
        # Set other args & check compatibility.
        self._init_method()
        # --------------------------------------------------------------------------------------------------------------
        # Deploy
        if deploy: self.deploy()

    def _init_method(self):
        """
        Set other args & check compatibility.
        """
        # Directories
        self._set_directory_args()
        if self.args.load_from == -1:
            assert not os.path.exists(self.args.trial_dir), "Rand seed '%d' has already been generated. " % self.args.rand_seed

        # Delete useless members
        del self._exp_dir

    def _deploy_method(self):
        pass

    def deploy(self):
        self._deploy_method()
        return self

    def _volatile_args(self):
        """
        These args will not be saved or displayed.
        """
        return ['desc'] + list(filter(lambda name: str(name).endswith('_dir'), vars(self.args).keys()))

    ####################################################################################################################
    # Args-generate
    ####################################################################################################################

    def _add_root_args(self):
        # (1) Description
        self.parser.add_argument("--desc",          type=str, default='unspecified')
        # (2) Rand Seed
        self.parser.add_argument("--rand_seed",     type=int, default=random.randint(0, 10000))
        # (3) Load_from
        self.parser.add_argument("--load_from",     type=int, default=-1)

    def _generate_load_rel_path(self):
        # 1. Init results
        load_args = {'desc': None, 'rand_seed': None, 'load_from': None}
        # 2. Check args
        for arg_index, arg in enumerate(sys.argv[1:]):
            if not str(arg).startswith("--"): continue
            # (1) Get arg name & value
            if "=" in arg:
                assign_index = arg.find("=")
                arg_name, arg_value = arg[2:assign_index], arg[assign_index + 1:]
            else:
                # Arg name
                arg_name = arg[2:]
                # Arg value
                arg_eindex = arg_index + 1
                while (arg_eindex < len(sys.argv)) and (not sys.argv[arg_eindex].startswith("--")): arg_eindex += 1
                arg_value = sys.argv[arg_index + 1:arg_eindex]
                # Check
                if len(arg_value) != 1: continue
                arg_value = arg_value[0]
            # (2) Save to args
            if arg_name in load_args.keys(): load_args[arg_name] = arg_value
        # 3. Generate
        # Get rand_seed automatically
        none_args_name = list(filter(lambda key: load_args[key] is None, load_args.keys()))
        if len(none_args_name) == 1 and none_args_name[0] == 'rand_seed':
            desc_dir = os.path.join(self._exp_dir, load_args['desc'])
            trials_dirs = list(filter(lambda _d: _d.startswith("RandSeed"), os.listdir(desc_dir)))
            if len(trials_dirs) != 1:
                raise AssertionError("Too many trials in '%s'. Please specify a rand seed. " % desc_dir)
            else:
                load_args['rand_seed'] = int(str(trials_dirs[0]).split('[')[1].split(']')[0])
        # (1) Invalid
        for arg_value in load_args.values():
            if arg_value is None: return None
        # (2) Valid
        else:
            load_rel_path = "%s/RandSeed[%s]/params/config[%s].pkl" % (
                load_args['desc'], load_args['rand_seed'], load_args['load_from'])
            return load_rel_path

    def _parse_inducted_args(self, args_inductor):
        # Not loading
        if args_inductor is None:
            return {}
        # Loading
        else:
            ############################################################################################################
            # Get desc & rand_seed & load_from
            ############################################################################################################
            # 1. Get description
            params_dir, load_config_file = os.path.split(args_inductor)
            desc_with_rand_seed = os.path.split(params_dir)[0]
            inducted_args_desc, rand_seed = os.path.split(desc_with_rand_seed)
            # 2. Set random seed
            inducted_args_rand_seed = int(rand_seed.split("[")[1].split("]")[0])
            # 3. Set load_from
            inducted_args_load_from = int(load_config_file.split("[")[1].split("]")[0])
            ############################################################################################################
            # Load from config & update
            ############################################################################################################
            with open(os.path.join(self._exp_dir, args_inductor), 'rb') as f:
                saved_args_dict = pickle.load(f)
            saved_args_dict.update({
                'desc': inducted_args_desc,
                'rand_seed': inducted_args_rand_seed,
                'load_from': inducted_args_load_from
            })
            # Return
            return saved_args_dict

    def _set_directory_args(self, **kwargs):
        # Set directories
        self.args.desc = os.path.join(self.args.desc, 'RandSeed[%d]' % self.args.rand_seed)
        self.args.exp_dir = self._exp_dir
        self.args.trial_dir = os.path.join(self.args.exp_dir, self.args.desc)
        self.args.params_dir = os.path.join(self.args.trial_dir, 'params')
        self.args.ana_dir = os.path.join(self.args.trial_dir, 'analyses')
        # Optional
        if not ('disable_ana_train_dir' in kwargs.keys() and kwargs['disable_ana_train_dir']):
            self.args.ana_train_dir = os.path.join(self.args.ana_dir, 'train')
        if not ('disable_tfboard' in kwargs.keys() and kwargs['disable_tfboard']):
            self.args.tfboard_dir = os.path.join(self.args.exp_dir, '../tensorboard')

    ####################################################################################################################
    # Utilization
    ####################################################################################################################

    def generate_save_path(self, n):
        return os.path.join(self.args.params_dir, 'config[%d].pkl' % n)

    def generate_checkpoint_path(self, n):
        return os.path.join(self.args.params_dir, 'checkpoint[%d].pth.tar' % n)

    def save(self, n, stale_n=None):
        save_path = self.generate_save_path(n)
        # 1. Save current
        if not os.path.exists(save_path):
            with open(save_path, 'wb') as f:
                args = vars(self.args)
                pickle.dump({k: args[k] for k in set(args.keys()) - set(self._volatile_args())}, f)
        # 2. Move stale
        if stale_n is not None:
            os.remove(self.generate_save_path(stale_n))

    def show_arguments(self, title=None, identifier='loaded'):
        return super(CanonicalConfig, self).show_arguments(title, identifier)


class CanonicalConfigPyTorch(CanonicalConfig):
    """
    Canonical config used in PyTorch.
    """
    def _init_method(self, **kwargs):
        super(CanonicalConfigPyTorch, self)._init_method()
        # Set device.
        self.args.device = 'cuda' if self.args.gpu_ids != '-1' else 'cpu'
        # Update freq args.
        if 'update_freq_args' not in kwargs.keys() or kwargs['update_freq_args']: update_freq_args(self)
        # Check.
        if 'multi_gpus' in kwargs.keys() and kwargs['multi_gpus']:
            assert ',' in self.args.gpu_ids, "Please specify multi GPU ids. "
        else:
            assert ',' not in self.args.gpu_ids, "Multi-GPUs training is not allowed. "

    def _deploy_method(self):
        super(CanonicalConfigPyTorch, self)._deploy_method()
        # Random seed
        random.seed(self.args.rand_seed)
        torch.manual_seed(self.args.rand_seed)
        # Context
        if self.args.device == 'cuda': os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu_ids

    def _add_root_args(self):
        super(CanonicalConfigPyTorch, self)._add_root_args()
        # Context
        self.parser.add_argument("--gpu_ids",   type=str,   default='0', help="GPU ids. Set to -1 for CPU mode. ")
