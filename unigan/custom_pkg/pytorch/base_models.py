# Basic models that can be widely re-implemented.

import os
import torch
from torch import nn
from functools import partial
from torch.utils.data import DataLoader
from ..io.logger import Logger, TfBoard
from ..pytorch.operations import network_param_m
from ..basic.operations import ValidDict, chk_d, chk_ns, fet_d
from ..basic.metrics import FreqCounter, StopWatch, TimersController, Pbar, ResumableMeter


class BaseModel(nn.Module):
    """
    Base model.
    ####################################################################################################################
    # Required cfg keys:
    ####################################################################################################################
        - tfboard_dir   (optional)

    ####################################################################################################################
    # Components
    ####################################################################################################################
        - self._logs['log_main']            (optional @ kwargs)
        - self._logs['tfboard']             (optional @ cfg.args.tfboard_dir & kwargs)

        - packs
            - 'log':                        (optional @ self._logs['log_main'] & kwargs)
            - 'tfboard':                    (optional @ self._logs['tfboard'] & kwargs)

        - self._meters['timers']
            - 'io':                         (optional @ kwargs)
            - 'optimize':                   (optional @ kwargs)
        - self._meters['i']                 (optional @ kwargs)
    """
    def __init__(self, cfg):
        super(BaseModel, self).__init__()
        # Config.
        self._cfg = cfg
        # 1. Build architectures
        self._build_architectures()
        # 2. Setup.
        self._setup()

    def _build_architectures(self, **modules):
        """
        :param modules: { name1: module1, name2: module2 }
        :return: Building modules. Skipping if module is None.
            self._networks = { name1: module1, name2: module2 }
            self._name1 = module1
            self._name2 = module2
        """
        assert not hasattr(self, '_networks')
        # Build architecture
        assert modules
        # 1. Dict to save all modules
        self._networks = {}
        # 2. Register modules
        for name, module in modules.items():
            if module is None: continue
            # Get attr name
            assert not name.startswith("_")
            # Save
            assert not hasattr(self, '_%s' % name)
            setattr(self, '_%s' % name, module)
            self._networks[name] = module

    ####################################################################################################################
    # Setup
    ####################################################################################################################

    def _set_criterions(self):
        pass

    def _set_optimizers(self):
        pass

    def _get_scheduler(self, optimizer, last_n):
        pass

    def _set_schedulers(self):
        """
        Default: shared lr_scheduler for all optimizers.
        """
        for key, optimizer in self._optimizers.items():
            self._schedulers.update({key: self._get_scheduler(optimizer, self._cfg.args.load_from)})

    def _set_logs(self, **kwargs):
        if not chk_d(kwargs, 'disable_log'):
            # Get log_dir
            if 'log_main_dir' in kwargs.keys(): log_dir = kwargs['log_main_dir']
            elif hasattr(self._cfg.args, 'ana_train_dir'): log_dir = self._cfg.args.ana_train_dir
            else: log_dir = self._cfg.args.ana_dir
            # Set log
            self._logs['log_main'] = Logger(
                log_dir=log_dir, log_name='train' if 'log_main_name' not in kwargs.keys() else kwargs['log_main_name'],
                formatted_prefix=self._cfg.args.desc, formatted_counters=kwargs['log_main_counters'],
                append_mode=False if self._cfg.args.load_from == -1 else True,
                pbar=self._meters['pbar'] if 'pbar' in self._meters.keys() else None)
        if hasattr(self._cfg.args, 'tfboard_dir') and not chk_d(kwargs, 'disable_tfboard'):
            self._logs['tfboard'] = TfBoard(os.path.join(self._cfg.args.tfboard_dir, self._cfg.args.desc))

    def _set_meters(self, **kwargs):
        # Timers
        self._meters['timers'] = TimersController()
        if not chk_d(kwargs, 'disable_timers'):
            self._meters['timers']['io'] = StopWatch()
            self._meters['timers']['opt'] = StopWatch()
        # Iterations
        if 'i' in kwargs.keys():
            self._meters['i'] = kwargs['i']
        # Progress bar
        pbar_total = None if 'pbar_total' not in kwargs.keys() else kwargs['pbar_total']
        if pbar_total != -1: self._meters['pbar'] = Pbar(total=kwargs['pbar_total'], desc=self._cfg.args.desc)

    def _deploy(self):
        # GPU
        if self._cfg.args.device == 'cuda':
            self.cuda()

    def _setup(self):

        def _check_and_set_dict(name):
            assert not hasattr(self, name)
            # Init
            setattr(self, name, {})
            # Call func to setup
            getattr(self, '_set%s' % name)()

        # 1. Deploy network to GPUs.
        self._deploy()

        # 2. Set items
        load = self._cfg.args.load_from != -1
        chkpt_pack = self._load_checkpoint() if load else None
        """ Principle: Setting an item does not affect anything of former set items. """
        # (1) Meters
        _check_and_set_dict('_meters')
        self._resume(mode='meters', checkpoint_pack=chkpt_pack)
        # (2) Logs
        _check_and_set_dict('_logs')
        # (3) Optimizers
        _check_and_set_dict('_optimizers')
        self._resume(mode='modules_and_optimizers', checkpoint_pack=chkpt_pack)
        # (4) Schedulers
        _check_and_set_dict('_schedulers')
        # (5) Criterions
        _check_and_set_dict('_criterions')
        # Other resumable items.
        self._resume(mode='others', checkpoint_pack=chkpt_pack, verbose=True)

    ####################################################################################################################
    # Save & Load
    ####################################################################################################################

    def _selecting_modules_and_optimizers_for_chkpt(self, **kwargs):
        # 1. Modules
        if 'modules' not in kwargs.keys():
            modules = {'default': self}
        elif isinstance(kwargs['modules'], dict):
            modules = kwargs['modules']
        else:
            # Given should be list/tuple
            modules = {k: getattr(self, k) for k in kwargs['modules']}
        # 2. Optimizers
        if chk_ns(self, '_optimizers'):
            if 'optimizers' not in kwargs.keys():
                optimizers = self._optimizers
            elif kwargs['optimizers'] == 'sync_with_modules':
                optimizers = {k: self._optimizers[k] for k in modules.keys()}
            elif kwargs['optimizers'] is None:
                optimizers = None
            else:
                # Given should be list/tuple
                optimizers = {k: self._optimizers[k] for k in kwargs['optimizers']}
        else:
            optimizers = None
        # Return
        return modules, optimizers

    def _save_checkpoint(self, n, stale_n=None, **kwargs):
        # 1. Save checkpoint
        save_path = self._cfg.generate_checkpoint_path(n)
        if not os.path.exists(save_path):
            modules, optimizers = self._selecting_modules_and_optimizers_for_chkpt(**kwargs)
            # ----------------------------------------------------------------------------------------------------------
            # (1) Get final state to save (last, iterations, state_dict, optimizer)
            # ----------------------------------------------------------------------------------------------------------
            state = {'last': n, 'state_dict': {k: v.state_dict() for k, v in modules.items()}}
            if 'i' in self._meters.keys(): state['i'] = self._meters['i']
            if optimizers is not None:
                state['optimizers'] = {k: v.state_dict() for k, v in optimizers.items()}
            # ----------------------------------------------------------------------------------------------------------
            # (2) Resumable meters
            # ----------------------------------------------------------------------------------------------------------
            for key, meter in self._meters.items():
                if isinstance(meter, ResumableMeter):
                    key = 'meter-' + key
                    assert key not in state.keys()
                    state[key] = meter.save()
            # ----------------------------------------------------------------------------------------------------------
            # Additional items
            # ----------------------------------------------------------------------------------------------------------
            if 'items' in kwargs.keys():
                for key, value in kwargs['items'].items():
                    assert key not in state.keys()
                    state[key] = value
            # Save
            try:
                torch.save(state, save_path)
            except FileNotFoundError:
                os.makedirs(os.path.split(save_path)[0])
                torch.save(state, save_path)
        if stale_n is not None: os.remove(self._cfg.generate_checkpoint_path(stale_n))
        # 2. Save config
        self._cfg.save(n, stale_n)

    def _load_checkpoint(self):
        assert self._cfg.args.load_from != -1, "Please specify args.load_from. "
        # 1. Load from file & check
        checkpoint_path = self._cfg.generate_checkpoint_path(self._cfg.args.load_from)
        checkpoint = torch.load(checkpoint_path)
        assert checkpoint['last'] == self._cfg.args.load_from
        # Return
        return checkpoint, checkpoint_path

    def _resume(self, mode, checkpoint_pack=None, verbose=False, **kwargs):
        if checkpoint_pack is None: return
        # Unpack
        checkpoint, checkpoint_path = checkpoint_pack
        # --------------------------------------------------------------------------------------------------------------
        # Meters.
        # --------------------------------------------------------------------------------------------------------------
        if mode == 'meters':
            # Iterations.
            if 'i' in checkpoint.keys(): self._meters['i'] = checkpoint['i']
            # Resumable meters.
            if not chk_d(kwargs, 'resume_meters', 'not'):
                for key, meter in self._meters.items():
                    if isinstance(meter, ResumableMeter):
                        try:
                            meter.load(**checkpoint['meter-' + key])
                        except KeyError:
                            pass
            # Additional items in meters
            if 'lmd_load_meter_items' in kwargs.keys():
                kwargs['lmd_load_meter_items'](checkpoint)
        # --------------------------------------------------------------------------------------------------------------
        # Modules & optimizers
        # --------------------------------------------------------------------------------------------------------------
        elif mode == 'modules_and_optimizers':
            modules, optimizers = self._selecting_modules_and_optimizers_for_chkpt(**kwargs)
            for k, module in modules.items(): module.load_state_dict(checkpoint['state_dict'][k])
            if optimizers is not None:
                for k, optimizer in optimizers.items(): optimizer.load_state_dict(checkpoint['optimizers'][k])
        # --------------------------------------------------------------------------------------------------------------
        # Others
        # --------------------------------------------------------------------------------------------------------------
        elif mode == 'others':
            if 'lmd_load_other_items' in kwargs.keys():
                kwargs['lmd_load_other_items'](checkpoint)
        else:
            raise NotImplementedError
        # Logging
        if verbose: print("Loaded from checkpoint '%s'. " % checkpoint_path)

    def _remove_params(self, n):
        if not os.path.exists(self._cfg.args.params_dir): return
        # --------------------------------------------------------------------------------------------------------------
        # Remove multiple params.
        # (1) Less than a threshold
        if isinstance(n, str) and n.startswith("<"):
            n_thr = int(n[1:])
            # Check & remove
            for _n in sorted([int(_f.split("checkpoint[")[1].split("]")[0]) for _f in os.listdir(self._cfg.args.params_dir) if _f.startswith("checkpoint")]):
                if _n < n_thr: self._remove_params(_n)
            return
        # (2) Given list
        if isinstance(n, list):
            for _n in n: self._remove_params(_n)
            return
        # --------------------------------------------------------------------------------------------------------------

        # Meta method: remove checkpoint & config.
        try:
            os.remove(self._cfg.generate_checkpoint_path(n))
            os.remove(self._cfg.generate_save_path(n))
        except FileNotFoundError: pass

    ####################################################################################################################
    # Train procedure.
    ####################################################################################################################

    def _train_procedure(self):
        raise NotImplementedError

    def _init_packs(self, *args, **kwargs):
        """ Init packages. """
        def _init(_k):
            return ValidDict(**(kwargs[_k] if _k in kwargs.keys() else {}))
        # 1. Init
        ret = {}
        # 2. Set packages
        # (1) Log
        if 'log_main' in self._logs.keys() and not chk_d(kwargs, 'disable_log'):
            assert 'log' not in args
            ret['log'] = _init('log')
        # (2) TfBoard
        if 'tfboard' in self._logs.keys() and not chk_d(kwargs, 'disable_tfboard'):
            assert 'tfboard' not in args
            ret['tfboard'] = _init('tfboard')
        # (3) Others
        if len(args) > 0:
            assert len(set(args)) == len(args)
            for k in args: ret[k] = _init(k)
        # Return
        return ret

    def _deploy_batch_data(self, batch_data):
        raise NotImplementedError

    def _set_to_train_mode(self):
        self.train()

    def _update_learning_rate(self):
        for scheduler in self._schedulers.values():
            if scheduler is not None: scheduler.step()

    def _process_log(self, log_counters, packs, **kwargs):
        if chk_d(self._meters, 'counter_log', lambda c: c.check()):
            if 'lmd_generate_log' in kwargs.keys(): kwargs['lmd_generate_log']()
            # (1) Logs
            if 'log_main' in self._logs.keys() and not chk_d(kwargs, 'disable_log'):
                # Update io & optimize timers
                if 'io' in self._meters['timers']:
                    # Make sure that timer information at the end.
                    if 't_io' in packs['log'].keys(): packs['log'].pop('t_io')
                    packs['log']['t_io'] = self._meters['timers']['io'].get_duration_and_reset()
                if 'opt' in self._meters['timers']:
                    if 't_opt' in packs['log'].keys(): packs['log'].pop('t_opt')
                    packs['log']['t_opt'] = self._meters['timers']['opt'].get_duration_and_reset()
                # Show information
                log_kwargs = {'items': packs['log']} if 'lmd_process_log' not in kwargs.keys() else kwargs['lmd_process_log'](packs['log'])
                self._logs['log_main'].info_formatted(log_counters, **log_kwargs)
            # (2) Tensorboard
            if 'tfboard' in self._logs.keys() and not chk_d(kwargs, 'disable_tfboard'):
                self._logs['tfboard'].add_multi_scalars(packs['tfboard'], self._meters['counter_log'].iter_fetcher())

    def _process_chkpt_and_lr(self, iter_checker=None):
        # Update learning rate
        self._update_learning_rate()
        # Save current epoch
        if chk_d(self._meters, 'counter_chkpt', lambda c: c.check(iter_checker)):
            self._save_checkpoint(self._meters['counter_chkpt'].iter_fetcher() if iter_checker is None else iter_checker)

    ####################################################################################################################
    # Eval procedure.
    ####################################################################################################################

    def _eval_procedure(self):
        pass

    ####################################################################################################################
    # APIs
    ####################################################################################################################

    def train_model(self, **kwargs):
        # Load data
        self._set_data(**kwargs)
        # Logging before training
        self._logging()
        # Train parameters
        self._train_procedure()

    def eval_model(self, **kwargs):
        """
        :param kwargs:
            - xxx_data:
            - show_cfg_xxx:
        :return:
        """
        get_log_keys = lambda: set(filter(lambda _k: isinstance(self._logs[_k], Logger), self._logs.keys()))
        ################################################################################################################
        # Load data
        ################################################################################################################
        self._set_data(**kwargs)
        ################################################################################################################
        # Setup for evaluation.
        ################################################################################################################
        # 1. Get current loggers.
        log_keys = get_log_keys()
        # 2. Setup for evaluation.
        # (1) Default log_dir
        default_log_kwargs = {'log_dir': self._cfg.args.ana_dir}
        # (2) Default formatted_counters
        if isinstance(self, EpochBatchBaseModel):
            formatted_counters = ['epoch', 'batch', 'iter']
        elif isinstance(self, IterativeBaseModel):
            formatted_counters = ['epoch', 'batch', 'step', 'iter']
        else:
            raise ValueError
        default_log_kwargs['formatted_counters'] = formatted_counters
        # Setup
        self._setup_eval(default_logger_kwargs=default_log_kwargs, **kwargs)
        ################################################################################################################
        # Logging
        ################################################################################################################
        # 1. Get show_cfg_kwargs
        if 'show_cfg_title' not in kwargs.keys():
            kwargs['show_cfg_title'] = ['Config - train']
            kwargs['show_cfg_title'] += ['Config - eval'] if getattr(self._cfg, '_merged') == 1 else \
                ['Config - eval[%d]' % i for i in range(getattr(self._cfg, '_merged'))]
        show_cfg_kwargs = fet_d(kwargs, prefix='show_cfg_')
        # 2. Get eval loggers.
        eval_log_keys = list(get_log_keys() - log_keys)
        # Show config
        self._logging(log_keys=eval_log_keys, **show_cfg_kwargs)
        ################################################################################################################
        # Evaluate
        ################################################################################################################
        self._eval_procedure()

    def _setup_eval(self, default_logger_kwargs, **kwargs):
        pass

    def _set_data(self, **kwargs):
        """
        Given xxx_data=yyy, there will be: self._data[xxx]=yyy.
        """
        setattr(self, '_data', {
            key[:-len('_data')]: kwargs[key]
            for key in filter(lambda k: k.endswith("_data") and kwargs[k] is not None, kwargs.keys())
        })

    def _logging(self, **kwargs):
        """
        Logging before training.
        :param:
            - log_keys: [..., ...]
            - show_cfg_xxx:
        :return: 
        """
        def _show_method(_lmd_show):
            # 1. Show cfg
            _lmd_show(self._cfg.show_arguments(**fet_d(kwargs, prefix='show_cfg_', replace='')))
            # 2. Show dataset
            for _n, _l in len_data.items():
                _lmd_show("Dataset[%s] size: %d. " % (_n, _l))
            # 3. Show network
            for _net_name, _network in self._networks.items():
                _lmd_show("Network[%s] total number of parameters: %.3f M. " % (_net_name, network_param_m(_network)))

        # Get data len
        len_data = {}
        for n, dl in self._data.items():
            if isinstance(dl, DataLoader):
                len_data[n] = len(dl.dataset)
            elif hasattr(dl, 'num_samples'):
                len_data[n] = dl.num_samples
        # 1. Get log keys
        if 'log_keys' in kwargs.keys():
            log_keys = kwargs['log_keys']
        else:
            log_keys = list(filter(lambda _k: isinstance(self._logs[_k], Logger), self._logs.keys()))
        # 2. Show in different ways
        # (1) Print to screen
        if not log_keys:
            _show_method(_lmd_show=lambda _x: print(_x))
        # (2) Write to each logger
        else:
            to_screen = True
            for key in log_keys:
                _show_method(_lmd_show=partial(self._logs[key].info_individually, to_screen=to_screen))
                to_screen = False


class EpochBatchBaseModel(BaseModel):
    """
    Epoch-batch based training model.
    ####################################################################################################################
    # Required cfg keys:
    ####################################################################################################################
        - epochs
        - freq_iter_log         (optional)
        - freq_epoch_chkpt      (optional)
        - (base) tfboard_dir    (optional)

    ####################################################################################################################
    # Components
    ####################################################################################################################
        - (base) self._logs['log_main']             (optional @ kwargs)
        - (base) self._logs['tfboard']              (optional @ cfg.args.tfboard_dir & kwargs)

        - (base) packs
            - 'log':                                (optional @ self._logs['log_main'] & kwargs)
            - 'tfboard':                            (optional @ self._logs['tfboard'] & kwargs)

        - (base) self._meters['timers']
            - 'io':                                 (optional @ kwargs)
            - 'optimize':                           (optional @ kwargs)

        - self._meters['i']
        - self._meters['counter_log']               (optional @ cfg.args.freq_iter_log)
        - self._meters['counter_chkpt']             (optional @ cfg.args.freq_epoch_chkpt)
    """
    ####################################################################################################################
    # Setup
    ####################################################################################################################

    def _set_logs(self, **kwargs):
        log_main_counters = kwargs.pop('log_main_counters') if 'log_main_counters' in kwargs.keys() else ['epoch', 'batch', 'iter']
        # Set logs
        super(EpochBatchBaseModel, self)._set_logs(log_main_counters=log_main_counters, **kwargs)

    def _set_meters(self, **kwargs):
        if chk_d(kwargs, 'disable_pbar'): kwargs['pbar_total'] = -1
        super(EpochBatchBaseModel, self)._set_meters(i={'iter': -1}, **kwargs)
        # --------------------------------------------------------------------------------------------------------------
        # Counters - log & chkpt
        # --------------------------------------------------------------------------------------------------------------
        if chk_ns(self._cfg.args, 'freq_iter_log', '>', 0):
            self._meters['counter_log'] = FreqCounter(self._cfg.args.freq_iter_log, iter_fetcher=lambda: self._meters['i']['iter'])
        if chk_ns(self._cfg.args, 'freq_epoch_chkpt', '>', 0):
            self._meters['counter_chkpt'] = FreqCounter(self._cfg.args.freq_epoch_chkpt)

    ####################################################################################################################
    # Train
    ####################################################################################################################

    def _set_data(self, **kwargs):
        super(EpochBatchBaseModel, self)._set_data(**kwargs)
        # Set progress bar.
        if 'pbar' in self._meters.keys() and 'train' in self._data.keys():
            self._meters['pbar'].total = self._cfg.args.epochs * len(self._data['train'])

    def _train_procedure(self, **kwargs):
        """
        Training procedure. 
        """
        # 1. Initialize packs used in training
        packs = self._init_packs()
        # 2. Training
        for epoch in range(self._cfg.args.load_from + 1, self._cfg.args.epochs):
            # 1. Train each batch
            # Start recording io time
            if 'io' in self._meters['timers']: self._meters['timers']['io'].resume()
            # Read batch data
            for batch_index, batch_data in enumerate(self._data['train']):
                # Deploy
                batch_iters, batch_data = self._deploy_batch_data(batch_data)
                self._meters['i']['iter'] += batch_iters
                # End recording io time & start recording optimize time
                if 'io' in self._meters['timers']: self._meters['timers']['io'].pause()
                # Batch optimization
                with self._meters['timers']('opt', void=chk_d(kwargs, 'disable_t_opt')):
                    self._set_to_train_mode()
                    self._train_batch(epoch, batch_index, batch_data, packs)
                ########################################################################################################
                # After-batch operations
                ########################################################################################################
                self._process_after_batch(epoch, batch_index, packs)
                try: self._meters['pbar'].update(1)
                except KeyError: pass
            # 2. Process after epoch
            early_stop = self._process_after_epoch(epoch, packs)
            if early_stop: return
        # Save final results
        self._save_checkpoint(self._cfg.args.epochs - 1)

    def _train_batch(self, epoch, batch_index, batch_data, packs):
        raise NotImplementedError

    def _process_after_batch(self, epoch, batch_index, packs, **kwargs):
        # Logging.
        self._process_log(dict(epoch=epoch, batch=batch_index, iter=self._meters['i']['iter']), packs, **kwargs)

    def _process_after_epoch(self, epoch, packs):
        """
        :rtype: Whether to early stop training (bool), none by default.
        """
        self._process_chkpt_and_lr(epoch)


class IterativeBaseModel(BaseModel):
    """
    Iteratively training model.
    ####################################################################################################################
    # Required cfg keys:
    ####################################################################################################################
        - steps
        - freq_freq_step_log                        (optional)
        - freq_step_chkpt                           (optional)
        - (base) tfboard_dir                        (optional)

    ####################################################################################################################
    # Components
    ####################################################################################################################
        - (base) self._logs['log_main']             (optional @ kwargs)
        - (base) self._logs['tfboard']              (optional @ cfg.args.tfboard_dir & kwargs)

        - (base) packs
            - 'log':                                (optional @ self._logs['log_main'] & kwargs)
            - 'tfboard':                            (optional @ self._logs['tfboard'] & kwargs)

        - (base) self._meters['timers']
            - 'io':                                 (optional @ kwargs)
            - 'optimize':                           (optional @ kwargs)

        - self._meters['i']
        - self._meters['counter_log']               (optional @ cfg.args.freq_step_log)
        - self._meters['counter_chkpt']             (optional @ cfg.args.freq_step_chkpt)

    """
    ####################################################################################################################
    # Setup
    ####################################################################################################################

    def _set_logs(self, **kwargs):
        log_main_counters = kwargs.pop('log_main_counters') if 'log_main_counters' in kwargs.keys() else ['epoch', 'batch', 'step', 'iter']
        # Set logs
        super(IterativeBaseModel, self)._set_logs(log_main_counters=log_main_counters, **kwargs)

    def _set_meters(self, **kwargs):
        kwargs['pbar_total'] = self._cfg.args.steps if not chk_d(kwargs, 'disable_pbar') else -1
        super(IterativeBaseModel, self)._set_meters(i={'step': -1, 'epoch': 0, 'batch': -1, 'iter': -1, 'num_cur_epoch': 0}, **kwargs)
        # --------------------------------------------------------------------------------------------------------------
        # Counters - log & chkpt
        # --------------------------------------------------------------------------------------------------------------
        if chk_ns(self._cfg.args, 'freq_step_log', '>', 0):
            self._meters['counter_log'] = FreqCounter(self._cfg.args.freq_step_log, iter_fetcher=lambda: self._meters['i']['step'])
        if chk_ns(self._cfg.args, 'freq_step_chkpt', '>', 0):
            self._meters['counter_chkpt'] = FreqCounter(self._cfg.args.freq_step_chkpt, iter_fetcher=lambda: self._meters['i']['step'])

    ####################################################################################################################
    # Train
    ####################################################################################################################

    def _set_data(self, **kwargs):
        super(IterativeBaseModel, self)._set_data(**kwargs)
        # Set num_train_samples in iterations.
        if 'num_train_samples' in kwargs.keys():
            self._meters['i']['num_train_samples'] = kwargs['num_train_samples']
        elif 'train' in self._data.keys():
            self._meters['i']['num_train_samples'] = self._data['train'].num_samples

    def _train_procedure(self, **kwargs):
        """
        Training procedure
        """
        # 1. Preliminaries
        packs = self._init_packs()
        # 2. Main
        while True:
            # Move forward.
            self._meters['i']['step'] += 1
            try: self._meters['pbar'].update(1)
            except KeyError: pass
            # 1. Train
            with self._meters['timers']('opt', void=chk_d(kwargs, 'dis_t_opt')):
                self._set_to_train_mode()
                self._train_step(packs)
            # 2. Process after train
            early_stop = self._process_after_step(packs)
            if early_stop: return
            """ Finish """
            if self._meters['i']['step'] + 1 == self._cfg.args.steps: break
        # Save final result
        self._save_checkpoint(self._meters['i']['step'])

    def _fetch_batch_data(self, **kwargs):
        # Fetch data & update iterations
        with self._meters['timers']('io'):
            # Fetch data
            batch_iters, batch_data_deployed = self._deploy_batch_data(next(self._data['train']))
            # Update iterations
            if not chk_d(kwargs, 'no_record'):
                # Iter_index
                self._meters['i']['iter'] += batch_iters
                # Epoch & batch & num_cur_epoch
                num_cur_epoch = self._meters['i']['num_cur_epoch'] + batch_iters
                num_train_samples = self._meters['i']['num_train_samples']
                if num_cur_epoch >= num_train_samples:
                    self._meters['i']['num_cur_epoch'] = num_cur_epoch % num_train_samples
                    self._meters['i']['batch'] = 0
                    self._meters['i']['epoch'] += 1
                else:
                    self._meters['i']['num_cur_epoch'] = num_cur_epoch
                    self._meters['i']['batch'] += 1
        # Return
        return batch_data_deployed

    def _train_step(self, packs):
        raise NotImplementedError

    def _process_log(self, packs, **kwargs):
        super(IterativeBaseModel, self)._process_log(self._meters['i'], packs, **kwargs)

    def _process_after_step(self, packs, **kwargs):
        """
        :rtype: Whether to early stop. None by default.
        """
        self._process_log(packs, **kwargs)
        self._process_chkpt_and_lr()
