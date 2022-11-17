
import os
import random
from tqdm import tqdm
from modellib.modules import *
from utils.criterions import *
from utils.evaluations import *
from custom_pkg.io.logger import Logger
from torchvision.utils import save_image
from custom_pkg.pytorch.base_models import IterativeBaseModel
from custom_pkg.pytorch.operations import sampling_z, summarize_losses_and_backward
from custom_pkg.basic.metrics import FreqCounter, TriggerPeriod, TriggerLambda, EMAPyTorchModel
from custom_pkg.basic.operations import fet_d, ValidDict, PathPreparation, BatchSlicerInt, BatchSlicerLenObj, IterCollector


class TrainerBase(IterativeBaseModel):
    """ Trainer. """
    def _set_logs(self, **kwargs):
        super(TrainerBase, self)._set_logs(**kwargs)
        # Get eval logs.
        for k in ['jacob-normal', 'jacob-ema']:
            self._logs['log_eval_%s' % k] = Logger(
                self._cfg.args.ana_train_dir, 'eval_%s' % k, formatted_prefix=self._cfg.args.desc, formatted_counters=['epoch', 'batch', 'step', 'iter'],
                append_mode=False if self._cfg.args.load_from == -1 else True, pbar=self._meters['pbar'])

    def _set_criterions(self):
        # Generator uniformity.
        self._criterions['gunif'] = GenUnifLoss(reduction=self._cfg.args.loss_gunif_reduction)

    def _set_optimizers(self):
        self._optimizers['gen'] = torch.optim.Adam(
            self._Gen.parameters(), lr=self._cfg.args.learning_rate, betas=(0.5, 0.9))
        self._optimizers['disc'] = torch.optim.Adam(
            self._Disc.parameters(), lr=self._cfg.args.learning_rate, betas=(0.5, 0.9))

    def _set_meters(self, **kwargs):
        super(TrainerBase, self)._set_meters()
        # --------------------------------------------------------------------------------------------------------------
        # Training
        # --------------------------------------------------------------------------------------------------------------
        self._meters['trigger_disc'] = TriggerPeriod(
            period=self._cfg.args.trigger_disc+self._cfg.args.trigger_gen, area=self._cfg.args.trigger_disc)
        self._meters['trigger_gunif'] = TriggerLambda(
            lmd_trigger=lambda _n: (self._cfg.args.version == 'v1' or self._meters['sv_ema'].avg is not None) and
                                   _n >= self._cfg.args.trigger_gunif and self._cfg.args.freq_step_gunif != -1 and
                                   self._meters['trigger_disc'].n_invalid % self._cfg.args.freq_step_gunif == 0,
            n_fetcher=lambda: self._meters['i']['step'])
        if self._cfg.args.version == 'v2':
            self._meters['trigger_update_sv_ema'] = TriggerLambda(
                lmd_trigger=lambda _n: _n >= self._cfg.args.trigger_update_sv_ema and self._cfg.args.freq_step_update_sv_ema != -1 and
                                       self._meters['trigger_disc'].n_invalid % self._cfg.args.freq_step_update_sv_ema == 0,
                n_fetcher=lambda: self._meters['i']['step'])
        self._meters['trigger_update_gen_ema'] = TriggerLambda(
            lmd_trigger=lambda _n: _n >= self._cfg.args.trigger_update_gen_ema and self._cfg.args.freq_step_update_gen_ema != -1 and
                                   self._meters['trigger_disc'].n_invalid % self._cfg.args.freq_step_update_gen_ema == 0,
            n_fetcher=lambda: self._meters['i']['step'])
        self._meters['sv_ema'] = kwargs['sv_ema']
        self._meters['gen_ema'] = EMAPyTorchModel(beta=0.99, model=self._get_generator())
        # --------------------------------------------------------------------------------------------------------------
        # Eval
        # --------------------------------------------------------------------------------------------------------------
        self._meters['counter_eval_vis'] = FreqCounter(
            self._cfg.args.freq_step_eval_vis, iter_fetcher=lambda: self._meters['i']['step'])
        self._meters['counter_eval_jacob'] = FreqCounter(
            self._cfg.args.freq_step_eval_jacob, iter_fetcher=lambda: self._meters['i']['step'])

    def _init_packs(self):
        return super(TrainerBase, self)._init_packs('vis', 'loss', log={'update_skip_none': True})

    ####################################################################################################################
    # Training
    ####################################################################################################################

    def _setup(self):
        super(TrainerBase, self)._setup()
        """ Deploy generator EMA to GPU. """
        if self._cfg.args.device == 'cuda': self._meters['gen_ema'].avg.cuda()

    def _sampling_z(self, batch_size):
        return sampling_z(batch_size, nz=self._cfg.args.nz, device=self._cfg.args.device, **self._cfg.fet_d(prefix="random_")).unsqueeze(-1).unsqueeze(-1)

    def _process_log(self, packs, **kwargs):

        def _lmd_generate_log():
            packs['tfboard'].update({
                'train/disc_pred': fet_d(packs['log'], prefix='pred_', replace=''),
                'train/loss': fet_d(packs['log'], prefix='loss_', replace=''),
                'train/sv': fet_d(packs['log'], prefix='sv_', replace='')
            })

        super(TrainerBase, self)._process_log(packs, lmd_generate_log=_lmd_generate_log)

    def _process_after_step(self, packs, **kwargs):
        # Clear losses.
        packs['log'].update(fet_d(packs['loss'], lmd_v=lambda _v: _v.item()))
        packs['loss'] = ValidDict()
        # 1. Logging.
        self._process_log(packs, **kwargs)
        # 2. Evaluation.
        if self._meters['counter_eval_vis'].check():
            self._eval_vis(packs)
        if self._meters['counter_eval_jacob'].check():
            self._eval_jacob(mode='normal')
            self._eval_jacob(mode='ema')
        """ lr & chkpt. """
        self._process_chkpt_and_lr()


class Trainer(TrainerBase):
    """ Trainer. """
    def _build_architectures(self):
        # --------------------------------------------------------------------------------------------------------------
        # Generator.
        # --------------------------------------------------------------------------------------------------------------
        generator = self._get_generator()
        # --------------------------------------------------------------------------------------------------------------
        # Discriminator.
        # --------------------------------------------------------------------------------------------------------------
        if self._cfg.args.disc == 'vanilla':
            discriminator = eval('Discriminator%dx%d' % (self._cfg.args.img_size, self._cfg.args.img_size))(
                nc=self._cfg.args.img_nc, ndf=self._cfg.args.ndf)
        elif self._cfg.args.disc == 'stylegan2':
            discriminator = DiscriminatorStyleGAN2(
                img_nc=self._cfg.args.img_nc, img_size=self._cfg.args.img_size, capacity=self._cfg.args.disc_capacity,
                max_nc=self._cfg.args.disc_max_nc, blur=self._cfg.args.disc_blur)
        else: raise ValueError
        """ Init. """
        super(Trainer, self)._build_architectures(Gen=generator, Disc=discriminator)

    def _get_generator(self):
        return (GeneratorV1 if self._cfg.args.version == 'v1' else GeneratorV2)(
            nz=self._cfg.args.nz, init_size=self._cfg.args.init_size, img_size=self._cfg.args.img_size,
            ncs=self._cfg.args.ncs, hidden_ncs=self._cfg.args.hidden_ncs, middle_ns_flows=self._cfg.args.middle_ns_flows)

    def _set_criterions(self):
        super(Trainer, self)._set_criterions()
        # Adversarial loss.
        if self._cfg.args.disc == 'vanilla':
            self._criterions['adv'] = BCELoss()
        elif self._cfg.args.disc == 'stylegan2':
            self._criterions['adv'] = StyleGAN2Loss()
            self._criterions['disc_gp'] = StyleGAN2GradientPenalty(weight=self._cfg.args.weight_disc_gp)

    def _set_meters(self):
        super(Trainer, self)._set_meters(sv_ema=EMA(beta=0.99))

    ####################################################################################################################
    # Training
    ####################################################################################################################

    def _deploy_batch_data(self, batch_data):
        images = batch_data.to(self._cfg.args.device)
        return images.size(0), images

    def _train_step(self, packs):
        ################################################################################################################
        # Critic
        ################################################################################################################
        if self._meters['trigger_disc'].check():
            # Clear grad.
            self._Gen.requires_grad_(False)
            self._optimizers['disc'].zero_grad()
            # Calculate loss & backward.
            for _ in range(self._cfg.args.n_grad_accum):
                # ------------------------------------------------------------------------------------------------------
                # 1. Get samples.
                # ------------------------------------------------------------------------------------------------------
                real_x = self._fetch_batch_data().requires_grad_(True)
                fake_x = self._Gen(self._sampling_z(real_x.size(0)))[:, :self._cfg.args.img_nc].detach()
                pred_real, pred_fake = self._Disc(real_x), self._Disc(fake_x)
                # ------------------------------------------------------------------------------------------------------
                # 2. Calculate losses & backward.
                # ------------------------------------------------------------------------------------------------------
                disc_losses = {}
                # (1) Adversarial.
                if self._cfg.args.disc == 'vanilla':
                    loss_disc_real, loss_disc_fake = self._criterions['adv']([pred_real, pred_fake], [True, False], lmd=self._cfg.args.lambda_disc_adv)
                    disc_losses.update({'loss_disc_real': loss_disc_real, 'loss_disc_fake': loss_disc_fake})
                elif self._cfg.args.disc == 'stylegan2':
                    # Adversarial.
                    disc_losses['loss_disc_adv'] = self._criterions['adv'](mode='disc', output_fake=pred_fake, output_real=pred_real, lmd=self._cfg.args.lambda_disc_adv)
                    # Gradient penalty.
                    if self._meters['trigger_disc'].n_valid % self._cfg.args.freq_step_disc_gp == 0:
                        disc_losses['loss_disc_gp'] = self._criterions['disc_gp'](real_x, pred_real, lmd=self._cfg.args.lambda_disc_gp)
                else: raise ValueError
                """ Backward. """
                summarize_losses_and_backward(*disc_losses.values(), weight=1.0/self._cfg.args.n_grad_accum)
                # ------------------------------------------------------------------------------------------------------
                """ Saving """
                # ------------------------------------------------------------------------------------------------------
                # (1) Visualization.
                vis_results = {}
                for k in ['real_x', 'fake_x']:
                    v = eval(k).detach().cpu()
                    vis_results[k] = (torch.cat([packs['vis'][k], v]) if packs['vis'] else v)[-self._cfg.args.eval_vis_sqrt_num**2:]
                packs['vis'].update(vis_results)
                # (2) Logging: disc prediction & losses.
                packs['log'].update({'pred_real_disc': pred_real.mean().item(), 'pred_fake_disc': pred_fake.mean().item()})
                packs['loss'].update(disc_losses)
            # Update disc.
            self._optimizers['disc'].step()
            self._Gen.requires_grad_(True)
        ################################################################################################################
        # Generator
        ################################################################################################################
        else:
            # Clear grad.
            self._Disc.requires_grad_(False)
            self._optimizers['gen'].zero_grad()
            # Calculate loss & backward.
            for _ in range(self._cfg.args.n_grad_accum):
                ########################################################################################################
                # 1. Get samples.
                ########################################################################################################
                real_z = self._sampling_z(self._cfg.args.batch_size)
                fake_x = self._Gen(real_z)
                pred_fake = self._Disc(fake_x[:, :self._cfg.args.img_nc])
                """ Saving """
                packs['log'].update({'pred_fake_gen': pred_fake.mean().item()})
                ########################################################################################################
                # 2. Calculate losses & backward.
                ########################################################################################################
                gen_losses = {}
                # ------------------------------------------------------------------------------------------------------
                # Adversarial loss depending on different discriminator.
                # ------------------------------------------------------------------------------------------------------
                if self._cfg.args.disc == 'vanilla':
                    gen_losses['loss_gen_fake'] = self._criterions['adv'](pred_fake, True, lmd=self._cfg.args.lambda_gen_adv)
                elif self._cfg.args.disc == 'stylegan2':
                    gen_losses['loss_gen_fake'] = self._criterions['adv'](mode='gen', output=pred_fake, lmd=self._cfg.args.lambda_gen_adv)
                else: raise ValueError
                """ Backward. """
                summarize_losses_and_backward(gen_losses['loss_gen_fake'], weight=1.0/self._cfg.args.n_grad_accum)
                # ------------------------------------------------------------------------------------------------------
                # Generator uniformity.
                # ------------------------------------------------------------------------------------------------------
                """ Regularization. """
                if self._meters['trigger_gunif'].check():
                    compute_logsv = 'max' if (self._meters['trigger_disc'].n_invalid//self._cfg.args.freq_step_gunif) % 2 == 0 else 'min'
                    # (1) Compute logsv. (batch, ).
                    compute_logsv_kwargs = {} if self._cfg.args.version == 'v1' else \
                        {'sv_ema_r': 1.0/self._meters['sv_ema'].avg.to(self._cfg.args.device)[None, :, None, None]}
                    if compute_logsv == 'max': logsv = self._Gen.compute_max_logsv(real_z, **compute_logsv_kwargs, sn_power=self._cfg.args.sn_power)
                    else: logsv = self._Gen.compute_min_logsv(fake_x, **compute_logsv_kwargs, sn_power=self._cfg.args.sn_power)
                    # (2) Compute loss.
                    gen_losses['loss_gen_gunif'] = self._criterions['gunif'](
                        logsv, **({'target': self._meters['sv_ema'].avg} if self._cfg.args.version == 'v1' else {}), lmd=self._cfg.args.lambda_gen_gunif)
                    """ Backward. """
                    summarize_losses_and_backward(gen_losses['loss_gen_gunif'], weight=1.0/self._cfg.args.n_grad_accum)
                    """ Update & saving. """
                    if self._cfg.args.version == 'v1':
                        self._meters['sv_ema'].update_average(new=logsv.mean().item())
                        packs['log'].update({'sv_ema': math.exp(self._meters['sv_ema'].avg)})
                    packs['log'].update({'sv_%s' % compute_logsv: logsv.detach().exp().mean().item()})
                # ------------------------------------------------------------------------------------------------------
                """ Update EMA for SV. """
                if self._cfg.args.version == 'v2' and self._meters['trigger_update_sv_ema'].check():
                    # Compute svs. (n_samples, nz) -> (nz, )
                    svs = torch.cat([self._Gen.compute_svs(
                        z=self._sampling_z(bsize), jacob_bsize=min(self._cfg.args.update_sv_ema_jacob_bsize, self._cfg.args.nz)
                    ) for bsize in BatchSlicerInt(self._cfg.args.update_sv_ema_n_samples, self._cfg.args.update_sv_ema_batch_size)]).mean(dim=0).cpu()
                    """ Update & saving. """
                    self._meters['sv_ema'].update_average(new=svs)
                    packs['log'].update({'sv_ema': self._meters['sv_ema'].avg.mean().item()})
                # ------------------------------------------------------------------------------------------------------
                packs['loss'].update(gen_losses)
            # Update generator.
            self._optimizers['gen'].step()
            self._Disc.requires_grad_(True)
            # Update EMA for generator.
            if self._meters['trigger_update_gen_ema'].check():
                self._meters['gen_ema'].update_average(self._Gen)

    ####################################################################################################################
    # Evaluation
    ####################################################################################################################

    def _eval_vis(self, packs):
        if not packs['vis']: return
        n_cats = len(packs['vis'])
        # 1. Concat real & fake.
        x = torch.cat([packs['vis'][k].unsqueeze(1) for k in packs['vis']], dim=1)
        x = x.reshape(x.size(0)*n_cats, *x.size()[2:])*0.5+0.5
        # 2. Visualize samples.
        with PathPreparation(self._cfg.args.eval_vis_dir) as save_dir:
            save_image(x, os.path.join(save_dir, 'step[%d].png' % self._meters['i']['step']), nrow=self._cfg.args.eval_vis_sqrt_num*n_cats)

    @api_empty_cache
    def _eval_jacob(self, mode):
        assert mode in ['normal', 'ema']
        if mode == 'ema' and not self._meters['gen_ema'].initialized: return
        generator = self._Gen if mode == 'normal' else self._meters['gen_ema'].avg
        # --------------------------------------------------------------------------------------------------------------
        # Get evaluation results.
        # --------------------------------------------------------------------------------------------------------------
        # 1. Init results.
        results = IterCollector()
        # 2. Collect from batch.
        pbar = tqdm(total=self._cfg.args.eval_jacob_num_samples, desc="Evaluating generator's Jacobian", disable=None)
        for batch_size in BatchSlicerInt(self._cfg.args.eval_jacob_num_samples, self._cfg.args.eval_jacob_batch_size):
            batch_ret = generator.eval_jacob(z=self._sampling_z(batch_size), jacob_size=self._cfg.args.eval_jacob_ag_bsize)
            """ Saving """
            results.collect(batch_ret)
            """ Progress """
            pbar.update(batch_size)
        pbar.close()
        # 3. Get results.
        results = results.pack()
        # (1) Singular values.
        svs, sv_ema = results.pop('svs'), self._meters['sv_ema'].avg
        if sv_ema is not None:
            if not isinstance(sv_ema, torch.Tensor): sv_ema = torch.tensor([sv_ema], dtype=torch.float32)
            normal = svs / sv_ema[None].numpy()
            results.update({'jacob@normal@avg': normal.mean(axis=1), 'jacob@normal@std': normal.std(axis=1, ddof=1)})
        else:
            results.update({'jacob@normal@std': (svs / svs.mean(axis=0, keepdims=True)).std(axis=1, ddof=1)})
        # (2) Logdet.
        logdet = results.pop('logdet')
        results = {k: v.mean().item() for k, v in results.items()}
        results.update({"logdet@avg": logdet.mean().item(), "logdet@std": logdet.std(ddof=1).item()})
        # --------------------------------------------------------------------------------------------------------------
        # Logging.
        # --------------------------------------------------------------------------------------------------------------
        """ Logger. """
        self._logs['log_eval_jacob-%s' % mode].info_formatted(counters=self._meters['i'], items=results)
        """ Tfboard. """
        self._logs['tfboard'].add_multi_scalars(multi_scalars={
            'eval/%s/%s' % (mode, k): {'': v} for k, v in results.items()
        }, global_step=self._meters['i']['step'])
        # --------------------------------------------------------------------------------------------------------------
        # Visualizing singular values.
        # --------------------------------------------------------------------------------------------------------------
        with PathPreparation(self._cfg.args.eval_sv_dir + "-%s" % mode) as save_dir:
            vis_singular_values(
                sv_ema.numpy() if sv_ema is not None else None, svs, os.path.join(save_dir, 'step[%d].png' % self._meters['i']['step']))


class TrainerConditional(TrainerBase):
    """ Trainer. """
    def _build_architectures(self):
        # --------------------------------------------------------------------------------------------------------------
        # 1. Generator.
        # --------------------------------------------------------------------------------------------------------------
        generator = self._get_generator()
        # --------------------------------------------------------------------------------------------------------------
        # 2. Discriminator.
        # --------------------------------------------------------------------------------------------------------------
        if self._cfg.args.disc == 'vanilla':
            discriminator = eval('Discriminator%dx%dConditional' % (self._cfg.args.img_size, self._cfg.args.img_size))(
                n_classes=self._cfg.args.n_classes, nc=self._cfg.args.img_nc, ndf=self._cfg.args.ndf)
        else: raise NotImplementedError
        """ Init. """
        super(TrainerConditional, self)._build_architectures(Gen=generator, Disc=discriminator)

    def _get_generator(self):
        if self._cfg.args.version == 'v1':
            return GeneratorV1Conditional(
                n_classes=self._cfg.args.n_classes, nz=self._cfg.args.nz, init_size=self._cfg.args.init_size, img_size=self._cfg.args.img_size,
                ncs=self._cfg.args.ncs, hidden_ncs=self._cfg.args.hidden_ncs, middle_ns_flows=self._cfg.args.middle_ns_flows)
        else: raise NotImplementedError

    def _set_criterions(self):
        super(TrainerConditional, self)._set_criterions()
        # Adversarial loss.
        if self._cfg.args.disc == 'vanilla':
            self._criterions['adv'] = CELoss(n_classes=self._cfg.args.n_classes)
        else: raise NotImplementedError

    def _set_meters(self):
        super(TrainerConditional, self)._set_meters(sv_ema=EMAMultiClass(n_classes=self._cfg.args.n_classes, beta=0.99))

    ####################################################################################################################
    # Training
    ####################################################################################################################

    def _deploy_batch_data(self, batch_data):
        images, labels = map(lambda _x: _x.to(self._cfg.args.device), batch_data)
        return images.size(0), (images, labels)

    def _sampling_cat(self, batch_size):
        return torch.randint(self._cfg.args.n_classes, size=(batch_size, ), dtype=torch.int64, device=self._cfg.args.device)

    def _sampling(self, batch_size):
        return self._sampling_z(batch_size), self._sampling_cat(batch_size)

    def _train_step(self, packs):
        ################################################################################################################
        # Critic
        ################################################################################################################
        if self._meters['trigger_disc'].check():
            # Clear grad.
            self._Gen.requires_grad_(False)
            self._optimizers['disc'].zero_grad()
            # Calculate loss & backward.
            for _ in range(self._cfg.args.n_grad_accum):
                # ------------------------------------------------------------------------------------------------------
                # 1. Get samples.
                # ------------------------------------------------------------------------------------------------------
                real_x, real_y = self._fetch_batch_data()
                fake_x = self._Gen(self._sampling(real_x.size(0)))[:, :self._cfg.args.img_nc].detach()
                pred_real, pred_fake = self._Disc(real_x), self._Disc(fake_x)
                # ------------------------------------------------------------------------------------------------------
                # 2. Calculate losses & backward.
                # ------------------------------------------------------------------------------------------------------
                disc_losses = {}
                # (1) Adversarial.
                if self._cfg.args.disc == 'vanilla':
                    loss_disc_real, loss_disc_fake = self._criterions['adv']([pred_real, pred_fake], [real_y, False], lmd=self._cfg.args.lambda_disc_adv)
                    disc_losses.update({'loss_disc_real': loss_disc_real, 'loss_disc_fake': loss_disc_fake})
                else: raise NotImplementedError
                """ Backward. """
                summarize_losses_and_backward(*disc_losses.values(), weight=1.0/self._cfg.args.n_grad_accum)
                # ------------------------------------------------------------------------------------------------------
                """ Saving """
                # ------------------------------------------------------------------------------------------------------
                # (1) Visualization.
                vis_results = {}
                for k in ['real_x', 'fake_x']:
                    v = eval(k).detach().cpu()
                    vis_results[k] = (torch.cat([packs['vis'][k], v]) if packs['vis'] else v)[-self._cfg.args.eval_vis_sqrt_num**2:]
                packs['vis'].update(vis_results)
                # (2) Logging: disc prediction & losses.
                packs['log'].update({
                    'pred_real_disc': np.choose(real_y.cpu().numpy(), pred_real.softmax(dim=1).detach().cpu().numpy().T).mean().item(),
                    'pred_fake_disc': pred_fake.softmax(dim=1)[:, -1].mean().item()})
                packs['loss'].update(disc_losses)
            # Update disc.
            self._optimizers['disc'].step()
            self._Gen.requires_grad_(True)
        ################################################################################################################
        # Generator
        ################################################################################################################
        else:
            # Clear grad.
            self._Disc.requires_grad_(False)
            self._optimizers['gen'].zero_grad()
            # Calculate loss & backward.
            for _ in range(self._cfg.args.n_grad_accum):
                ########################################################################################################
                # 1. Get samples.
                ########################################################################################################
                latent = self._sampling(self._cfg.args.batch_size)
                fake_x = self._Gen(latent)
                pred_fake = self._Disc(fake_x[:, :self._cfg.args.img_nc])
                """ Saving """
                packs['log'].update({'pred_fake_gen': np.choose(latent[1].cpu().numpy(), pred_fake.softmax(dim=1).detach().cpu().numpy().T).mean().item()})
                ########################################################################################################
                # 2. Calculate losses & backward.
                ########################################################################################################
                gen_losses = {}
                # ------------------------------------------------------------------------------------------------------
                # Adversarial loss depending on different discriminator.
                # ------------------------------------------------------------------------------------------------------
                if self._cfg.args.disc == 'vanilla':
                    gen_losses['loss_gen_fake'] = self._criterions['adv'](pred_fake, latent[1], lmd=self._cfg.args.lambda_gen_adv)
                else: raise NotImplementedError
                """ Backward. """
                summarize_losses_and_backward(gen_losses['loss_gen_fake'], weight=1.0/self._cfg.args.n_grad_accum)
                # ------------------------------------------------------------------------------------------------------
                # Generator uniformity.
                # ------------------------------------------------------------------------------------------------------
                if self._meters['trigger_gunif'].check():
                    compute_logsv = 'max' if (self._meters['trigger_disc'].n_invalid//self._cfg.args.freq_step_gunif) % 2 == 0 else 'min'
                    # (1) Compute logsv. (batch, ).
                    if compute_logsv == 'max': logsv = self._Gen.compute_max_logsv(latent, sn_power=self._cfg.args.sn_power)
                    else: logsv = self._Gen.compute_min_logsv(fake_x, sn_power=self._cfg.args.sn_power)
                    # (2) Compute loss.
                    gen_losses['loss_gen_gunif'] = self._criterions['gunif'](
                        logsv, target=self._meters['sv_ema'].avg[latent[1]] if self._meters['sv_ema'].avg is not None else None,
                        lmd=self._cfg.args.lambda_gen_gunif)
                    """ Backward. """
                    summarize_losses_and_backward(gen_losses['loss_gen_gunif'], weight=1.0/self._cfg.args.n_grad_accum)
                    """ Update & saving. """
                    self._meters['sv_ema'].update_average(new=(logsv, latent[1]))
                    packs['log'].update({'sv_ema': self._meters['sv_ema'].avg.exp().mean().item()})
                    packs['log'].update({'sv_%s' % compute_logsv: logsv.detach().exp().mean().item()})
                # ------------------------------------------------------------------------------------------------------
                packs['loss'].update(gen_losses)
            # Update encoder & generator.
            self._optimizers['gen'].step()
            self._Disc.requires_grad_(True)
            # Update EMA for generator.
            if self._meters['trigger_update_gen_ema'].check():
                self._meters['gen_ema'].update_average(self._Gen)

    ####################################################################################################################
    # Evaluation
    ####################################################################################################################

    def _eval_vis(self, packs):
        if not packs['vis']: return
        n_cats = len(packs['vis'])
        # --------------------------------------------------------------------------------------------------------------
        # Training
        # --------------------------------------------------------------------------------------------------------------
        # 1. Concat real & fake.
        x = torch.cat([packs['vis'][k].unsqueeze(1) for k in packs['vis']], dim=1)
        x = x.reshape(x.size(0)*n_cats, *x.size()[2:])*0.5+0.5
        # 2. Visualize samples.
        with PathPreparation(self._cfg.args.eval_vis_dir, 'training') as save_dir:
            save_image(x, os.path.join(save_dir, 'step[%d].png' % self._meters['i']['step']), nrow=self._cfg.args.eval_vis_sqrt_num*n_cats)
        # --------------------------------------------------------------------------------------------------------------
        # Categorical generation
        # --------------------------------------------------------------------------------------------------------------
        n_imgs_per_cat = self._cfg.args.eval_vis_sqrt_num**2
        dataset = self._data['train'].dataset
        # Visualize samples
        with PathPreparation(self._cfg.args.eval_vis_dir, 'categorical_generation', 'step[%d]' % self._meters['i']['step']) as save_dir:
            for cat in range(self._cfg.args.n_classes):
                # Randomly select real samples.
                x_real = torch.cat([dataset[i][0].unsqueeze(0) for i in random.choices(dataset.sample_indices[cat], k=n_imgs_per_cat)]).to(self._cfg.args.device)
                # Randomly generate fake samples.
                x_fake = self._Gen((self._sampling_z(n_imgs_per_cat), cat*torch.ones(size=(n_imgs_per_cat, ), dtype=torch.int64, device=self._cfg.args.device)))[:, :self._cfg.args.img_nc]
                """ Visualize """
                x = torch.cat([x_real.unsqueeze(1), x_fake.unsqueeze(1)], dim=1)
                x = x.reshape(x.size(0)*2, *x.size()[2:])*0.5+0.5
                save_image(x, os.path.join(save_dir, 'cat[%d].png' % cat), nrow=self._cfg.args.eval_vis_sqrt_num*2)

    @api_empty_cache
    def _eval_jacob(self, mode):
        assert mode in ['normal', 'ema']
        if mode == 'ema' and not self._meters['gen_ema'].initialized: return
        generator = self._Gen if mode == 'normal' else self._meters['gen_ema'].avg
        # --------------------------------------------------------------------------------------------------------------
        # Get evaluation results.
        # --------------------------------------------------------------------------------------------------------------
        # 1. Init results.
        results = IterCollector()
        # 2. Collect from batch.
        pbar = tqdm(total=self._cfg.args.eval_jacob_num_samples, desc="Evaluating generator's Jacobian", disable=None)
        categories = torch.arange(self._cfg.args.n_classes, dtype=torch.int64, device=self._cfg.args.device).unsqueeze(1).expand(
            self._cfg.args.n_classes, self._cfg.args.eval_jacob_num_samples//self._cfg.args.n_classes).reshape(-1, )
        for batch_cat in BatchSlicerLenObj(categories, batch_size=self._cfg.args.eval_jacob_batch_size):
            batch_ret = generator.eval_jacob((self._sampling_z(len(batch_cat)), batch_cat), jacob_size=self._cfg.args.eval_jacob_ag_bsize)
            """ Saving """
            results.collect(batch_ret)
            """ Progress """
            pbar.update(len(batch_cat))
        pbar.close()
        # 3. Get results.
        results = results.pack()
        # (1) Singular values.
        svs, sv_ema = results.pop('svs'), self._meters['sv_ema'].avg
        if sv_ema is not None:
            if len(sv_ema.size()) == 1: sv_ema = sv_ema[:, None]
            normal = svs / sv_ema[categories].cpu().numpy()
            results.update({'jacob@normal@avg': normal.mean(axis=1), 'jacob@normal@std': normal.std(axis=1, ddof=1)})
        else:
            svs_cat = svs.reshape(self._cfg.args.n_classes, -1, svs.shape[1])
            results.update({'jacob@normal@std': (svs_cat / svs_cat.mean(axis=1, keepdims=True)).std(axis=2, ddof=1).mean()})
        # (2) Logdet.
        logdet = results.pop('logdet')
        results = {k: v.mean().item() for k, v in results.items()}
        results.update({"logdet@avg": logdet.mean().item(), "logdet@std": logdet.reshape(self._cfg.args.n_classes, -1).std(axis=1, ddof=1).mean().item()})
        # --------------------------------------------------------------------------------------------------------------
        # Logging.
        # --------------------------------------------------------------------------------------------------------------
        """ Logger. """
        self._logs['log_eval_jacob-%s' % mode].info_formatted(counters=self._meters['i'], items=results)
        """ Tfboard. """
        self._logs['tfboard'].add_multi_scalars(multi_scalars={
            'eval/%s/%s' % (mode, k): {'': v} for k, v in results.items()
        }, global_step=self._meters['i']['step'])
        # --------------------------------------------------------------------------------------------------------------
        # Visualizing singular values.
        # --------------------------------------------------------------------------------------------------------------
        with PathPreparation(self._cfg.args.eval_sv_dir + "-%s" % mode) as save_dir:
            vis_singular_values(None, svs, os.path.join(save_dir, 'step[%d].png' % self._meters['i']['step']))
