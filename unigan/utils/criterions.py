
from torch import nn
from utils.operations import *
from torch.nn import functional as F
from custom_pkg.pytorch.operations import BaseCriterion


########################################################################################################################
# Adversarial losses.
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# GAN loss (vanilla)
# ----------------------------------------------------------------------------------------------------------------------

class BCELoss(BaseCriterion):
    """
    Binary cross entropy loss for unconditional generation.
    """
    def __init__(self, lmd=None):
        super(BCELoss, self).__init__(lmd=lmd)
        # Criterion
        self._loss = nn.BCELoss()

    def _call_method(self, ipt, target):
        """ Calculate loss. """
        if isinstance(ipt, list):
            assert isinstance(target, list)
            return [self._call_method(i, t) for i, t in zip(ipt, target)]

        # Meta method.
        assert target in [True, False]
        # Get target
        target = (torch.ones if target is True else torch.zeros)(size=(len(ipt), ), dtype=ipt.dtype, device=ipt.device)
        # Calculate loss
        return self._loss(ipt, target)


class CELoss(BaseCriterion):
    """
    Cross entropy loss for conditional generation.
    """
    def __init__(self, n_classes, lmd=None):
        super(CELoss, self).__init__(lmd=lmd)
        # Config.
        self._n_classes = n_classes
        # Criterion
        self._loss = nn.CrossEntropyLoss()

    def _call_method(self, ipt, target):
        """
        :param ipt: (batch, n_classes+1)
        :param target:
            - Tensor (batch, ), where elements represent category (range in 0 ~ n_classes-1).
            - False, where represents the last category (value is n_classes).
        :return: (1, )
        """
        """ Calculate loss. """
        if isinstance(ipt, list):
            assert isinstance(target, list)
            return [self._call_method(i, t) for i, t in zip(ipt, target)]

        """ Meta method. """
        if target is False: target = self._n_classes * torch.ones(size=(len(ipt), ), dtype=torch.int64, device=ipt.device)
        return self._loss(ipt, target.to(torch.int64))


# ----------------------------------------------------------------------------------------------------------------------
# StyleGAN2 loss
# ----------------------------------------------------------------------------------------------------------------------

def gradient_penalty(images, output, weight=10.0):
    gradients = autograd.grad(outputs=output, inputs=images, grad_outputs=torch.ones(output.size(), device=images.device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.reshape(images.shape[0], -1)
    return weight * ((gradients.norm(2, dim=1)-1)**2).mean()


class StyleGAN2Loss(BaseCriterion):
    """
    GAN objectives.
    """
    def _call_method(self, mode, **kwargs):
        assert mode in ['disc', 'gen']
        # 1. Discriminator
        if mode == 'disc':
            loss = (F.relu(1 + kwargs['output_real']) + F.relu(1 - kwargs['output_fake'])).mean()
        # 2. Generator
        else:
            loss = kwargs['output'].mean()
        # Return
        return loss


class StyleGAN2GradientPenalty(BaseCriterion):
    """
    Gradient Penalty.
    """
    def __init__(self, weight=10.0, lmd=None):
        super(StyleGAN2GradientPenalty, self).__init__(lmd=lmd)
        # Config
        self._weight = weight

    def _call_method(self, images, output):
        return gradient_penalty(images, output, weight=self._weight)


########################################################################################################################
# Generator uniformity loss.
########################################################################################################################

class GenUnifLoss(BaseCriterion):
    """
    The generator uniformity loss.
    """
    def __init__(self, reduction, lmd=None):
        super(GenUnifLoss, self).__init__(lmd=lmd)
        # Config.
        assert reduction in ['sum', 'mean']
        self._reduction = reduction

    def _call_method(self, logsv, **kwargs):
        """ Compute loss & update EMA.
        :param logsv: (batch, ).
        """
        if 'target' in kwargs:
            ema_avg = kwargs['target']
            if ema_avg is None: return None
            loss = (logsv - ema_avg)**2
        else:
            loss = logsv**2
        # Return
        return loss.sum() if self._reduction == 'sum' else loss.mean()
