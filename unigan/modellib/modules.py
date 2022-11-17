
import math
from modellib.layers import *
from utils.operations import *
from kornia.filters import filter2D


########################################################################################################################
# Generator.
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# Unconditional generator.
# ----------------------------------------------------------------------------------------------------------------------

class Generator(InvertibleModule):
    """
    Decoder.
    """
    def __init__(self, nz, init_size, img_size, ncs, hidden_ncs, middle_ns_flows):
        """
        For example:
            - ncs:                  128(4,4),   64(8,8),   32(16,16),  16(32,32),   4(64,64)
            - hidden_ncs:           256,        128,       64,         32,          16
            - middle_ns_flows:                  3,         3,          3,           3
        """
        super(Generator, self).__init__()
        ################################################################################################################
        # Architecture.
        ################################################################################################################
        n_layers = int(math.log2(img_size//init_size))
        assert init_size*2**n_layers == img_size
        # 1. Init layer. (nz, 1, 1) -> (ncs[0], init_size, init_size)
        self._init_diu_block = InitDiuBlock(nz, ncs[0], init_size, hidden_ncs[0])
        # 2. Middle layers. (ncs[i], init_size*2^i, init_size*2^i) -> (ncs[i+1], init_size*2^(i+1), init_size*2^(i+1))
        self._middle_dius, self._middle_blocks = nn.ModuleList([]), nn.ModuleList([])
        for index in range(n_layers):
            # (1) DimIncreaseUnsqueeze2.
            self._middle_dius.append(DimIncreaseUnsqueeze2(ncs[index], ncs[index+1]))
            # (2) Block.
            self._middle_blocks.append(Block(
                ncs[index+1], init_size*2**(index+1), hidden_ncs[index], hidden_ncs[index+1], n_flows=middle_ns_flows[index]))
        # 3. Final conv.
        self._final_conv = InvConv2d1x1Fixed(ncs[-1], ncs[-1])
        self._final_tanh = Tanh()

    def forward(self, z, linearize=False):
        # 1. Init layer.
        output = self._init_diu_block(z, linearize=linearize)
        # 2. Middle layers.
        for middle_diu, middle_block in zip(self._middle_dius, self._middle_blocks):
            output = middle_diu(output)
            output = middle_block(output, linearize=linearize)
        # 3. Final.
        output = self._final_conv(output)
        output = self._final_tanh(output, linearize=linearize)
        # Return
        return output

    def linearized_transpose(self, x):
        # 1. Final.
        output = self._final_tanh.linearized_transpose(x)
        output = self._final_conv.linearized_transpose(output)
        # 2. Middle layers.
        for middle_diu, middle_block in zip(self._middle_dius[::-1], self._middle_blocks[::-1]):
            output = middle_block.linearized_transpose(output)
            output = middle_diu.linearized_transpose(output)
        # 3. Init layer.
        output = self._init_diu_block.linearized_transpose(output)
        # Return
        return output

    def inverse(self, x, linearize=False):
        # 1. Final.
        output = self._final_tanh.inverse(x, linearize=linearize)
        output = self._final_conv.inverse(output)
        # 2. Middle layers.
        for middle_diu, middle_block in zip(self._middle_dius[::-1], self._middle_blocks[::-1]):
            output = middle_block.inverse(output, linearize=linearize)
            output = middle_diu.inverse(output)
        # 3. Init layer.
        output = self._init_diu_block.inverse(output, linearize=linearize)
        # Return
        return output

    def inverse_lt(self, eps):
        # 1. Init layer.
        output = self._init_diu_block.inverse_lt(eps)
        # 2. Middle layers.
        for middle_diu, middle_block in zip(self._middle_dius, self._middle_blocks):
            output = middle_diu.inverse_lt(output)
            output = middle_block.inverse_lt(output)
        # 3. Final.
        output = self._final_conv.inverse_lt(output)
        output = self._final_tanh.inverse_lt(output)
        # Return
        return output

    # ------------------------------------------------------------------------------------------------------------------
    # Evaluation.
    # ------------------------------------------------------------------------------------------------------------------

    @api_empty_cache
    def eval_jacob(self, z, jacob_size=16):
        """ Evaluating Jacobian-related metrics. """
        ################################################################################################################
        # Compute J (batch, nx, nz) & JTJ (batch, nz, nz).
        ################################################################################################################
        jacob = autograd_jacob(z, func=self.forward, bsize=jacob_size)
        jtj = torch.matmul(jacob.transpose(1, 2), jacob)
        ################################################################################################################
        # Statistics
        ################################################################################################################
        # 1. Jacobian orthogonality. (batch, )
        ortho_sign = measure_ortho(jtj)
        # 2. Singular values. (batch, nz).
        svs = np.sqrt(np.concatenate([np.linalg.svd(j, compute_uv=False)[np.newaxis] for j in jtj.cpu().numpy()]))
        # 3. Logdet of Jacobian. (batch, ).
        logdet = np.array([np.linalg.slogdet(_jtj)[1] for _jtj in jtj.cpu().numpy()], dtype=np.float32)*0.5
        # Return
        return {'jacob@ortho': ortho_sign, 'svs': svs, 'logdet': logdet}


class GeneratorV1(Generator):
    """
    Decoder.
    """
    def compute_max_logsv(self, z, sn_power=3):
        """ Compute logarithm of the maximum singular values.
        :param z: The latent codes.
        :param sn_power: The power of iterations for estimating spectral norm.
        """
        # 1. Forward & LT.
        # (1) Forward & linearize.
        z = z.requires_grad_(True)
        output = self.forward(z, linearize=True)
        # (2) LT.
        x_t = torch.randn_like(output).requires_grad_(True)
        output_t = self.linearized_transpose(x_t)
        # 2. Compute.
        logsv_max = fast_approx_logsn(z, output, x_t, output_t, sn_power=sn_power)
        # Return
        return logsv_max

    def compute_min_logsv(self, output, sn_power=3):
        """ Compute logarithm of the minimum singular values.
        :param output: The output of the generator.
        :param sn_power: The power of iterations for estimating spectral norm.
        """
        output = output.detach().clamp(-0.9, 0.9)
        # 1. Inverse & LT.
        # (1) Inverse & linearize.
        output = output.requires_grad_(True)
        z = self.inverse(output, linearize=True)
        # (2) Inverted LT.
        x_t = torch.randn_like(z).requires_grad_(True)
        output_t = self.inverse_lt(x_t)
        # 2. Compute.
        logsv_max = fast_approx_logsn(output, z, x_t, output_t, sn_power=sn_power)
        # Return
        return -logsv_max


class GeneratorV2(Generator):
    """
    Decoder.
    """
    def _fwd(self, e, z, sv_ema_r, linearize=False):
        z = (z - sv_ema_r).detach() + sv_ema_r * e
        output = self.forward(z, linearize=linearize)
        # Return
        return output

    def _lt(self, eps, sv_ema_r):
        output = self.linearized_transpose(eps)
        output = sv_ema_r * output
        # Return
        return output

    def _inv(self, x, sv_ema_r, linearize=False):
        output = self.inverse(x, linearize=linearize)
        output = (output - (output - sv_ema_r).detach()) / sv_ema_r
        # Return
        return output

    def _inv_lt(self, eps, sv_ema_r):
        eps = eps / sv_ema_r
        output = self.inverse_lt(eps)
        # Return
        return output

    def compute_max_logsv(self, z, sv_ema_r, sn_power=3):
        """ Compute logarithm of the maximum singular values.
        :param z: The latent codes.
        :param sv_ema_r: The reciprocal of the singular value EMA.
        :param sn_power: The power of iterations for estimating spectral norm.
        """
        # 1. Forward & LT.
        # (1) Forward & linearize.
        e = torch.ones_like(z).requires_grad_(True)
        output = self._fwd(e, z, sv_ema_r, linearize=True)
        # (2) LT.
        x_t = torch.randn_like(output).requires_grad_(True)
        output_t = self._lt(x_t, sv_ema_r)
        # 2. Compute.
        logsv_max = fast_approx_logsn(e, output, x_t, output_t, sn_power=sn_power)
        # Return
        return logsv_max

    def compute_min_logsv(self, output, sv_ema_r, sn_power=3):
        """ Compute logarithm of the minimum singular values.
        :param output: The output of the generator.
        :param sv_ema_r: The reciprocal of the singular value EMA.
        :param sn_power: The power of iterations for estimating spectral norm.
        """
        output = output.detach().clamp(-0.9, 0.9)
        # 1. Inverse & LT.
        # (1) Inverse & linearize.
        output = output.requires_grad_(True)
        e = self._inv(output, sv_ema_r, linearize=True)
        # (2) Inverted LT.
        x_t = torch.randn_like(e).requires_grad_(True)
        output_t = self._inv_lt(x_t, sv_ema_r)
        # 2. Compute.
        logsv_max = fast_approx_logsn(output, e, x_t, output_t, sn_power=sn_power)
        # Return
        return -logsv_max

    @api_empty_cache
    def compute_svs(self, z, jacob_bsize):
        """ Compute singular values for Jacobian. """
        with torch.no_grad(): output = self.forward(z, linearize=True)
        # Compute transposed Jacob & jtj. (batch, nz, nz).
        jacob = autograd_jacob(torch.randn_like(output), func=self.linearized_transpose, bsize=jacob_bsize)
        jacob = torch.matmul(jacob, jacob.transpose(1, 2))
        # Compute singular values. (batch, nz)
        torch.cuda.empty_cache()
        svs = torch.cat([torch.svd(j, compute_uv=False).S.sqrt()[None] for j in jacob])
        # Return
        return svs


# ----------------------------------------------------------------------------------------------------------------------
# Conditional generator.
# ----------------------------------------------------------------------------------------------------------------------

class GeneratorConditional(Generator):
    """
    Conditional generator.
    """
    def __init__(self, n_classes, nz, init_size, img_size, ncs, hidden_ncs, middle_ns_flows):
        super(GeneratorConditional, self).__init__(nz*2, init_size, img_size, ncs, hidden_ncs, middle_ns_flows)
        # For conditional generation.
        self._nz = nz
        self.register_parameter("_param_cat_emb", nn.Parameter(torch.randn(size=(n_classes, nz, 1, 1))))

    def forward(self, z, linearize=False):
        """ Given z is (latent_code, category_label). """
        z = torch.cat([z[0], self._param_cat_emb[z[1]]], dim=1)
        return super(GeneratorConditional, self).forward(z, linearize=linearize)

    def linearized_transpose(self, x):
        output = super(GeneratorConditional, self).linearized_transpose(x)
        return output[:, :self._nz]

    def inverse(self, x, linearize=False):
        output = super(GeneratorConditional, self).inverse(x, linearize=linearize)
        return output[:, :self._nz]

    def inverse_lt(self, eps):
        eps = torch.cat([eps, torch.zeros_like(eps)], dim=1)
        return super(GeneratorConditional, self).inverse_lt(eps)

    # ------------------------------------------------------------------------------------------------------------------
    # Evaluation.
    # ------------------------------------------------------------------------------------------------------------------

    @api_empty_cache
    def eval_jacob(self, z, jacob_size=16):
        """ Given z is (latent_code, category_label). """
        ################################################################################################################
        # Compute J (batch, nx, nz) & JTJ (batch, nz, nz).
        ################################################################################################################
        jacob = autograd_jacob(z[0], func=lambda _z: self.forward((_z, z[1].unsqueeze(1).expand(z[1].size(0), jacob_size).reshape(-1, ))), bsize=jacob_size)
        jtj = torch.matmul(jacob.transpose(1, 2), jacob)
        ################################################################################################################
        # Statistics
        ################################################################################################################
        # 1. Jacobian orthogonality. (batch, )
        ortho_sign = measure_ortho(jtj)
        # 2. Singular values. (batch, nz).
        svs = np.sqrt(np.concatenate([np.linalg.svd(j, compute_uv=False)[np.newaxis] for j in jtj.cpu().numpy()]))
        # 3. Logdet of Jacobian. (batch, ).
        logdet = np.array([np.linalg.slogdet(_jtj)[1] for _jtj in jtj.cpu().numpy()], dtype=np.float32)*0.5
        # Return
        return {'jacob@ortho': ortho_sign, 'svs': svs, 'logdet': logdet}


class GeneratorV1Conditional(GeneratorConditional):
    """
    Conditional generator v1.
    """
    def compute_max_logsv(self, z, sn_power=3):
        """ Given z is (latent_code, category_label). """
        # 1. Forward & LT.
        # (1) Forward & linearize.
        z = (z[0].requires_grad_(True), z[1])
        output = self.forward(z, linearize=True)
        # (2) LT.
        x_t = torch.randn_like(output).requires_grad_(True)
        output_t = self.linearized_transpose(x_t)
        # 2. Compute.
        logsv_max = fast_approx_logsn(z[0], output, x_t, output_t, sn_power=sn_power)
        # Return
        return logsv_max

    def compute_min_logsv(self, output, sn_power=3):
        """ Compute logarithm of the minimum singular values.
        :param output: The output of the generator.
        :param sn_power: The power of iterations for estimating spectral norm.
        """
        output = output.detach().clamp(-0.9, 0.9)
        # 1. Inverse & LT.
        # (1) Inverse & linearize.
        output = output.requires_grad_(True)
        z = self.inverse(output, linearize=True)
        # (2) Inverted LT.
        x_t = torch.randn_like(z).requires_grad_(True)
        output_t = self.inverse_lt(x_t)
        # 2. Compute.
        logsv_max = fast_approx_logsn(output, z, x_t, output_t, sn_power=sn_power)
        # Return
        return -logsv_max


########################################################################################################################
# Discriminators.
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# Unconditional: Vanilla
# ----------------------------------------------------------------------------------------------------------------------

def init_weights_vanilla(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Discriminator32x32(nn.Module):
    """
    Discriminator for image size 32x32.
    """
    def __init__(self, nc, ndf):
        super(Discriminator32x32, self).__init__()
        # 1. Architecture
        self._module = nn.Sequential(
            # (ndf, 16, 16)
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2, 8, 8)
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4, 4, 4)
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, inplace=True),
            # (n_classes+1, 1, 1)
            nn.Conv2d(ndf*4, 1, kernel_size=4, stride=1, padding=0, bias=False), nn.Sigmoid())
        # 2. Init weights.
        self.apply(init_weights_vanilla)

    def forward(self, x):
        return self._module(x).view(-1, )


class Discriminator64x64(nn.Module):
    """
    Discriminator for image size 64x64.
    """
    def __init__(self, nc, ndf):
        super(Discriminator64x64, self).__init__()
        # 1. Architecture
        self._module = nn.Sequential(
            # (ndf, 32, 32)
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2, 16, 16)
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4, 8, 8)
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, inplace=True),
            # (ndf*8, 4, 4)
            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(ndf*8), nn.LeakyReLU(0.2, inplace=True),
            # (1, 1, 1)
            nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=0, bias=False), nn.Sigmoid())
        # 2. Init weights.
        self.apply(init_weights_vanilla)

    def forward(self, x):
        return self._module(x).view(-1, 1).squeeze(1)


# ----------------------------------------------------------------------------------------------------------------------
# Unconditional: StyleGAN2
# ----------------------------------------------------------------------------------------------------------------------

def init_weights_stylegan2disc(net):
    for m in net.modules():
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')


def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)


class Flatten(nn.Module):
    """
    Flatten module.
    """
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class Blur(nn.Module):
    """
    Blur module.
    """
    def __init__(self):
        super(Blur, self).__init__()
        self.register_buffer('f', torch.Tensor([1, 2, 1]))

    def forward(self, x):
        f = self.f[None, None, :] * self.f[None, :, None]
        return filter2D(x, f, normalized=True)


class DiscriminatorStyleGAN2Block(nn.Module):
    """
    Block used in StyleGAN2 discriminator.
    """
    def __init__(self, input_nc, output_nc, downsample=True, blur=True):
        super(DiscriminatorStyleGAN2Block, self).__init__()
        # Architecture
        self._conv_res = nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=2 if downsample else 1)
        self._blocks = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=1), leaky_relu(),
            nn.Conv2d(output_nc, output_nc, kernel_size=3, padding=1), leaky_relu())
        if downsample:
            blocks_down = [Blur()] if blur else []
            blocks_down.append(nn.Conv2d(output_nc, output_nc, kernel_size=3, padding=1, stride=2))
            self._downsample = nn.Sequential(*blocks_down)
        else:
            self._downsample = None

    def forward(self, x):
        """
        :param x: (batch, C, H, W)
        :return:
            - downsample=True:  (batch, output_nc, H/2, W/2)
            - downsample=False: (batch, output_nc, H, W)
        """
        # 1. Residual
        res = self._conv_res(x)
        # 2. Main
        x = self._blocks(x)
        if self._downsample is not None: x = self._downsample(x)
        # Return
        x = (x + res) * (1 / math.sqrt(2))
        return x


class DiscriminatorStyleGAN2(nn.Module):
    """
    StyleGAN2 Discriminator.
    """
    def __init__(self, img_nc, img_size, capacity=64, max_nc=512, blur=True):
        super(DiscriminatorStyleGAN2, self).__init__()
        # Configs
        n_layers = int(math.log2(img_size) - 1)
        # --------------------------------------------------------------------------------------------------------------
        # 1. Architecture
        # --------------------------------------------------------------------------------------------------------------
        #   ncs: A list of (n_layers+1, ), [2^0, 2^1, ..., 2^n_layers] (*capacity)
        ncs = [min(capacity*(2**i), max_nc) for i in range(n_layers+1)]
        # (1) Discriminator blocks
        #   Init blocks
        self._blocks_disc = nn.ModuleList()
        #   Generate for each layer
        for index, (input_nc, output_nc) in enumerate(zip([img_nc] + ncs[:-1], ncs)):
            # Config
            not_last = (index != n_layers)
            # Set blocks
            self._blocks_disc.append(DiscriminatorStyleGAN2Block(input_nc, output_nc, downsample=not_last, blur=blur))
        # (2) Tail blocks
        self._conv_final = nn.Conv2d(ncs[-1], ncs[-1], kernel_size=3, padding=1)
        self._flatten = Flatten()
        self._fc = nn.Linear(2*2*ncs[-1], 1)
        # --------------------------------------------------------------------------------------------------------------
        # 2. Init weights
        # --------------------------------------------------------------------------------------------------------------
        init_weights_stylegan2disc(self)

    def forward(self, x):
        """
        :param x: (batch, C, H, W)
        :return: (1, )
        """
        # Process through each layer
        for block_disc in self._blocks_disc:
            x = block_disc(x)
        # Final
        x = self._conv_final(x)
        x = self._flatten(x)
        x = self._fc(x)
        # Return
        return x.squeeze()


# ----------------------------------------------------------------------------------------------------------------------
# Conditional: Vanilla
# ----------------------------------------------------------------------------------------------------------------------

class Discriminator32x32Conditional(nn.Module):
    """
    Conditional discriminator for image size 32x32.
    """
    def __init__(self, n_classes, nc, ndf):
        super(Discriminator32x32Conditional, self).__init__()
        # 1. Architecture
        self._module = nn.Sequential(
            # (ndf, 16, 16)
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2, 8, 8)
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4, 4, 4)
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, inplace=True),
            # (n_classes+1, 1, 1)
            nn.Conv2d(ndf*4, n_classes+1, kernel_size=4, stride=1, padding=0, bias=False))
        # 2. Init weights.
        self.apply(init_weights_vanilla)

    def forward(self, x):
        return self._module(x).squeeze(-1).squeeze(-1)
