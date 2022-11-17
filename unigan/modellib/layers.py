
from modellib.components import *


########################################################################################################################
# Dimensionality increasing.
########################################################################################################################

class DimIncreaseUnsqueeze2(InvertibleModule):
    """
    Use an invertible conv to increase dim first, then unsqueeze feature map.
    """
    def __init__(self, input_nc, output_nc):
        super(DimIncreaseUnsqueeze2, self).__init__()
        # Config.
        assert output_nc*4 >= input_nc
        # --------------------------------------------------------------------------------------------------------------
        # Architecture.
        # --------------------------------------------------------------------------------------------------------------
        self._conv = InvConv2d1x1Fixed(input_nc, output_nc*4)
        self._usqz = Unsqueeze(s=2)

    def forward(self, x):
        output = self._conv(x)
        output = self._usqz(output)
        # Return
        return output

    def linearized_transpose(self, eps):
        output = self._usqz.linearized_transpose(eps)
        output = self._conv.linearized_transpose(output)
        # Return
        return output

    def inverse(self, x):
        output = self._usqz.inverse(x)
        output = self._conv.inverse(output)
        # Return
        return output

    def inverse_lt(self, eps):
        output = self._conv.inverse_lt(eps)
        output = self._usqz.inverse_lt(output)
        # Return
        return output


########################################################################################################################
# Block.
########################################################################################################################

class AdditiveCoupling(InvertibleModule):
    """
    Coupling.
    """
    def __init__(self, input_nc, input_size, hidden_nc1, hidden_nc2, **kwargs):
        """
        :param input_nc:
        :param input_size:
        :param hidden_nc1:
        :param hidden_nc2:
        :param kwargs:
            - ds_conv: For init layer.
        :return:
        """
        super(AdditiveCoupling, self).__init__()
        # Config.
        init_phase = True if 'ds_conv' in kwargs.keys() else False
        # --------------------------------------------------------------------------------------------------------------
        # Architecture of NN.
        # --------------------------------------------------------------------------------------------------------------
        self._nn = nn.ModuleList([])
        # 1. Downsampling. (input_nc//2, input_size, input_size) -> (hidden_nc1, input_size//2, input_size//2)
        self._nn.append(Squeeze(s=input_size if init_phase else 2))
        self._nn.append(kwargs['ds_conv'] if init_phase else InvConv2d1x1Fixed(input_nc*2, hidden_nc1))
        # 2. Hidden. (hidden_nc1, input_size//2, input_size//2) -> (hidden_nc2, input_size, input_size)
        convt_kwargs = {'kernel_size': input_size, 'stride': 1, 'padding': 0} if init_phase else {'kernel_size': 4, 'stride': 2, 'padding': 1}
        self._nn.append(ConvTranspose2d(hidden_nc1, hidden_nc2, **convt_kwargs, bias=True))
        self._nn.append(ReLU())
        # 3. Channel changer. (hidden_nc2, input_size, input_size) -> (input_nc//2, input_size, input_size)
        self._nn.append(InvConv2d1x1Fixed(hidden_nc2, input_nc//2))

    def forward(self, x, linearize=False):
        # 1. Split.
        x_a, x_b = x.chunk(2, dim=1)
        # 2. Coupling.
        # (1) NN.
        output_nn = x_a
        for module in self._nn:
            kwargs = {'linearize': linearize} if isinstance(module, ReLU) else {}
            # Forward.
            output_nn = module(output_nn, **kwargs)
        # (2) Additive.
        output_b = x_b + output_nn
        # 3. Merge.
        output = torch.cat([x_a, output_b], dim=1)
        # Return
        return output

    def linearized_transpose(self, eps):
        # 1. Split.
        x_a, x_b = eps.chunk(2, dim=1)
        # 2. Coupling.
        # (1) NN.
        output_nn = x_b
        for module in self._nn[::-1]:
            output_nn = module.linearized_transpose(output_nn)
        # (2) Additive.
        output_a = x_a + output_nn
        # 3. Merge.
        output = torch.cat([output_a, x_b], dim=1)
        # Return
        return output

    def inverse(self, x, linearize=False):
        # 1. Split.
        x_a, output_b = x.chunk(2, dim=1)
        # 2. Coupling.
        # (1) NN.
        output_nn = x_a
        for module in self._nn:
            kwargs = {'linearize': linearize} if isinstance(module, ReLU) else {}
            # Forward.
            output_nn = module(output_nn, **kwargs)
        # (2) Additive.
        x_b = output_b - output_nn
        # 2. Concat.
        output = torch.cat([x_a, x_b], dim=1)
        # Return
        return output

    def inverse_lt(self, eps):
        # 1. Split.
        x_a, x_b = eps.chunk(2, dim=1)
        # 2. Coupling.
        # (1) NN.
        output_nn = x_b
        for module in self._nn[::-1]:
            output_nn = module.linearized_transpose(output_nn)
        # (2) Additive.
        output_a = x_a - output_nn
        # 3. Merge.
        output = torch.cat([output_a, x_b], dim=1)
        # Return
        return output


class Flow(InvertibleModule):
    """
    Flow.
    """
    def __init__(self, input_nc, input_size, hidden_nc1, hidden_nc2, **kwargs):
        """
        :param input_nc:
        :param input_size:
        :param hidden_nc1:
        :param hidden_nc2:
        :param kwargs:
            - ds_conv: For init layer.
        :return:
        """
        super(Flow, self).__init__()
        # Config.
        self._init_phase = True if 'ds_conv' in kwargs.keys() else False
        # --------------------------------------------------------------------------------------------------------------
        # Architecture.
        # --------------------------------------------------------------------------------------------------------------
        # 1. Conv.
        if not self._init_phase:
            self._conv = InvConv2d1x1Fixed(input_nc, input_nc)
        # 2. ActNorm + Coupling.
        self._actnorm = ActNorm(input_nc)
        self._coupling = AdditiveCoupling(input_nc, input_size, hidden_nc1, hidden_nc2, **kwargs)

    def forward(self, x, linearize=False):
        # Forward.
        if not self._init_phase:
            x = self._conv(x)
        x = self._actnorm(x)
        output = self._coupling(x, linearize=linearize)
        # Return
        return output

    def linearized_transpose(self, eps):
        output = self._coupling.linearized_transpose(eps)
        output = self._actnorm.linearized_transpose(output)
        if not self._init_phase:
            output = self._conv.linearized_transpose(output)
        # Return
        return output

    def inverse(self, x, linearize=False):
        output = self._coupling.inverse(x, linearize=linearize)
        output = self._actnorm.inverse(output)
        if not self._init_phase:
            output = self._conv.inverse(output)
        # Return
        return output

    def inverse_lt(self, eps):
        if not self._init_phase:
            eps = self._conv.inverse_lt(eps)
        eps = self._actnorm.inverse_lt(eps)
        output = self._coupling.inverse_lt(eps)
        # Return
        return output


class Block(InvertibleModule):
    """
    Block.
    """
    def __init__(self, input_nc, input_size, hidden_nc1, hidden_nc2, **kwargs):
        """
        :param input_nc:
        :param input_size:
        :param hidden_nc1:
        :param hidden_nc2:
        :param kwargs
             - ds_conv: For init layer.
             - n_flows: For middle layers.
        """
        super(Block, self).__init__()
        # --------------------------------------------------------------------------------------------------------------
        # Architecture.
        # --------------------------------------------------------------------------------------------------------------
        # Flows.
        self._flows = nn.ModuleList([])
        for _ in range(1 if 'ds_conv' in kwargs.keys() else kwargs.pop('n_flows')):
            self._flows.append(Flow(input_nc, input_size, hidden_nc1, hidden_nc2, **kwargs))

    def forward(self, x, linearize=False):
        for flow in self._flows:
            x = flow(x, linearize=linearize)
        # Return
        return x

    def linearized_transpose(self, eps):
        for flow in self._flows[::-1]:
            eps = flow.linearized_transpose(eps)
        # Return
        return eps

    def inverse(self, x, linearize=False):
        for flow in self._flows[::-1]:
            x = flow.inverse(x, linearize=linearize)
        # Return
        return x

    def inverse_lt(self, eps):
        for flow in self._flows:
            eps = flow.inverse_lt(eps)
        # Return
        return eps


class InitDiuBlock(InvertibleModule):
    """
    Init DIU + Block module.
    """
    def __init__(self, nz, output_nc, output_size, hidden_nc):
        super(InitDiuBlock, self).__init__()
        assert output_nc % 2 == 0 and output_nc > nz*2
        # --------------------------------------------------------------------------------------------------------------
        # Architecture.
        # --------------------------------------------------------------------------------------------------------------
        # 1. Conv & usqz & concat0.
        self._conv = InvConv2d1x1Fixed(nz, (output_nc//2)*(output_size**2))
        self._usqz = Unsqueeze(s=output_size)
        self._cat = ConcatZero()
        # 2. Block.
        self._block = Block(
            output_nc, output_size, hidden_nc1=nz, hidden_nc2=hidden_nc,
            ds_conv=InvConv2d1x1Fixed((output_nc//2)*(output_size**2), nz, matrix_r=self._conv._matrix_r.T))

    def forward(self, x, linearize=False):
        # 1. DIU.
        output = self._conv(x)
        output = self._usqz(output)
        output = self._cat(output)
        # 2. Block.
        ret = self._block(output, linearize=linearize)
        # Return
        return ret

    def linearized_transpose(self, eps):
        # 1. Block.
        output = self._block.linearized_transpose(eps)
        # 2. DIU.
        output = self._cat.linearized_transpose(output)
        output = self._usqz.linearized_transpose(output)
        output = self._conv.linearized_transpose(output)
        # Return
        return output

    def inverse(self, x, linearize=False):
        # 1. Block.
        output = self._block.inverse(x, linearize=linearize)
        # 2. DIU.
        output = self._cat.inverse(output)
        output = self._usqz.inverse(output)
        output = self._conv.inverse(output)
        # Return
        return output

    def inverse_lt(self, eps):
        # 1. DIU.
        output = self._conv.inverse_lt(eps)
        output = self._usqz.inverse_lt(output)
        output = self._cat.inverse_lt(output)
        # 2. Block.
        output = self._block.inverse_lt(output)
        # Return
        return output
