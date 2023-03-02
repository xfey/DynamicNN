from dynn.layer.basicOp import BasicLayer


__all__ = ['ChannelGate']


class ChannelGate(BasicLayer):
    """
        Channel gate: controls the width of layer
    """
    def __init__(self, layer:BasicLayer):
        super().__init__(layer.C_in, layer.C_out, layer.stride)
        self.gate = 1.
        self.layer = layer
        self.mode = 'out'
    
    def gateControlHood(self, mode='out', value=1.):
        self.gate = value
        self.mode = mode
        assert self.mode in ['in', 'out']
        self.layer.partialChannel(mode, value)
    
    def forward(self, x):
        return self.layer.forward(x)
