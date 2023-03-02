from dynn.layer.basicOp import BasicLayer, Identity, FactorizedReduce, Zero


__all__ = ['PassGate', 'SkipGate']


class PassGate(BasicLayer):
    """
        Pass gate: controls whether a layer is activated.
    """
    def __init__(self, layer:BasicLayer):
        super().__init__(layer.C_in, layer.C_out, layer.stride)
        self.gate = True
        self.layer = layer
        self.zeroOp = Zero(layer.stride)

    def gateControlHook(self, value):
        self.gate = value

    def forward(self, x):
        if self.gate:
            return self.layer.forward(x)
        else:
            return self.zeroOp.forward(x)


class SkipGate(BasicLayer):
    """
        Skip gate: controls whether a layer is replaced by a skip connection.
    """
    def __init__(self, layer:BasicLayer):
        super().__init__(layer.C_in, layer.C_out, layer.stride)
        self.gate = True
        self.layer = layer
        self.skipOp = Identity(
        ) if layer.stride == 1 else FactorizedReduce(layer.C_in, layer.C_out)

    def gateControlHook(self, value):
        self.gate = value

    def forward(self, x):
        if self.gate:
            return self.layer.forward(x)
        else:
            return self.skipOp.forward(x)
