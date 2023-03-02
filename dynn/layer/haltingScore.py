from dynn.layer.basicOp import BasicLayer


__all__ = ['HaltingScore']


class HaltingScore(BasicLayer):
    """
        saving the halting score of each layer.
    """
    def __init__(self, layer:BasicLayer):
        super().__init__(layer.C_in, layer.C_out, layer.stride)
        self.haltingScore = 0.
        self.layer = layer
        self.calFunc = lambda x:x
    
    def haltingScoreHook(self, func):
        self.calFunc = func
    
    def forward(self, x, *args, **kwargs):
        self.haltingScore = self.calFunc(*args, **kwargs)
        return self.layer.forward(x)
