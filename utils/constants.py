from collections import namedtuple

from enum import Enum, auto

# class Branch(str, Enum):
#     _global = 'global'
#     local = 'local'
#     fusion = 'fusion'

# class Backbone(str, Enum):
#     resnet50 = 'resnet50'
#     densenet121 = 'densenet121'
#     resnet101 = 'resnet101'

# class LastPool(str, Enum):
#     max = 'max'
#     avg = 'avg'
#     lse = 'lse'
#     adaptive_avg = 'adaptive_avg'

# Branch = namedtuple('Branch', ['_global', 'local', 'fusion'])(0, 1, 2)

class StrEnum(str, Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return self.value

    def equals(self, value):
       return self.value == value

class C:
    # m : model
    class m(StrEnum):
        global_ = 'global'
        local = auto()
        fusion = auto()
    
    # b : backbone
    class b(StrEnum):
        resnet50 = auto()
        densenet121 = auto()
        resnet101 = auto()
    
    # p : pooling
    class p(StrEnum):
        max = auto()
        avg = auto()
        lse = auto()
        adaptive_avg = auto()
    
    # o : optimizer
    class o(StrEnum):
        Adam = auto()
        SGD = auto()
    
    # l : loss function
    class l(StrEnum):
        BCELoss = auto()
        WBCELoss = auto()

    # ls : learning rate scheduler
    class ls(StrEnum):
        ReduceLROnPlateau = auto()
        StepLR = auto()

    # lf : L function
    class lf(StrEnum):
        L1 = auto()
        L2 = auto()
        Lmax = auto()

# Backbone = StrEnum('Backbone', 'resnet50, densenet121, resnet101', qualname='SomeData.Backbone')

print(C.m.local)