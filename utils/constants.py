from enum import Enum, auto

class StrEnum(str, Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return self.value

    def equals(self, value):
       return self.value == value

# C : Constants
class C:
    # m : model
    class m(StrEnum):
        global_ = 'global'
        local = auto()
        fusion = auto()
        att = 'attention'
    
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
        WBCELoss = 'WeightedBCELoss'
        BCELogitsLoss = 'BCEWithLogitsLoss'

    # ls : learning rate scheduler
    class ls(StrEnum):
        ReduceLROnPlateau = auto()
        StepLR = auto()

    # lf : L function
    class lf(StrEnum):
        L1 = auto()
        L2 = auto()
        Lmax = auto()

class attrdict(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value
