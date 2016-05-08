import numpy
import chainer
import chainer.functions as F
import chainer.links as L

class AutoEncoder(chainer.Chain):
    def __init__(self, il, ol, actfunc):
        super(AutoEncoder, self).__init__(
            l_in = il,
            l_out = ol,
        )
        self.actfunc = actfunc


    def forwardm(self, x, activate=True):
        if activate :
            return F.dropout(self.actfunc(self.l_in(x)), train = False)
        else :
            return self.l_in(x)

    def forwardo(self, x):
        return self.actfunc(self.l_out(x))
        
    def clear(self):
            self.loss = None

    def __call__(self, x, train=True):
        self.clear()
        h = F.dropout(self.actfunc(self.l_in(x)), train = train)
        y = self.actfunc(self.l_out(h))
        
        self.loss = F.mean_squared_error(y, x)
        return self.loss
    
