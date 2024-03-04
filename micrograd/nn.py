from Value import Value
import random
class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):

    def __init__(self, nin):
        '''
        nin: the dimension of input data point, which is a vector.

        Here we initialize the weights and thr bias via a uniform distribution. In practice it's up to you to choose the initialization method, sometimes a normal distribution is better, sometimes we set them to be all zero, there are also more complicated methods such as [Xavier](https://cs230.stanford.edu/section/4/).
        '''
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        '''
        w * x + b, where "*" is dot product
        Then apply tanh as the activation function.
        The output is a scalar.
        '''
        act = sum(w_i * x_i for w_i, x_i in zip(self.w, x)) + self.b
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):

    def __init__(self, nin, nout):
        '''
        nin: the dimension of input data point, which is a vector.
        nout: the dimension of the output, which equals to the number of the neurons in the layer since each neuron outputs a scalar as one element of the output vector.
        '''
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(
            outs) == 1 else outs  # If the output is a vector with length=1, then we output its scalar version instead.

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):

    def __init__(self, nin, nouts):
        '''
        nin: the dimension of input data point, which is a vector.
        nouts: the list of the `nout` of each layer.
        '''

        # sz: the size of the MLP.
        # For example, if input vector has dimension=3, we have 3 layers with dimension=4,4,1 separately, then
        # total size of the MLP is [3,4,4,1], i.e., we add a first layer with size (3,4).
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]