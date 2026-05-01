from micrograd.day02_implementation.value import Value
import random

class Module:
    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin: int):
        self.ws = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    # w*x + b
    def __call__(self, xs):
        val = sum([x * w for x, w in zip(xs, self.ws)], self.b)
        act = val.tanh()
        return act

    def parameters(self):
        return self.ws + [self.b]


class Layer(Module):
    def __init__(self, nin: int, layer_size: int):
        self.neurons = [Neuron(nin) for _ in range(layer_size)]

    def __call__(self, xs):
        # xs -> self.neurons -> out(layer_size)
        outs = [neuron(xs) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]


class MLP(Module):
    def __init__(self, nin: int, nouts: list[int]):
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i-1], sizes[i]) for i in range(1, len(sizes))]

    def __call__(self, xs):
        out = self.layers[0](xs)
        for layer in self.layers[1:]:
            out = layer(out)
        return out

    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]
