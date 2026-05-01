import math


class Value:

    def __init__(self, data: float, prev: tuple = ()):
        self.data = data
        self._backward = lambda: None
        self.prev = set(prev)
        self.grad = 0.0

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __radd__(self, other):
        return self + other

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, val):
        assert isinstance(val, (int, float))
        out = Value(self.data ** val, (self, ))

        def _backward():
            self.grad += val * (self.data ** (val - 1)) * out.grad
        out._backward = _backward

        return out

    def __rsub__(self, other):
        return Value(other) - self

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return (-1) * self

    def exp(self):
        out = Value(math.exp(self.data), (self, ))

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def __rtruediv__(self, other): # some_num / x
        return other * (self ** -1)

    def __truediv__(self, other): # x / some_num
        if not isinstance(other, Value):
            other = Value(other)
        return self * (other ** -1)

    # activation functions
    def tanh(self):
        pos_e = self.exp()
        neg_e = (-self).exp()
        return (pos_e - neg_e) / (pos_e + neg_e)

    def sigmoid(self):
        return 1 / (1 + (-self).exp())

    def relu(self):
        if self.data < 0:
            out = Value(0, (self, ))
            def _backward():
                pass
        else:
            out = Value(self.data, (self, ))
            def _backward():
                self.grad += out.grad

        out._backward = _backward

        return out

    # backprop
    def backward(self):
        self.grad = 1.0
        # topsort
        nodes = []
        visited = set()
        def topsort(node):
            # add kids
            for v in node.prev:
                if v in visited:
                    continue
                visited.add(v)
                topsort(v)
            # add self
            nodes.append(node)

        topsort(self)
        for node in reversed(nodes):
            node._backward()
