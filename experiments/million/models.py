"""Architectures preserved from nonlinear screen commit68c142af3f3e42f1db597059ceeb7dca17100dc6."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, inputs, outputs, width, depth, residual=False):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(inputs, width)] +
            [nn.Linear(width, width) for _ in range(depth - 2)] + [nn.Linear(width, outputs)])
        self.residual = residual

    def forward(self, x):
        h = F.gelu(self.layers[0](x))
        for layer in self.layers[1:-1]:
            z = F.gelu(layer(h))
            h = h + z / math.sqrt(len(self.layers) - 2) if self.residual else z
        return self.layers[-1](h)


class Transformer(nn.Module):
    def __init__(self, inputs, outputs, width, depth):
        super().__init__()
        self.embedding = nn.Linear(1, width)
        self.position = nn.Parameter(torch.randn(1, inputs + 1, width) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(
            width, 4, width * 2, dropout=0.0, activation="gelu", batch_first=True,
            norm_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, outputs)

    def forward(self, x):
        h = torch.cat([self.cls.expand(x.shape[0], -1, -1),
                       self.embedding(x.unsqueeze(-1))], 1) + self.position
        for block in self.blocks:
            h = block(h)
        return self.output(self.norm(h[:, 0]))


class Relational(nn.Module):
    def __init__(self, inputs, outputs, width, depth):
        super().__init__()
        self.embedding = nn.Linear(1, width)
        self.position = nn.Parameter(torch.randn(1, inputs, width) * 0.02)
        self.relations = nn.ModuleList([nn.Sequential(nn.Linear(2 * width, width),
            nn.GELU(), nn.Linear(width, width), nn.GELU()) for _ in range(depth)])
        self.output = nn.Linear(width, outputs)

    def forward(self, x):
        h = self.embedding(x.unsqueeze(-1)) + self.position
        n = h.shape[1]
        for relation in self.relations:
            pairs = torch.cat([h[:, :, None, :].expand(-1, -1, n, -1),
                               h[:, None, :, :].expand(-1, n, -1, -1)], -1)
            h = h + relation(pairs).mean(2) / math.sqrt(len(self.relations))
        return self.output(h.mean(1))


class Hypernetwork(nn.Module):
    """Only theta=g.parameters() is learned. f has no stored Parameters."""
    def __init__(self, inputs, outputs, width, depth, f_depth=3):
        super().__init__()
        dims = [inputs] + [8] * (f_depth - 1) + [outputs]
        self.shapes = list(zip(dims[:-1], dims[1:]))
        self.generated_count = sum(a * b + b for a, b in self.shapes)
        self.g = MLP(inputs, self.generated_count, width, depth)
        with torch.no_grad():
            self.g.layers[-1].weight.mul_(0.1)
            offset = 0
            for a, b in self.shapes:
                self.g.layers[-1].bias[offset:offset+a*b].normal_(0, 1 / math.sqrt(a))
                offset += a * b
                self.g.layers[-1].bias[offset:offset+b].zero_()
                offset += b

    def forward(self, x):
        weights = self.g(x)
        h = x
        offset = 0
        for i, (a, b) in enumerate(self.shapes):
            w = weights[:, offset:offset+a*b].reshape(-1, a, b)
            offset += a * b
            bias = weights[:, offset:offset+b]
            offset += b
            h = torch.bmm(h.unsqueeze(1), w).squeeze(1) + bias
            if i + 1 < len(self.shapes):
                h = F.gelu(h)
        return h


def model_for(variant, meta, width):
    args = (meta["input_dim"], meta["output_dim"], width, variant["depth"])
    family = variant["family"]
    if family in ("mlp", "resmlp"):
        return MLP(*args, residual=family == "resmlp")
    if family == "transformer":
        return Transformer(*args)
    if family == "relational":
        return Relational(*args)
    if family == "hyper":
        return Hypernetwork(*args, f_depth=variant["f_depth"])
    raise ValueError(family)


def parameter_count(variant, meta, width):
    i, o, w, d = meta["input_dim"], meta["output_dim"], width, variant["depth"]
    family = variant["family"]
    if family == "hyper":
        dims = [i] + [8] * (variant["f_depth"] - 1) + [o]
        o = sum(a * b + b for a, b in zip(dims[:-1], dims[1:]))
    if family in ("mlp", "resmlp", "hyper"):
        return (i + 1) * w + (d - 2) * (w + 1) * w + (w + 1) * o
    if family == "transformer":
        return 2*w + (i+1)*w + w + d*(8*w*w + 11*w) + 2*w + (w+1)*o
    if family == "relational":
        return 2*w + i*w + d*(3*w*w + 2*w) + (w+1)*o
    raise ValueError(family)


def choose_width(variant, meta, target):
    step = 4 if variant["family"] == "transformer" else 1
    return min(range(8, 513, step), key=lambda w: abs(parameter_count(variant, meta, w) - target))
