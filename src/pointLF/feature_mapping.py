import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, multires, input_dims=3, include_input=True, log_sampling=True):

        super().__init__()
        self.embed_fns = []
        self.out_dims = 0

        if include_input:
            self.embed_fns.append(lambda x: x)
            self.out_dims += input_dims

        max_freq = multires - 1
        N_freqs = multires

        if log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for periodic_fn in [torch.sin, torch.cos]:
                self.embed_fns.append(
                    lambda x, periodic_fn=periodic_fn, freq=freq: periodic_fn(x * freq)
                )
                self.out_dims += input_dims

    def forward(self, x: torch.Tensor):
        return torch.cat([fn(x) for fn in self.embed_fns], -1)


# Positional Encoding Old (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, input_dims=3):
    if i == -1:
        return nn.Identity(), input_dims

    embed_kwargs = {
        "include_input": True,
        "input_dims": input_dims,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim
