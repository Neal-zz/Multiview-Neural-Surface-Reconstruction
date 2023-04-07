import torch

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        # 3
        d = self.kwargs['input_dims']
        out_dim = 0
        # True
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        # 5
        max_freq = self.kwargs['max_freq_log2']
        # 6
        N_freqs = self.kwargs['num_freqs']
        # True
        if self.kwargs['log_sampling']:
            # [1.,2.,4.,8.,16.,32.]
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d
        # [x,sin(x),cos(x),...,sin(32x),cos(32x)] 13 个函数
        self.embed_fns = embed_fns
        # 39
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim
