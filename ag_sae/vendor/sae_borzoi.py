"""LN and SparseAutoencoder copied unmodified from calico/sae-borzoi.

Source: core/sae.py, commit 2839eeb05be2e827fa9b0aeb83bdcd4735226522
sha256 of the source file: a5e09f1c834c88004913ccf7ae798dfd73e1559667fb1505509a509fff7f8d00
Licence: MIT, see LICENSE.sae-borzoi in this directory.

Only the imports were reduced. The class bodies are byte-for-byte upstream, so
the recipe under test is the published one and not a reimplementation.
"""
import torch
from torch import nn
from typing import Any

def LN(x: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, mu, std

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, k, sparsity_method="topk", normalize=True):
        """
        Enhanced sparse autoencoder with multiple sparsity options.

        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden layer
            k (int): Number of top activations to keep or sparsity target
            sparsity_method (str): 'topk' or 'threshold' or 'smooth_topk'
        """
        super().__init__()
        self.k = k
        self.sparsity_method = sparsity_method
        self.normalize = normalize

        self.activation = nn.ReLU()

        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        self.latent_bias = nn.Parameter(torch.zeros(hidden_dim))

        # Temperature parameter for smooth top-k
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def load_pretrained(self, pretrained_path):
        """Load pretrained weights for encoder and decoder"""
        pretrained = torch.load(pretrained_path)

        self.encoder.weight = nn.Parameter(pretrained["model_state_dict"]["encoder.weight"])
        self.decoder.weight = nn.Parameter(pretrained["model_state_dict"]["decoder.weight"])
        self.pre_bias = nn.Parameter(pretrained["model_state_dict"]["pre_bias"])
        self.latent_bias = nn.Parameter(pretrained["model_state_dict"]["latent_bias"])
        

    def preprocess(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.normalize:
            return x, dict()
        x, mu, std = LN(x)
        return x, dict(mu=mu, std=std)
    
    def encode(self, x):
        x, params = self.preprocess(x)
        return self.encoder(x - self.pre_bias) + self.latent_bias, params
    
    def decode(self, h, params):
        recon = self.decoder(h) + self.pre_bias
        if self.normalize:
            assert params is not None
            recon = recon * params["std"] + params["mu"]
        return recon
    
    def get_sparse_activations(self, h):
        """Apply sparsity using the selected method"""
        if self.sparsity_method == "topk":
            # Hard top-k selection
            # topk_values, _ = torch.topk(h.abs(), k=self.k, dim=1)
            topk_values, _ = torch.topk(h, k=self.k, dim=1) # no abs
            threshold = topk_values[:, -1].unsqueeze(1)
            # return h * (h.abs() >= threshold)
            return h * (h >= threshold)

        if self.sparsity_method == "topk_o":
            topk = torch.topk(h, k=self.k, dim=-1)
            values = topk.values
            result = torch.zeros_like(h)
            result.scatter_(-1, topk.indices, values)
            return result
    
        elif self.sparsity_method == "threshold":
            # Adaptive threshold based on activation statistics
            threshold = h.abs().mean(dim=1, keepdim=True) + h.abs().std(
                dim=1, keepdim=True
            )
            return h * (h.abs() >= threshold)

        elif self.sparsity_method == "smooth_topk":
            # Smooth top-k using softmax
            scores = h.abs() / self.temperature
            mask = torch.softmax(scores, dim=1)
            mask = mask >= torch.topk(mask, k=self.k, dim=1)[0][:, -1:]
            return h * mask

    def get_metrics(self, h_sparse):
        """Calculate additional sparsity metrics"""
        batch_size = h_sparse.size(0)

        # Sparsity ratio (% of zero activations)
        sparsity_ratio = (h_sparse == 0).float().mean(dim=1)

        # Activation statistics
        mean_activation = h_sparse.abs().mean().item()
        std_activation = h_sparse.abs().std().item()

        return {
            "sparsity_ratio": sparsity_ratio.mean().item(),
            "mean_activation": mean_activation,
            "std_activation": std_activation,
        }

    def forward(self, x):

        input_shape = x.shape

        x = x.view(-1, x.size(-1))

        h, params = self.encode(x) # (batch_size * length, hidden_dim)
        h = self.activation(h)

        h_sparse = self.get_sparse_activations(h) # (batch_size * length, hidden_dim)

        x_recon = self.decode(h_sparse, params) # (batch_size * length, channels)

        metrics = self.get_metrics(h_sparse)

        x_recon = x_recon.view(*input_shape)

        return x_recon, h_sparse, metrics
    
    def infer(self, x):
        """
        Run inference on a sparse autoencoder model.

        Args:
            model: SparseAutoencoder model
            x: Input tensor
        """
        input_shape = x.shape

        x = x.view(-1, x.size(-1))

        h, params = self.encode(x) # (batch_size * length, hidden_dim)

        h_sparse = self.get_sparse_activations(h) # (batch_size * length, hidden_dim)

        x_recon = self.decode(h_sparse, params) # (batch_size * length, channels)

        x_recon = x_recon.view(*input_shape)

        return h_sparse, x_recon
