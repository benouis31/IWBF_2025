

from torch import nn
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, Optional
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import math
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange





class CustomLoss:
    def __init__(
            self,
            temperature=0.5,
            tau=0.01,
            beta=2,
            sim_coeff=25,
            std_coeff=25,
            cov_coeff=1,
            std_const=1e-4,
            lambd=3.9e-3,
            scale_loss=1 / 32,
            reduction='none'
    ):
        self.temperature = temperature
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.beta = beta
        self.tau = tau
        self.lambd = lambd
        self.scale_loss = scale_loss
        self.std_const = std_const
        self.reduction = reduction
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum')
        self._similarity_fn = DotProduct()

    def cov_loss_each(self, z, batch_size):
        c = torch.matmul(z, z.T)
        c = c / (batch_size - 1)

        num_features = c.shape[0]
        off_diag_c = self.off_diagonal(c)
        off_diag_c = torch.pow(off_diag_c, 2)
        off_diag_c = torch.sum(off_diag_c) / num_features

        return off_diag_c

    def off_diagonal(self, x):
        n = x.shape[0]
        flattened = x.flatten()[:-1]
        off_diagonals = flattened.view(n - 1, n + 1)[:, 1:]
        off_diag = off_diagonals.flatten()
        return off_diag

    def mean_center_columns(self, x):
        col_mean = torch.mean(x, dim=0)
        norm_col = x - col_mean
        return norm_col

    def get_loss_fn(self, loss_type):
        if loss_type == "nce":
            def loss(r1, r2):
                dot_prod = self._similarity_fn(r1, r2)
                all_sim = torch.exp(dot_prod / self.temperature)
                logits = all_sim / torch.sum(all_sim, dim=1, keepdim=True)
                lbl = torch.ones_like(logits)
                error = self.criterion(logits, lbl)
                return error

        elif loss_type in ["dcl", "harddcl"]:
            def loss(r1, r2):
                sim_mat = self._similarity_fn(r1, r2)
                N = sim_mat.shape[0]
                all_sim = torch.exp(sim_mat / self.temperature)
                pos_sim = torch.diag(all_sim)

                tri_mask = torch.ones(N, N, dtype=torch.bool)
                #tri_mask[torch.diag_indices(N)] = False
                tri_mask = torch.ones(N, N, dtype=torch.bool)
                for i in range(N):
                    tri_mask[i, i] = False
                neg_sim = all_sim[tri_mask].view(N, N - 1)

                reweight = 1.0
                if loss_type == "harddcl":
                    reweight = (self.beta * neg_sim) / torch.mean(neg_sim, dim=1, keepdim=True)

                Ng = self.tau * (1 - N) * pos_sim + torch.sum(reweight * neg_sim, dim=-1)
                Ng = torch.clip(Ng, min=(N - 1) * np.exp(-1 / self.temperature), max=float('inf'))
                #Ng = torch.clip(Ng, min=(N - 1) * np.exp(-1 / self.temperature), max=torch.float32.max)
                error = torch.mean(-torch.log(pos_sim / (pos_sim + Ng)))
                return error

        elif loss_type == "cocoa":
            def loss(ytrue, ypred):
                batch_size, dim_size = ypred.shape[1], ypred.shape[0]
                pos_error = []
                for i in range(batch_size):
                    sim = torch.matmul(ypred[:, i, :], ypred[:, i, :].T)
                    sim = 1 - sim
                    sim = torch.exp(sim / self.temperature)
                    pos_error.append(torch.mean(sim))

                neg_error = 0
                for i in range(dim_size):
                    sim = torch.matmul(ypred[i], ypred[i].T)
                    sim = torch.exp(sim / self.temperature)
                    tri_mask = torch.ones(batch_size, batch_size, dtype=torch.bool)
                    tri_mask[torch.diag_indices(batch_size)] = False
                    off_diag_sim = sim[tri_mask].view(batch_size, batch_size - 1)
                    neg_error += torch.mean(off_diag_sim, dim=-1)

                error = torch.sum(torch.stack(pos_error)) * self.scale_loss + self.lambd * torch.sum(neg_error)
                return error

        elif loss_type == "vicreg":
            def loss(za, zb):
                sim_loss = F.mse_loss(za, zb, reduction='none')

                za = self.mean_center_columns(za)
                zb = self.mean_center_columns(zb)

                std_za = torch.sqrt(torch.var(za, dim=0) + self.std_const)
                std_zb = torch.sqrt(torch.var(zb, dim=0) + self.std_const)

                std_loss_za = torch.mean(torch.maximum(torch.tensor(0.0), 1 - std_za))
                std_loss_zb = torch.mean(torch.maximum(torch.tensor(0.0), 1 - std_zb))

                std_loss = (std_loss_za + std_loss_zb) / 2

                off_diag_ca = self.cov_loss_each(za, za.shape[0])
                off_diag_cb = self.cov_loss_each(zb, zb.shape[0])

                cov_loss = off_diag_ca + off_diag_cb

                error_value = (
                        self.sim_coeff * torch.mean(sim_loss) +
                        self.std_coeff * std_loss +
                        self.cov_coeff * cov_loss
                )

                return error_value

        elif loss_type == "mse":
            def loss(ytrue, ypred):
                return F.mse_loss(ytrue, ypred)

        else:
            raise ValueError("Undefined loss function.")

        return loss
    




class Similarity_Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, z_list, z_avg):
        z_sim = 0
        num_masked = len(z_list)
        
        for z in z_list:
            z_sim -= F.cosine_similarity(z, z_avg, dim=1).mean()
            
        z_sim = z_sim/num_masked
        
        return z_sim

class Similarity_Losss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, z_list, z_avg):
        #print(z_list)
        z_sim = F.mse_loss(z_list, z_avg)
        return z_sim

class TotalCodingRate(nn.Module):
    def __init__(self, eps=0.01):
        super(TotalCodingRate, self).__init__()
        self.eps = eps

    def compute_discrimn_loss(self, W):
        """Discriminative Loss."""
        p, m = W.shape #[d, B]
        I = torch.eye(p,device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.

    def forward(self, z_list):
        loss = 0
        for X in z_list:
            loss -= self.compute_discrimn_loss(X.T)
        loss = loss/len(z_list)
        return loss

def posemb_sincos_1d(patches, temperature = 10000, dtype = torch.float32):
    _, n, dim, device, dtype = *patches.shape, patches.device, patches.dtype
    n = torch.arange(n, device = device)
    assert (dim % 2) == 0, 'feature dimension must be multiple of 2 for sincos emb'
    omega = torch.arange(dim // 2, device = device) / (dim // 2 - 1)
    omega = 1. / (temperature ** omega)
    n = n.flatten()[:, None] * omega[None, :]
    pe = torch.cat((n.sin(), n.cos()), dim = 1)
    return pe.type(dtype)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class TemporalConvNet(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_channels,
                 kernel_size=2,
                 dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers += [
                nn.Conv1d(in_channels,
                          out_channels,
                          kernel_size,
                          padding=padding,
                          dilation=dilation_size),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout)
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TMS_BIO(nn.Module):
    def __init__(self, configs):
        super(TMS_BIO, self).__init__()

        # TCN parameters (assuming these are in configs)
        tcn_nfilters = configs.tcn_nfilters
        tcn_kernel_size = configs.tcn_kernel_size
        tcn_dropout = configs.tcn_dropout
        trans_d_model = configs.trans_d_model #512
        embed_dim = configs.embed_dim #64

        # Separate TCNs for each modality
        self.tcn1 = TemporalConvNet(num_inputs=1, num_channels=tcn_nfilters,
                                    kernel_size=tcn_kernel_size, dropout=tcn_dropout)
        self.tcn2 = TemporalConvNet(num_inputs=1, num_channels=tcn_nfilters,
                                    kernel_size=tcn_kernel_size, dropout=tcn_dropout)
        self.tcn3 = TemporalConvNet(num_inputs=1, num_channels=tcn_nfilters,
                                    kernel_size=tcn_kernel_size, dropout=tcn_dropout)

        # Linear layers for projection
        self.project_1 = nn.Linear(tcn_nfilters[-1], trans_d_model)
        self.project_2 = nn.Linear(tcn_nfilters[-1], trans_d_model)
        self.project_3 = nn.Linear(tcn_nfilters[-1], trans_d_model)

        # Layer norms
        self.layernorm_1 = nn.LayerNorm(trans_d_model)
        self.layernorm_2 = nn.LayerNorm(trans_d_model)
        self.layernorm_3 = nn.LayerNorm(trans_d_model)

        # Projection layer after concatenation
        self.final_project = nn.Linear(trans_d_model * 3, embed_dim)

        self.transformer_encoder = Transformer(embed_dim, depth = 6, heads = 8, dim_head=embed_dim//8, mlp_dim=embed_dim*4)
        self.linear = nn.Linear(embed_dim, configs.num_classes)
        self.inv_loss = Similarity_Loss()
        self.tcr_loss = TotalCodingRate(eps=0.2)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder = Transformer(embed_dim, depth = 4, heads = 8, dim_head=embed_dim//8, mlp_dim=embed_dim*4)

        # Embedding reducer
        self.embedding_reducer = nn.Linear(embed_dim, 64)

    def forward(self, x1, x2, x3):
        # Pass each modality through its TCN
        x1 = self.tcn1(x1.unsqueeze(1)).transpose(1, 2)
        x2 = self.tcn2(x2.unsqueeze(1)).transpose(1, 2)
        x3 = self.tcn3(x3.unsqueeze(1)).transpose(1, 2)

        # Project and apply layer norm
        x1 = self.layernorm_1(self.project_1(x1))
        x2 = self.layernorm_2(self.project_2(x2))
        x3 = self.layernorm_3(self.project_3(x3))

        # Concatenate the outputs
        x = torch.cat((x1, x2, x3), dim=2)

        # Project after concatenation
        x = self.final_project(x)

        b, n, _ = x.shape
        pe = posemb_sincos_1d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe
        x = self.transformer_encoder(x).mean(dim=1)

        # Reduce batch size
        if x.shape[0] > 64:
            x = x[:64]

        # Reduce embedding dimension
        x = self.embedding_reducer(x)
        rep = x.detach()

        return self.linear(x), rep

    def supervised_train_forward(self, x1, x2, x3, y, criterion=nn.CrossEntropyLoss()):
        pred, _ = self.forward(x1, x2, x3)
        loss = criterion(pred, y)
        return loss, pred.detach()

    def ssl_train_forward(self, x1, x2, x3, mask_ratio=0.75, num_masked=20):
        # Pass each modality through its TCN
        x1 = self.tcn1(x1.unsqueeze(1)).transpose(1, 2)
        x2 = self.tcn2(x2.unsqueeze(1)).transpose(1, 2)
        x3 = self.tcn3(x3.unsqueeze(1)).transpose(1, 2)

        # Project and apply layer norm
        x1 = self.layernorm_1(self.project_1(x1))
        x2 = self.layernorm_2(self.project_2(x2))
        x3 = self.layernorm_3(self.project_3(x3))

        # Concatenate the outputs
        x = torch.cat((x1, x2, x3), dim=2)

        # Project after concatenation
        x = self.final_project(x)

        b, n, _ = x.shape
        pe = posemb_sincos_1d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe
        z_avg = self.transformer_encoder(x).mean(dim=1)
        z_avg = F.normalize(z_avg, p=2)
        z_list = []
        
        for _ in range(num_masked):
            z, mask, ids_restore = self.random_masking(x, mask_ratio)
            z = self.transformer_encoder(z)
            mask_tokens = self.mask_token.repeat(z.shape[0],
                                                ids_restore.shape[1] - z.shape[1],
                                                1)
            z = torch.cat([z, mask_tokens], dim=1)
            z = torch.gather(z,
                            dim=1,
                            index=ids_restore.unsqueeze(-1).repeat(1, 1,
                                                                    z.shape[2])) # unshuffle
            pe = posemb_sincos_1d(z)
            z = rearrange(z, 'b ... d -> b (...) d') + pe
            z = self.decoder(z).mean(dim=1)
            z = F.normalize(z, p=2)
            z_list.append(z)
           #z_list = z
        contrastive_loss = 100 * self.inv_loss(z_list, z_avg)
      #loss = contrastive_loss
        diversity_loss = self.tcr_loss(z_list)
        loss = contrastive_loss + diversity_loss

        return loss, contrastive_loss.item()

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device) # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise,
                                    dim=1) # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x,
                                dim=1,
                                index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

if __name__ == "__main__":
    class Configs:
        input_channels = 3
        kernel_size = 5
        stride = 1
        num_classes = 16
        embed_dim = 64
        tcn_nfilters = [48, 96, 192, 384, 64]
        tcn_kernel_size = 2
        tcn_dropout = 0.2
        trans_d_model = 512
    configs = Configs()
    # Example usage:
    batch_size = 64
    seq_len = 240
    x1 = torch.randn(batch_size, seq_len)  # Modality 1
    x2 = torch.randn(batch_size, seq_len)  # Modality 2
    x3 = torch.randn(batch_size, seq_len)  # Modality 3

    model = TMS_BIO(configs)
    mask_ratio = 0.75
    num_masked = 20

    # Forward pass
    y, rep = model(x1, x2, x3)
    print("Output shape:", y.shape)
    print("Representation shape:", rep.shape)

    # SSL training forward pass
    loss, contrastive_loss = model.ssl_train_forward(x1, x2, x3, mask_ratio=mask_ratio, num_masked=num_masked)
    print("SSL Loss:", loss)
    print("Contrastive Loss:", contrastive_loss)

