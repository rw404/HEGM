import torch
from torch import nn
import math


class PLEBucketEmbedderHEGM(nn.Module):
    def __init__(
        self,
        feature_count: int,
        border_count: int,
        bucket_edges_list: torch.Tensor,
        emb_dim: int = 16,
        replace_float_min_value: float = -1000.0,
    ):
        super().__init__()
        self.features_count = feature_count
        self.border_counts = border_count
        self.emb_dim = emb_dim
        self.replace_float_min_value = replace_float_min_value

        self.register_buffer("bucket_edges", bucket_edges_list)
        self.register_buffer("edges_pad", self.bucket_edges.t().contiguous())

        total_bins = (self.border_counts + 1) * self.features_count
        bin_offsets = torch.arange(0, self.features_count, step=1) * (self.border_counts + 1)
        self.register_buffer("bin_offsets", bin_offsets)

        self.deep_emb = nn.Embedding(num_embeddings=total_bins, embedding_dim=self.emb_dim)

    def forward(self, features: torch.Tensor):
        x = torch.clamp(features, min=self.replace_float_min_value)
        B = x.shape[0]

        edges_pad = self.edges_pad
        edges_len = self.border_counts
        bin_offsets = self.bin_offsets

        local_offsets = (x.unsqueeze(-1) >= edges_pad.unsqueeze(0)).sum(dim=-1).to(torch.long)
        local_offsets = torch.clamp(local_offsets, max=edges_len)

        intra_features_mask = (local_offsets > 0) & (local_offsets < self.border_counts)

        left_bin = torch.where(intra_features_mask, local_offsets - 1, local_offsets)
        right_bin = local_offsets

        left_edge_idx = torch.clamp((local_offsets - 1).clamp(min=0), max=edges_len - 1)
        right_edge_idx = torch.clamp(local_offsets.clamp(min=0), max=edges_len - 1)

        edges_b = edges_pad.unsqueeze(0).expand(B, -1, -1)
        left_edge = edges_b.gather(2, left_edge_idx.unsqueeze(-1)).squeeze(-1)
        right_edge = edges_b.gather(2, right_edge_idx.unsqueeze(-1)).squeeze(-1)

        denom = right_edge - left_edge
        denom_safe = torch.where(denom.abs() > 1e-12, denom, torch.ones_like(denom))
        t_raw = (x - left_edge) / denom_safe
        t = torch.where(intra_features_mask, t_raw, torch.zeros_like(t_raw)).clamp(0.0, 1.0)

        left_idx_flat = (bin_offsets.unsqueeze(0) + left_bin).to(torch.long)
        right_idx_flat = (bin_offsets.unsqueeze(0) + right_bin).to(torch.long)

        e_left = self.deep_emb(left_idx_flat)
        e_right = self.deep_emb(right_idx_flat)

        emb = torch.lerp(e_left, e_right, t.unsqueeze(-1))

        return emb


class FeatureEmbedder(nn.Module):
    def __init__(
        self,
        num_cont_features: int,
        bucket_edges: torch.Tensor,
        cat_cardinalities: list[int],
        emb_dim: int = 16,
    ):
        super().__init__()
        self.cont_embedder = PLEBucketEmbedderHEGM(
            feature_count=num_cont_features,
            border_count=32,
            bucket_edges_list=bucket_edges,
            emb_dim=emb_dim,
        )
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(num_embeddings=card, embedding_dim=emb_dim) for card in cat_cardinalities]
        )

    def forward(self, num_features: torch.Tensor, cat_features: torch.Tensor) -> torch.Tensor:
        num_features = torch.nan_to_num(num_features, nan=-1000.0, posinf=-1000.0, neginf=-1000.0)
        cont_embs = self.cont_embedder(num_features)

        cat_embs_list = []
        for i, emb_layer in enumerate(self.cat_embeddings):
            e = emb_layer(cat_features[:, i])
            cat_embs_list.append(e.unsqueeze(1))

        if len(cat_embs_list) > 0:
            cat_embs = torch.cat(cat_embs_list, dim=1)
            return torch.cat([cont_embs, cat_embs], dim=1)
        return cont_embs


class DCNBlock(nn.Module):
    def __init__(self, features_count: int):
        super().__init__()
        self.block = nn.Linear(features_count, features_count)

        with torch.no_grad():
            torch.nn.init.kaiming_normal_(self.block.weight)
            torch.nn.init.zeros_(self.block.bias)

    def forward(self, x_0: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        return x_i + x_0 * self.block(x_i)


class DCNEncoder(nn.Module):
    def __init__(self, n_features: int, num_layers: int = 2, init_dropout_p: float = 0.1):
        super().__init__()
        self.init_drop = nn.Dropout(p=init_dropout_p)
        self.backbone = nn.ModuleList(
            [DCNBlock(features_count=n_features) for _ in range(num_layers)]
        )

    def forward(self, embedded_features: torch.Tensor) -> torch.Tensor:
        x = self.init_drop(embedded_features)
        for layer in self.backbone:
            x = layer(x_0=embedded_features, x_i=x)
        return x


class CommonResNet(nn.Module):
    def __init__(self, flattened_dim, dropout=0.0):
        super().__init__()
        self.subblock = nn.Sequential(
            nn.LayerNorm(flattened_dim),
            nn.Linear(flattened_dim, flattened_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            torch.nn.init.kaiming_normal_(self.subblock[1].weight)
            torch.nn.init.zeros_(self.subblock[1].bias)

    def forward(self, x):
        return x + self.subblock(x)


class ResNet(nn.Module):
    def __init__(self, flattened_dim, dropout=0.0):
        super().__init__()
        self.ln_block = nn.LayerNorm(flattened_dim)
        self.subblock = nn.Sequential(
            nn.Linear(flattened_dim, flattened_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            torch.nn.init.kaiming_normal_(self.subblock[0].weight)
            torch.nn.init.zeros_(self.subblock[0].bias)

    def forward(self, x):
        x_norm = self.ln_block(x)
        return x + self.subblock(x_norm)


class ResNetDecoderHEGM(nn.Module):
    def __init__(self, flattened_dim: int, n_gaussians: int = 3, dropout: float = 0.1):
        super().__init__()
        self.n_gaussians = n_gaussians
        # Tower outputs: [lambda (1), mu (K), sigma (K), weights (K+1)]
        self.head_dims = [1, self.n_gaussians, self.n_gaussians, self.n_gaussians + 1]

        self.common_tower = nn.Sequential(
            nn.LeakyReLU(), CommonResNet(flattened_dim, dropout=dropout)
        )

        self.tower = nn.ModuleList(
            [
                nn.Sequential(
                    ResNet(flattened_dim=flattened_dim, dropout=dropout),
                    nn.LeakyReLU(),
                    nn.LayerNorm(flattened_dim),
                    nn.Linear(flattened_dim, dim),
                )
                for dim in self.head_dims
            ]
        )

        with torch.no_grad():
            target_lambda, beta = 20.0, 0.5
            lambda_bias = (1.0 / beta) * math.log(math.exp(beta * target_lambda) - 1 + 1e-6)
            self.tower[0][-1].bias.fill_(lambda_bias)
            self.tower[0][-1].weight.fill_(1e-4)

            target_means = torch.clamp(torch.linspace(0.1, 0.9, self.n_gaussians), min=1e-3)
            self.tower[1][-1].bias.copy_(torch.log(torch.exp(target_means) - 1 + 1e-6))
            self.tower[1][-1].weight.fill_(1e-4)

            sigma_bias = math.log(math.exp(1.0 / (1.5 * self.n_gaussians)) - 1 + 1e-6)
            self.tower[2][-1].bias.fill_(sigma_bias)
            self.tower[2][-1].weight.fill_(1e-4)

            torch.nn.init.zeros_(self.tower[3][-1].weight)
            torch.nn.init.zeros_(self.tower[3][-1].bias)

    def forward(self, encoded_vector: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(encoded_vector, start_dim=1)
        shared_rep = self.common_tower(x)
        logits_list = [tower(shared_rep) for tower in self.tower]
        return torch.cat(logits_list, dim=1)


class UnifiedRanker(nn.Module):
    def __init__(self, embedder, encoder, decoder, num_users=None, num_items=None, emb_dim=16):
        super().__init__()
        self.embedder = embedder
        self.encoder = encoder
        self.decoder = decoder

        self.use_ids = (num_users is not None) and (num_items is not None)
        if self.use_ids:
            self.user_emb = nn.Embedding(num_users, emb_dim)
            self.item_emb = nn.Embedding(num_items, emb_dim)
            torch.nn.init.xavier_uniform_(self.user_emb.weight.data)
            torch.nn.init.xavier_uniform_(self.item_emb.weight.data)

    def forward(
        self, num_features: torch.Tensor, cat_features: torch.Tensor, user_id=None, item_id=None
    ) -> torch.Tensor:
        embedded_features = self.embedder(num_features, cat_features)

        if self.use_ids and user_id is not None and item_id is not None:
            u_e = self.user_emb(user_id).unsqueeze(1)
            i_e = self.item_emb(item_id).unsqueeze(1)
            embedded_features = torch.cat([embedded_features, u_e, i_e], dim=1)

        embedded_features = torch.flatten(embedded_features, start_dim=1)
        encoded_vector = self.encoder(embedded_features)

        scores = self.decoder(encoded_vector)
        return scores
