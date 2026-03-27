from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool
except ImportError:  # pragma: no cover
    torch = None
    F = None
    nn = None
    GATv2Conv = None
    global_max_pool = None
    global_mean_pool = None

from backend.config import ModelConfig


def _scatter_logsumexp(logits, batch_index):
    if torch is None:
        raise RuntimeError("torch is required")
    num_graphs = int(batch_index.max().item() + 1) if batch_index.numel() else 1
    outputs = []
    for graph_idx in range(num_graphs):
        graph_logits = logits[batch_index == graph_idx]
        outputs.append(torch.logsumexp(graph_logits, dim=0))
    return torch.stack(outputs, dim=0)


if nn is not None and GATv2Conv is not None:
    class ResidualGATBlock(nn.Module):
        def __init__(self, hidden_dim: int, heads: int, edge_dim: int, dropout: float):
            super().__init__()
            out_per_head = hidden_dim // heads
            self.conv = GATv2Conv(
                hidden_dim,
                out_per_head,
                heads=heads,
                edge_dim=edge_dim,
                dropout=dropout,
            )
            self.norm = nn.LayerNorm(hidden_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, edge_index, edge_attr):
            residual = x
            x = self.conv(x, edge_index, edge_attr)
            x = self.norm(x)
            x = F.elu(x)
            x = self.dropout(x)
            return x + residual


    class HexGNNModel(nn.Module):
        def __init__(self, config: ModelConfig):
            super().__init__()
            self.config = config
            self.input_proj = nn.Sequential(
                nn.Linear(config.node_feature_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.ELU(),
            )
            self.blocks = nn.ModuleList(
                [
                    ResidualGATBlock(
                        hidden_dim=config.hidden_dim,
                        heads=config.num_attention_heads,
                        edge_dim=config.edge_dim,
                        dropout=config.dropout,
                    )
                    for _ in range(config.num_gnn_layers)
                ]
            )
            self.policy_head = nn.Sequential(
                nn.Linear(config.hidden_dim, 64),
                nn.ELU(),
                nn.Linear(64, 1),
            )
            self.value_head = nn.Sequential(
                nn.Linear(config.hidden_dim * 2, 128),
                nn.ELU(),
                nn.Linear(128, 64),
                nn.ELU(),
                nn.Linear(64, 1),
                nn.Tanh(),
            )
            self.aux_head = nn.Sequential(
                nn.Linear(config.hidden_dim * 2, 128),
                nn.ELU(),
                nn.Linear(128, 4),
            )

        def forward(self, data):
            x = self.input_proj(data.x)
            for block in self.blocks:
                x = block(x, data.edge_index, data.edge_attr)

            batch = getattr(data, "batch", None)
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

            logits = self.policy_head(x).squeeze(-1)
            logits = logits.masked_fill(~data.legal_mask, -1e9)

            pooled = torch.cat(
                [
                    global_mean_pool(x, batch),
                    global_max_pool(x, batch),
                ],
                dim=-1,
            )
            value = self.value_head(pooled).squeeze(-1)
            aux = self.aux_head(pooled)
            return logits, value, aux

        def policy_log_probs(self, data):
            logits, _, _ = self.forward(data)
            return self.logits_to_log_probs(logits, data)

        def logits_to_log_probs(self, logits, data):
            batch = getattr(data, "batch", None)
            if batch is None:
                batch = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            lse = _scatter_logsumexp(logits, batch)
            return logits - lse[batch]
else:
    class ResidualGATBlock:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch and torch_geometric are required for ResidualGATBlock")


    class HexGNNModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch and torch_geometric are required for HexGNNModel")
