"""
Neural network model definitions for binary/multiclass sequence classification.

Available architectures:
  - LSTMClassifier              — Bidirectional LSTM with optional attention & layer norm
  - LSTMAttentionSEClassifier   — LSTM + Squeeze-Excite + scaled-dot attention pooling
  - TransformerClassifier       — Vanilla transformer encoder with CLS token
  - TransformerSEClassifier     — Transformer + Squeeze-Excite + attention pooling
  - HybridLSTMTransformer       — LSTM encoder feeding into a transformer encoder
  - TCNAttentionSEClassifier    — Dilated causal TCN + Squeeze-Excite + multi-head attention pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Shared utilities ──────────────────────────────────────────────────────────

class AttentionPool(nn.Module):
    """Additive (Bahdanau-style) attention for collapsing a sequence to a single vector."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, S, H)
        weights = torch.softmax(self.proj(h).squeeze(-1), dim=1)  # (B, S)
        return (h * weights.unsqueeze(-1)).sum(dim=1)              # (B, H)


def _bias_init(module: nn.Linear, bias_init):
    """Apply log-prior bias initialisation and small-weight init to a Linear output layer."""
    with torch.no_grad():
        module.bias.copy_(torch.tensor(bias_init, dtype=torch.float32))
        nn.init.normal_(module.weight, mean=0.0, std=0.01)


# ── Models ────────────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """
    Bidirectional LSTM with optional layer norm and additive attention pooling.

    Supports an optional bias_init for the output layer (e.g. log class priors)
    to improve early training stability on imbalanced datasets.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        bidirectional: bool = True,
        dropout: float = 0.2,
        dropout_out: float = 0.3,
        use_attention: bool = True,
        use_layer_norm: bool = True,
        use_residual: bool = False,   # reserved for future use
        bias_init=None,
    ):
        super().__init__()

        self.bidirectional  = bidirectional
        self.use_attention  = use_attention
        self.use_layer_norm = use_layer_norm

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        if use_layer_norm:
            self.ln = nn.LayerNorm(lstm_out_dim)

        if use_attention:
            self.attention = AttentionPool(lstm_out_dim)

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_out),
            nn.Linear(lstm_out_dim // 2, num_classes),
        )

        if bias_init is not None:
            _bias_init(self.fc[3], bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, F)
        h, _ = self.lstm(x)  # (B, S, lstm_out_dim)

        if self.use_layer_norm:
            h = self.ln(h)

        pooled = self.attention(h) if self.use_attention else h[:, -1, :]
        return self.fc(pooled)


class TransformerClassifier(nn.Module):
    """
    Transformer encoder with a learnable CLS token for sequence classification.

    Supports "cls" pooling (CLS token output) or "mean" pooling over the sequence.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        num_classes: int = 3,
        dropout: float = 0.2,
        pooling: str = "cls",
    ):
        super().__init__()

        self.pooling = pooling

        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed  = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))  # +1 for CLS

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, F)
        B = x.size(0)
        x = self.input_proj(x)

        # Prepend CLS token and add positional embeddings
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)           # (B, S+1, d_model)
        x   = x + self.pos_embed[:, :x.size(1), :]

        x = self.transformer(x)

        rep = x[:, 0, :] if self.pooling == "cls" else x[:, 1:, :].mean(dim=1)
        return self.fc(rep)

class HybridLSTMTransformer(nn.Module):
    """
    LSTM encoder whose output is passed into a Transformer encoder for classification.

    The LSTM captures local sequential patterns; the Transformer refines global
    context before pooling via a CLS token.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        d_model: int = 128,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        num_classes: int = 3,
        bidirectional: bool = True,
        pooling: str = "cls",
    ):
        super().__init__()

        self.pooling = pooling

        self.lstm = nn.LSTM(
            input_dim,
            lstm_hidden,
            lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.project = nn.Linear(lstm_out_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, F)
        B = x.size(0)

        lstm_out, _ = self.lstm(x)                     # (B, S, lstm_out_dim)
        tokens = self.project(lstm_out)                # (B, S, d_model)

        # Prepend CLS token and add positional embeddings
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)       # (B, S+1, d_model)
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]

        tokens = self.transformer(tokens)

        rep = tokens[:, 0, :] if self.pooling == "cls" else tokens.mean(dim=1)
        return self.fc(rep)


class LSTMAttentionSEClassifier(nn.Module):
    """
    Bidirectional LSTM + LayerNorm + Squeeze-Excite gating + multi-head attention pooling.

    SE gates suppress noisy timesteps using a global mean-pool context signal.
    Attention pooling uses either a learned query token (recommended) or the final
    LSTM hidden state as the query.

    Supports both binary and multiclass trade-signal models:

    Binary (HOLD vs TRADE):
      - num_classes=2
      - bias_init: [log(1-p), log(p)]  where p = positive-class rate

    Multiclass (SELL / FLAT / BUY, classes 0 / 1 / 2):
      - num_classes=3
      - bias_init: [log(p_sell), log(p_flat), log(p_buy)]
      - use TradeProfitabilityLoss with trade_classes=(0, 2)

    In both cases:
      - use_learned_query=True  improves precision vs. using the final hidden state
      - bias_init starts the output layer at the log class-prior distribution,
        which stabilises early training on imbalanced labels
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,          # default 2 for binary (HOLD vs TRADE)
        bidirectional: bool = True,
        dropout: float = 0.2,
        dropout_out: float = 0.3,
        attn_heads: int = 4,
        attn_dropout: float = 0.1,
        use_learned_query: bool = True,
        se_context_window: int = 32,
        bias_init=None,
    ):
        super().__init__()

        self.bidirectional     = bidirectional
        self.use_learned_query = use_learned_query
        self.se_context_window = se_context_window

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim  = hidden_dim * (2 if bidirectional else 1)
        se_bottleneck = max(8, lstm_out_dim // 8)

        self.ln        = nn.LayerNorm(lstm_out_dim)
        self.se_reduce = nn.Linear(lstm_out_dim, se_bottleneck)
        self.se_expand = nn.Linear(se_bottleneck, lstm_out_dim)

        if use_learned_query:
            self.query_token = nn.Parameter(torch.zeros(1, 1, lstm_out_dim))
            nn.init.normal_(self.query_token, std=0.02)

        self.attn = nn.MultiheadAttention(
            embed_dim=lstm_out_dim,
            num_heads=attn_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_out),
            nn.Linear(lstm_out_dim // 2, num_classes),
        )

        if bias_init is not None:
            _bias_init(self.fc[3], bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, F)
        B = x.size(0)

        h, (hn, _) = self.lstm(x)   # h: (B, S, lstm_out_dim)
        h = self.ln(h)

        # Squeeze-Excite: gate timesteps using the most-recent bars as context.
        # A full-sequence mean dilutes recent signal with historical noise; for
        # short-horizon entries the last se_context_window bars are more predictive.
        ctx_start = max(0, h.size(1) - self.se_context_window)
        context = h[:, ctx_start:, :].mean(dim=1)                            # (B, lstm_out_dim)
        se = torch.sigmoid(self.se_expand(torch.relu(self.se_reduce(context))))
        h  = h * se.unsqueeze(1)

        # Multi-head attention pooling
        if self.use_learned_query:
            q = self.query_token.expand(B, -1, -1)                           # (B, 1, lstm_out_dim)
        else:
            num_dirs = 2 if self.bidirectional else 1
            q = hn[-num_dirs:].transpose(0, 1).reshape(B, -1).unsqueeze(1)   # (B, 1, lstm_out_dim)

        pooled, _ = self.attn(q, h, h, need_weights=False)                   # (B, 1, lstm_out_dim)
        pooled    = pooled.squeeze(1)                                         # (B, lstm_out_dim)

        return self.fc(pooled)


class TransformerSEClassifier(nn.Module):
    """
    Transformer encoder + Squeeze-Excite gating + learned-query attention pooling.

    Uses Pre-LN transformer layers for training stability. SE gating and attention
    pooling follow the same pattern as LSTMAttentionSEClassifier.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        num_classes: int = 3,
        dropout: float = 0.2,
        dropout_out: float = 0.3,
    ):
        super().__init__()

        self.d_model  = d_model
        se_bottleneck = max(8, d_model // 8)

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed  = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for better gradient flow
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)

        self.se_reduce = nn.Linear(d_model, se_bottleneck)
        self.se_expand = nn.Linear(se_bottleneck, d_model)

        # Learned query token replaces CLS; keeps architecture clean (no append/slice needed)
        self.query_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.query_token, std=0.02)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_out),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, F)
        B, S, _ = x.shape

        x = self.input_proj(x) + self.pos_embed[:, :S, :]
        h = self.ln(self.transformer(x))           # (B, S, d_model)

        # Squeeze-Excite: gate timesteps using global mean context
        context = h.mean(dim=1)
        se = torch.sigmoid(self.se_expand(torch.relu(self.se_reduce(context))))
        h  = h * se.unsqueeze(1)

        # Attention pooling via learned query
        q = self.q_proj(self.query_token.expand(B, -1, -1))  # (B, 1, d_model)
        k = self.k_proj(h)
        v = self.v_proj(h)

        scale        = self.d_model ** 0.5
        attn_weights = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / scale, dim=-1)
        pooled       = torch.matmul(attn_weights, v).squeeze(1)  # (B, d_model)

        return self.fc(pooled)


class TemporalBlock(nn.Module):
    """
    Residual block for TCN using dilated causal convolutions.

    Causal padding is applied and then cropped so the output sequence length
    matches the input — no future information leaks across blocks.
    Weight normalisation is applied to both conv layers for training stability.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.padding = (kernel_size - 1) * dilation  # causal left-padding amount

        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, dilation=dilation, padding=self.padding)
        )
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, dilation=dilation, padding=self.padding)
        )

        self.relu1    = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.relu2    = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.relu_out = nn.ReLU()

        # 1×1 projection if channel dimensions differ
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

    def _causal_crop(self, x: torch.Tensor) -> torch.Tensor:
        """Remove the right-hand padding introduced by causal convolution."""
        return x[:, :, :-self.padding] if self.padding > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        out = self.dropout1(self.relu1(self._causal_crop(self.conv1(x))))
        out = self.dropout2(self.relu2(self._causal_crop(self.conv2(out))))

        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)

class TCNAttentionSEClassifier(nn.Module):
    """
    Temporal Convolutional Network (TCN) + Squeeze-Excite + multi-head attention pooling.

    The TCN stack uses exponentially increasing dilations (2^i) to cover a large
    receptive field with few parameters. SE gating and MHA pooling are applied after
    the TCN to focus on the most informative timesteps.

    Supports both binary and 3-class (multiclass) output heads via `num_classes`:

    **Binary** (``num_classes=2``)::

        bias_init = [math.log(p_hold), math.log(p_trade)]

    **Multiclass** (``num_classes=3``, SELL=0 / FLAT=1 / BUY=2)::

        bias_init = [math.log(p_sell), math.log(p_flat), math.log(p_buy)]

    Receptive-field rule: ``(kernel_size - 1) * (2^num_layers - 1) * 2 + 1 ≈ SEQ_LEN``
    e.g. kernel_size=3, num_layers=6 → RF ≈ 253 bars (matches SEQ_LEN=256).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_channels: int = 128,
        num_layers: int = 4,
        kernel_size: int = 3,
        num_classes: int = 2,
        dropout: float = 0.2,
        dropout_out: float = 0.3,
        attn_heads: int = 4,
        attn_dropout: float = 0.1,
        use_learned_query: bool = True,
        se_context_window: int = 32,
        bias_init=None,
    ):
        super().__init__()

        self.use_learned_query = use_learned_query
        self.se_context_window = se_context_window
        se_bottleneck = max(8, hidden_channels // 8)

        # Build TCN stack with exponentially increasing dilation
        self.tcn = nn.Sequential(*[
            TemporalBlock(
                n_inputs=input_dim if i == 0 else hidden_channels,
                n_outputs=hidden_channels,
                kernel_size=kernel_size,
                dilation=2 ** i,
                dropout=dropout,
            )
            for i in range(num_layers)
        ])

        self.ln        = nn.LayerNorm(hidden_channels)
        self.se_reduce = nn.Linear(hidden_channels, se_bottleneck)
        self.se_expand = nn.Linear(se_bottleneck, hidden_channels)

        if use_learned_query:
            self.query_token = nn.Parameter(torch.zeros(1, 1, hidden_channels))
            nn.init.normal_(self.query_token, std=0.02)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=attn_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout_out),
            nn.Linear(hidden_channels // 2, num_classes),
        )

        if bias_init is not None:
            _bias_init(self.fc[3], bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, F)
        B = x.size(0)

        h = self.tcn(x.transpose(1, 2)).transpose(1, 2)  # (B, S, hidden_channels)
        h = self.ln(h)

        # Squeeze-Excite: gate timesteps using the most-recent bars as context.
        # A full-sequence mean dilutes recent signal; for short-horizon entries
        # the last se_context_window bars are more predictive.
        ctx_start = max(0, h.size(1) - self.se_context_window)
        context = h[:, ctx_start:, :].mean(dim=1)
        se = torch.sigmoid(self.se_expand(torch.relu(self.se_reduce(context))))
        h  = h * se.unsqueeze(1)

        # Multi-head attention pooling
        q      = self.query_token.expand(B, -1, -1) if self.use_learned_query else context.unsqueeze(1)
        pooled, _ = self.attn(q, h, h, need_weights=False)
        pooled = pooled.squeeze(1)                       # (B, hidden_channels)

        return self.fc(pooled)

