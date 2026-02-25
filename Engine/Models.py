import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    """Simple additive attention for sequence pooling."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        # h: (batch, seq_len, hidden_dim)
        scores = self.proj(h).squeeze(-1)         # (batch, seq_len)
        weights = torch.softmax(scores, dim=1)    # attention weights
        pooled = (h * weights.unsqueeze(-1)).sum(dim=1)
        return pooled                             # (batch, hidden_dim)


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=2,
        num_classes=3,
        bidirectional=True,
        dropout=0.2,
        dropout_out=0.3,
        use_attention=True,
        use_layer_norm=True,
        use_residual=False,
        bias_init=None
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.dropout_out = dropout_out
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        # Optional layer normalization after LSTM for training stability
        if use_layer_norm:
            self.ln = nn.LayerNorm(lstm_output_dim)
        
        if use_attention:
            self.attention = AttentionPool(lstm_output_dim)
            classifier_input_dim = lstm_output_dim
        else:
            classifier_input_dim = lstm_output_dim

        # Multi-layer classifier head with better regularization
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_out),
            nn.Linear(lstm_output_dim // 2, num_classes)
        )

        # Initialize final output layer bias with log-priors if provided
        if bias_init is not None:
            with torch.no_grad():
                # Access the final Linear layer (index 3 in Sequential)
                self.fc[3].bias.copy_(torch.tensor(bias_init, dtype=torch.float32))
                # Small normal initialization for output weights to prevent saturation
                nn.init.normal_(self.fc[3].weight, mean=0.0, std=0.01)

        

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        h, _ = self.lstm(x)  # (batch, seq_len, hidden_dim * directions)

        # Optional layer normalization for training stability
        if self.use_layer_norm:
            h = self.ln(h)

        if self.use_attention:
            pooled = self.attention(h)       # (batch, hidden_dim)
        else:
            pooled = h[:, -1, :]             # last timestep

        logits = self.fc(pooled)
        return logits


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        seq_len,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        num_classes=3,
        dropout=0.2,
        pooling="cls"  # or "mean"
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pooling = pooling

        # 1. Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # 2. Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # 3. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Classifier head
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        B, S, F = x.shape
        x = self.input_proj(x)  # (B, S, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, S+1, d_model)
        x = x + self.pos_embed[:, :x.size(1), :]

        # Transformer
        x = self.transformer(x)  # (B, S+1, d_model)

        # Pooling
        if self.pooling == "cls":
            rep = x[:, 0, :]  # CLS token
        else:
            rep = x[:, 1:, :].mean(dim=1)  # mean over sequence

        logits = self.fc(rep)
        return logits

class HybridLSTMTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        seq_len,
        lstm_hidden=128,
        lstm_layers=2,
        d_model=128,
        nhead=4,
        num_transformer_layers=2,
        dim_feedforward=256,
        dropout=0.2,
        num_classes=3,
        bidirectional=True,
        pooling="cls",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pooling = pooling


        ## LSTM Encoder
        self.lstm = nn.LSTM(
            input_dim,
            lstm_hidden,
            lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_output_dim = lstm_hidden * (2 if bidirectional else 1)

        # Project LSTM output -> Transformer dimensions
        self.project = nn.Linear(lstm_output_dim, d_model)


        ## Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))


        ## Classifier
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        B, S, F = x.shape

        # 1) LSTM
        lstm_out, _ = self.lstm(x)  # (B, seq_len, lstm_output_dim)
        tokens = self.project(lstm_out)  # (B, seq_len, d_model)

        # 2) Transformer: prepend CLS
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, seq_len+1, d_model)

        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]

        # 3) Transformer encoder
        tokens = self.transformer(tokens)

        # 4) Pooling
        if self.pooling == "cls":
            rep = tokens[:, 0, :]
        else:
            rep = tokens.mean(dim=1)

        # 5) Classifier
        logits = self.fc(rep)
        return logits
