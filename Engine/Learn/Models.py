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


class LSTMAttentionSEClassifier(nn.Module):
    """
    LSTM + LayerNorm + Squeeze-and-Excite over time + Scaled-Dot Attention pooling.

    Designed for FX microstructure where salient patterns are sparse; attention focuses
    on informative timesteps while SE gates reduce noise. Minimal API change vs LSTMClassifier.
    """
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=2,
        num_classes=3,
        bidirectional=True,
        dropout=0.2,
        dropout_out=0.3,
    ):
        super().__init__()

        self.bidirectional = bidirectional
        self.dropout_out = dropout_out

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.ln = nn.LayerNorm(lstm_output_dim)

        # Squeeze-and-Excitation across time (gate important timesteps)
        # Implemented as 2-layer MLP on mean pooled representation to get a scalar gate per timestep.
        self.se_reduce = nn.Linear(lstm_output_dim, max(8, lstm_output_dim // 8))
        self.se_expand = nn.Linear(max(8, lstm_output_dim // 8), lstm_output_dim)

        # Scaled-Dot Attention pooling: query is last state, keys=values are sequence
        self.q_proj = nn.Linear(lstm_output_dim, lstm_output_dim)
        self.k_proj = nn.Linear(lstm_output_dim, lstm_output_dim)
        self.v_proj = nn.Linear(lstm_output_dim, lstm_output_dim)

        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_out),
            nn.Linear(lstm_output_dim // 2, num_classes),
        )

    def forward(self, x):
        # x: (B, S, F)
        h, (hn, cn) = self.lstm(x)  # h: (B, S, H)
        h = self.ln(h)

        # Squeeze-Excite over time: create gates per timestep
        # Use global mean over time as context
        context = h.mean(dim=1)  # (B, H)
        se = torch.relu(self.se_reduce(context))
        se = torch.sigmoid(self.se_expand(se))  # (B, H)
        h = h * se.unsqueeze(1)  # (B, S, H)

        # Attention pooling: query from last layer hidden (concat directions if bidirectional)
        # hn: (num_layers * num_directions, B, hidden_dim)
        num_dirs = 2 if self.bidirectional else 1
        last_hn = hn[-num_dirs:]                         # (num_dirs, B, hidden_dim)
        q = last_hn.transpose(0, 1).reshape(h.size(0), -1)  # (B, hidden_dim * num_dirs) == lstm_output_dim
        q = self.q_proj(q).unsqueeze(1)                # (B, 1, H)
        k = self.k_proj(h)                             # (B, S, H)
        v = self.v_proj(h)                             # (B, S, H)

        # scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(1, 2)) / (k.size(-1) ** 0.5)  # (B, 1, S)
        attn_weights = torch.softmax(attn_scores, dim=-1)                       # (B, 1, S)
        pooled = torch.matmul(attn_weights, v).squeeze(1)                       # (B, H)

        logits = self.fc(pooled)
        return logits


class TransformerSEClassifier(nn.Module):
    """
    Transformer + Squeeze-and-Excite + Multi-Head Self-Attention pooling.
    
    Designed for FX microstructure classification. Uses:
    - Transformer encoder for capturing long-range temporal dependencies
    - Squeeze-and-Excite across time for gating important timesteps
    - Multi-head attention pooling to focus on salient patterns
    - Learnable positional embeddings for temporal awareness
    
    Similar architecture philosophy to LSTMAttentionSEClassifier but with transformer backbone.
    """
    def __init__(
        self,
        input_dim,
        seq_len,
        d_model=256,
        nhead=8,
        num_layers=3,
        dim_feedforward=512,
        num_classes=3,
        dropout=0.2,
        dropout_out=0.3,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.dropout_out = dropout_out
        self.seq_len = seq_len
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization after transformer
        self.ln = nn.LayerNorm(d_model)
        
        # Squeeze-and-Excitation across time (gate important timesteps)
        # Same design as LSTMAttentionSEClassifier
        self.se_reduce = nn.Linear(d_model, max(8, d_model // 8))
        self.se_expand = nn.Linear(max(8, d_model // 8), d_model)
        
        # Multi-head attention pooling (collapse sequence to single representation)
        # Query is learned CLS-like token, keys/values are sequence
        self.query_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.query_token, mean=0.0, std=0.02)
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Classifier head
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_out),
            nn.Linear(d_model // 2, num_classes),
        )
    
    def forward(self, x):
        # x: (B, S, F)
        B, S, F = x.shape
        
        # Project input to d_model dimensions
        x = self.input_proj(x)  # (B, S, d_model)
        
        # Add positional embeddings
        x = x + self.pos_embed[:, :S, :]
        
        # Transformer encoding
        h = self.transformer(x)  # (B, S, d_model)
        h = self.ln(h)
        
        # Squeeze-Excite over time: create gates per timestep
        # Use global mean over time as context
        context = h.mean(dim=1)  # (B, d_model)
        se = torch.relu(self.se_reduce(context))
        se = torch.sigmoid(self.se_expand(se))  # (B, d_model)
        h = h * se.unsqueeze(1)  # (B, S, d_model)
        
        # Attention pooling: query from learned token, keys/values from sequence
        q = self.query_token.expand(B, -1, -1)  # (B, 1, d_model)
        q = self.q_proj(q)                      # (B, 1, d_model)
        k = self.k_proj(h)                      # (B, S, d_model)
        v = self.v_proj(h)                      # (B, S, d_model)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(1, 2)) / (self.d_model ** 0.5)  # (B, 1, S)
        attn_weights = torch.softmax(attn_scores, dim=-1)                         # (B, 1, S)
        pooled = torch.matmul(attn_weights, v).squeeze(1)                         # (B, d_model)
        
        # Classification
        logits = self.fc(pooled)
        return logits


class TemporalBlock(nn.Module):
    """
    Residual block for TCN with dilated causal convolutions.
    
    Features:
    - Dilated causal convolutions for large receptive field
    - Weight normalization for training stability
    - Residual connection with optional projection
    - Dropout for regularization
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()
        
        # Calculate padding for causal convolution (no future information leakage)
        self.padding = (kernel_size - 1) * dilation
        
        # First convolutional layer
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, dilation=dilation, padding=self.padding)
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second convolutional layer
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, dilation=dilation, padding=self.padding)
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection with 1x1 conv if dimensions don't match
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu_out = nn.ReLU()
        
    def forward(self, x):
        # x: (B, C, T)
        # Apply causal convolutions
        out = self.conv1(x)
        out = out[:, :, :-self.padding] if self.padding > 0 else out  # Remove future padding
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


# class TCNAttentionSEClassifier(nn.Module):
#     """
#     Temporal Convolutional Network + Squeeze-and-Excite + Attention pooling.
    
#     Designed for binary classification of FX microstructure time series. Uses:
#     - Dilated causal convolutions for large receptive fields without future leakage
#     - Residual connections for gradient flow in deep networks
#     - Squeeze-and-Excite gating to focus on important temporal patterns
#     - Scaled-Dot Attention pooling for adaptive sequence aggregation
#     - Lightweight architecture suitable for high-frequency trading signals
    
#     Architecture inspired by LSTMAttentionSEClassifier but using TCN backbone for:
#     - Faster training and inference (parallelizable convolutions)
#     - Explicit control over receptive field via dilation
#     - Better gradient flow in very long sequences
    
#     Args:
#         input_dim: Number of input features per timestep
#         hidden_channels: Number of channels in TCN layers (default: 128)
#         num_layers: Number of temporal blocks (default: 4)
#         kernel_size: Kernel size for convolutions (default: 3)
#         num_classes: Number of output classes (default: 2 for binary)
#         dropout: Dropout rate within TCN blocks (default: 0.2)
#         dropout_out: Dropout rate before final classifier (default: 0.3)
#     """
#     def __init__(
#         self,
#         input_dim,
#         hidden_channels=128,
#         num_layers=4,
#         kernel_size=3,
#         num_classes=2,
#         dropout=0.2,
#         dropout_out=0.3,
#     ):
#         super().__init__()
        
#         self.hidden_channels = hidden_channels
#         self.dropout_out = dropout_out
        
#         # Build TCN with exponentially increasing dilation
#         layers = []
#         num_levels = num_layers
#         for i in range(num_levels):
#             dilation = 2 ** i
#             in_channels = input_dim if i == 0 else hidden_channels
#             layers.append(
#                 TemporalBlock(
#                     in_channels,
#                     hidden_channels,
#                     kernel_size,
#                     dilation=dilation,
#                     dropout=dropout
#                 )
#             )
#         self.tcn = nn.Sequential(*layers)
        
#         # Layer normalization after TCN for training stability
#         self.ln = nn.LayerNorm(hidden_channels)
        
#         # Squeeze-and-Excitation across time (gate important timesteps)
#         # Same design as LSTMAttentionSEClassifier
#         self.se_reduce = nn.Linear(hidden_channels, max(8, hidden_channels // 8))
#         self.se_expand = nn.Linear(max(8, hidden_channels // 8), hidden_channels)
        
#         # Scaled-Dot Attention pooling
#         # Use global pooling as query, sequence as keys/values
#         self.q_proj = nn.Linear(hidden_channels, hidden_channels)
#         self.k_proj = nn.Linear(hidden_channels, hidden_channels)
#         self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        
#         # Classifier head
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_channels, hidden_channels // 2),
#             nn.ReLU(),
#             nn.Dropout(self.dropout_out),
#             nn.Linear(hidden_channels // 2, num_classes),
#         )
    
#     def forward(self, x):
#         # x: (B, S, F) - batch, sequence, features
#         B, S, F = x.shape
        
#         # TCN expects (B, C, T) - batch, channels, time
#         x = x.transpose(1, 2)  # (B, F, S)
#         h = self.tcn(x)        # (B, hidden_channels, S)
#         h = h.transpose(1, 2)  # (B, S, hidden_channels)
        
#         # Layer normalization
#         h = self.ln(h)
        
#         # Squeeze-Excite over time: create gates per timestep
#         # Use global mean over time as context
#         context = h.mean(dim=1)  # (B, hidden_channels)
#         se = torch.relu(self.se_reduce(context))
#         se = torch.sigmoid(self.se_expand(se))  # (B, hidden_channels)
#         h = h * se.unsqueeze(1)  # (B, S, hidden_channels)
        
#         # Attention pooling: query from global context, keys/values from sequence
#         q = context.unsqueeze(1)           # (B, 1, hidden_channels)
#         q = self.q_proj(q)                 # (B, 1, hidden_channels)
#         k = self.k_proj(h)                 # (B, S, hidden_channels)
#         v = self.v_proj(h)                 # (B, S, hidden_channels)
        
#         # Scaled dot-product attention
#         attn_scores = torch.matmul(q, k.transpose(1, 2)) / (self.hidden_channels ** 0.5)  # (B, 1, S)
#         attn_weights = torch.softmax(attn_scores, dim=-1)                                 # (B, 1, S)
#         pooled = torch.matmul(attn_weights, v).squeeze(1)                                 # (B, hidden_channels)
        
#         # Classification
#         logits = self.fc(pooled)
#         return logits

class TCNAttentionSEClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_channels=128,
        num_layers=4,
        kernel_size=3,
        num_classes=2,
        dropout=0.2,
        dropout_out=0.3,
        attn_heads=4,
        attn_dropout=0.1,
        use_learned_query=True,
        bias_init=None,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.dropout_out = dropout_out
        self.use_learned_query = use_learned_query

        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_channels = input_dim if i == 0 else hidden_channels
            layers.append(
                TemporalBlock(
                    in_channels,
                    hidden_channels,
                    kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        self.tcn = nn.Sequential(*layers)

        self.ln = nn.LayerNorm(hidden_channels)

        self.se_reduce = nn.Linear(hidden_channels, max(8, hidden_channels // 8))
        self.se_expand = nn.Linear(max(8, hidden_channels // 8), hidden_channels)

        # Multi-head attention pooling (more compute than single-head)
        if self.use_learned_query:
            self.query_token = nn.Parameter(torch.zeros(1, 1, hidden_channels))
            nn.init.normal_(self.query_token, mean=0.0, std=0.02)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=attn_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_out),
            nn.Linear(hidden_channels // 2, num_classes),
        )

        if bias_init is not None:
            with torch.no_grad():
                self.fc[3].bias.copy_(torch.tensor(bias_init, dtype=torch.float32))
                nn.init.normal_(self.fc[3].weight, mean=0.0, std=0.01)

    def forward(self, x):
        B, S, F = x.shape

        x = x.transpose(1, 2)  # (B, F, S)
        h = self.tcn(x)        # (B, hidden_channels, S)
        h = h.transpose(1, 2)  # (B, S, hidden_channels)

        h = self.ln(h)

        context = h.mean(dim=1)  # (B, hidden_channels)
        se = torch.relu(self.se_reduce(context))
        se = torch.sigmoid(self.se_expand(se))  # (B, hidden_channels)
        h = h * se.unsqueeze(1)  # (B, S, hidden_channels)

        # Attention pooling
        if self.use_learned_query:
            q = self.query_token.expand(B, -1, -1)  # (B, 1, hidden_channels)
        else:
            q = context.unsqueeze(1)                # (B, 1, hidden_channels)

        pooled, _ = self.attn(q, h, h, need_weights=False)  # (B, 1, hidden_channels)
        pooled = pooled.squeeze(1)                          # (B, hidden_channels)

        logits = self.fc(pooled)
        return logits