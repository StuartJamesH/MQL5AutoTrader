import torch
from torch import nn
import torch.nn.functional as F

class TradeProfitabilityLoss(nn.Module):
    """Directional precision-optimised loss for multiclass trade entry classification.

    Designed for 3-class (SELL=0 / FLAT=1 / BUY=2) trade entry models where
    high-precision trade predictions are the primary objective, with a hard
    floor on per-class recall to prevent minority-class collapse.

    Three structural improvements over the previous implementation:

    1. **Per-class precision** (SELL and BUY measured independently).
       The old combined p_sell+p_buy formulation treated a BUY prediction on a
       SELL bar as a soft true-positive — rewarding the wrong direction as
       precision.  Each class is now measured against its own denominator.

    2. **Recall floor hinge** (replaces linear recall penalty).
       A linear recall penalty constantly opposes precision.  The hinge is zero
       when recall >= recall_floor and increases quadratically only below it.
       This prevents minority-class collapse without competing with precision.

    3. **Direction confusion penalty**.
       Explicitly penalises the model when it concentrates probability mass in
       the wrong trade direction (BUY mass on SELL bars, or SELL mass on BUY bars).

    Forward signature is identical to the previous class:
        loss = criterion(logits, targets, trade_outcomes)
    ``trade_outcomes`` remains optional; when profit_weight > 0 it provides
    realised sell/buy barrier outcomes for the expected-profit term.
    """

    def __init__(
        self,
        alpha=None,
        gamma: float = 2.5,
        trade_classes=(0, 2),
        pr_weight: float = 10.0,
        pr_weight_sell: float = None,
        pr_weight_buy: float = None,
        precision_power: float = 1.0,
        recall_floor: float = 0.15,
        rec_floor_weight: float = 20.0,
        direction_penalty: float = 1.5,
        profit_weight: float = 0.0,
        eps: float = 1e-6,
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            alpha:             Class weights tensor for focal CE [SELL, FLAT, BUY].
            gamma:             Focal loss exponent. Higher = more focus on hard examples.
            trade_classes:     (sell_idx, buy_idx) — indices of the two trade classes.
            pr_weight:         Base weight on the mean per-class precision penalty.
                               Retained for backward compatibility and used when
                               pr_weight_sell/pr_weight_buy are not provided.
            pr_weight_sell:    SELL precision penalty weight. If None, uses pr_weight.
            pr_weight_buy:     BUY precision penalty weight. If None, uses pr_weight.
            precision_power:   Exponent for the precision penalty (default 1.0 = linear).
                               Set to 2.0 for quadratic: stronger gradient when precision is
                               far below target, gentler as precision improves. Quadratic
                               requires pr_weight to be scaled up by ~1.56× vs linear to
                               maintain equivalent gradient at precision=0 (e.g. linear
                               pr_weight=7.5 ≈ quadratic pr_weight=11.7 at prec=0.0).
            recall_floor:      Minimum acceptable recall for each trade class before
                               the hinge penalty activates. Default 0.15 prevents
                               collapse without fighting precision above the floor.
            rec_floor_weight:  Strength of the quadratic recall floor hinge.
                               Increase (e.g. to 30–40) if a class still collapses.
            direction_penalty: Weight on the SELL↔BUY direction confusion penalty.
            profit_weight:     Weight on the expected-profit term. When > 0, uses precomputed
                               per-bar trade outcomes (shape [B, 2]: sell_outcome, buy_outcome;
                               values +1=TP hit, -1=SL hit, 0=unresolved) passed as
                               trade_outcomes to forward(). The term is
                               -profit_weight × mean(p_sell × sell_out + p_buy × buy_out).
                               Unlike the precision penalty, this is label-agnostic and rewards
                               predictions on ANY bar with a profitable realised outcome,
                               including regime-filtered bars that the labeller assigned FLAT.
                               Set to 0.0 (default) to disable and preserve existing behaviour.
            eps:               Numerical stability constant.
            label_smoothing:   Label smoothing coefficient for focal CE (0.0 = off, 0.05 recommended).
                               Bounds the maximum CE gradient, smoothing the precision↔recall adversarial cycle.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = float(gamma)
        self.sell_cls = int(trade_classes[0])
        self.buy_cls  = int(trade_classes[1])
        self.pr_weight        = float(pr_weight)
        self.pr_weight_sell   = float(pr_weight_sell) if pr_weight_sell is not None else float(pr_weight)
        self.pr_weight_buy    = float(pr_weight_buy) if pr_weight_buy is not None else float(pr_weight)
        self.precision_power  = float(precision_power)  # 1.0 = linear (default), 2.0 = quadratic
        self.recall_floor     = float(recall_floor)
        self.rec_floor_weight = float(rec_floor_weight)
        self.direction_penalty = float(direction_penalty)
        self.profit_weight = float(profit_weight)
        self.eps = float(eps)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, targets, trade_outcomes=None, return_components: bool = False):
        probs = torch.softmax(logits, dim=1)

        # ── 1. Focal cross-entropy ────────────────────────────────────────────
        ce    = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none', label_smoothing=self.label_smoothing)
        pt    = probs[torch.arange(len(targets)), targets]
        focal = ((1.0 - pt) ** self.gamma) * ce
        mean_focal = focal.mean()

        p_sell    = probs[:, self.sell_cls]          # (B,)
        p_buy     = probs[:, self.buy_cls]           # (B,)
        true_sell = (targets == self.sell_cls).float()
        true_buy  = (targets == self.buy_cls).float()

        # ── 2. Per-class soft precision (SELL and BUY independently) ─────────
        prec_sell = (p_sell * true_sell).sum() / (p_sell.sum() + self.eps)
        prec_buy  = (p_buy  * true_buy ).sum() / (p_buy.sum()  + self.eps)
        precision_loss = (
            self.pr_weight_sell * (1.0 - prec_sell) ** self.precision_power +
            self.pr_weight_buy  * (1.0 - prec_buy)  ** self.precision_power
        ) / 2.0

        # ── 3. Recall floor hinge (quadratic below floor, zero above) ────────
        rec_sell = (p_sell * true_sell).sum() / (true_sell.sum() + self.eps)
        rec_buy  = (p_buy  * true_buy ).sum() / (true_buy.sum()  + self.eps)
        hinge_sell = F.relu(self.recall_floor - rec_sell) ** 2
        hinge_buy  = F.relu(self.recall_floor - rec_buy)  ** 2
        recall_loss = self.rec_floor_weight * (hinge_sell + hinge_buy)

        # ── 4. Direction confusion penalty ────────────────────────────────────
        n_trade = true_sell.sum() + true_buy.sum() + self.eps
        confusion = (
            (p_buy  * true_sell).sum() +
            (p_sell * true_buy ).sum()
        ) / n_trade
        confusion_loss = self.direction_penalty * confusion

        # ── 5. Expected profit (label-agnostic; uses realised barrier outcomes) ──
        # trade_outcomes shape: (B, 2) — [:, 0]=sell_outcome, [:, 1]=buy_outcome
        # Values: +1 (TP hit), -1 (SL hit), 0 (unresolved/neutral)
        # Rewards p_sell↑ on bars where sell is profitable regardless of label.
        # Handles direction implicitly: predicting BUY on a bar where sell_out=+1
        # but buy_out=-1 incurs a buy penalty without needing direction_penalty.
        if self.profit_weight > 0.0 and trade_outcomes is not None:
            sell_out = trade_outcomes[:, 0].float()   # (B,)
            buy_out  = trade_outcomes[:, 1].float()   # (B,)
            expected_profit = (p_sell * sell_out + p_buy * buy_out).mean()
            profit_loss = -self.profit_weight * expected_profit
        else:
            profit_loss = torch.tensor(0.0, device=logits.device)

        total = mean_focal + precision_loss + recall_loss + confusion_loss + profit_loss
        if return_components:
            return {
                "total":           float(total.item()) if not total.requires_grad else total,
                "focal_ce":        float(mean_focal.item()) if not mean_focal.requires_grad else mean_focal,
                "precision_loss":  float(precision_loss.item()) if not precision_loss.requires_grad else precision_loss,
                "recall_loss":     float(recall_loss.item()) if not recall_loss.requires_grad else recall_loss,
                "confusion_loss":  float(confusion_loss.item()) if not confusion_loss.requires_grad else confusion_loss,
                "profit_loss":     float(profit_loss.item()) if not profit_loss.requires_grad else profit_loss,
            }
        return total


class BinaryTradeProfitabilityLoss(nn.Module):
    """Binary classification version of TradeProfitabilityLoss.
    
    Optimizes for trade profitability using binary classification (trade vs hold).
    Designed for separate BUY and SELL models that each predict trade vs hold.
    
    Key features:
    - Rewards predictions that lead to profitable trades
    - Penalizes predictions that lead to unprofitable trades
    - Maintains precision/recall optimization for the trade class
    - Uses triple barrier outcomes (TP/SL) to compute profitability
    
    Usage:
        For BUY model:
            loss = BinaryTradeProfitabilityLoss()
            loss_value = loss(logits, targets, trade_outcomes[:, 1])  # buy outcomes
        
        For SELL model:
            loss = BinaryTradeProfitabilityLoss()
            loss_value = loss(logits, targets, trade_outcomes[:, 0])  # sell outcomes
        
        trade_outcomes values:
            1 = Trade hit Take Profit (profitable)
            0 = Trade hit timeout/vertical barrier (neutral)
           -1 = Trade hit Stop Loss (unprofitable)
    """
    
    def __init__(self,
                 alpha=None,
                 gamma: float = 2.0,
                 trade_class: int = 1,
                 pr_weight: float = 8.0,
                 rec_weight: float = 8.0,
                 f1_weight: float = 0.0,
                 profit_weight: float = 5.0,
                 loss_penalty: float = 3.0,
                 direction_bonus: float = 1.0,
                 eps: float = 1e-6):
        """
        Args:
            alpha: Class weights for focal loss [weight_hold, weight_trade]
            gamma: Focal loss gamma parameter
            trade_class: Which class index represents trade (default: 1 for binary)
            pr_weight: Weight for precision penalty
            rec_weight: Weight for recall penalty
            f1_weight: Weight for F1 penalty
            profit_weight: Weight for profitability reward
            loss_penalty: Additional penalty for predicting trades that hit SL
            direction_bonus: Bonus when prediction matches profitable outcome
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.trade_class = trade_class
        self.hold_class = 1 - trade_class  # opposite class
        self.pr_weight = float(pr_weight)
        self.rec_weight = float(rec_weight)
        self.f1_weight = float(f1_weight)
        self.profit_weight = float(profit_weight)
        self.loss_penalty = float(loss_penalty)
        self.direction_bonus = float(direction_bonus)
        self.eps = float(eps)

    def forward(self, logits, targets, trade_outcomes=None):
        """
        Args:
            logits: Model predictions (batch_size, 2) for binary classification
            targets: True labels (batch_size,) with 0=hold, 1=trade
            trade_outcomes: Trade profitability outcomes (batch_size,)
                Values: 1 = TP hit, 0 = timeout, -1 = SL hit
        """
        # --- Base Focal Loss ---
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        probs = torch.softmax(logits, dim=1)
        pt = probs[torch.arange(len(targets)), targets]
        focal = ((1.0 - pt) ** self.gamma) * ce
        mean_focal = focal.mean()

        # --- Precision/Recall Optimization ---
        # Probability of predicting "trade"
        p_trade = probs[:, self.trade_class]
        
        # True positive mask: true label is trade
        tp_mask = (targets == self.trade_class)
        tn_mask = ~tp_mask  # true label is hold

        # Soft counts
        TP = (p_trade * tp_mask.float()).sum()
        FP = (p_trade * tn_mask.float()).sum()
        FN = ((1.0 - p_trade) * tp_mask.float()).sum()

        precision = TP / (TP + FP + self.eps)
        recall = TP / (TP + FN + self.eps)

        pr_loss = 1.0 - precision
        rec_loss = 1.0 - recall
        total_pr_rec = self.pr_weight * pr_loss + self.rec_weight * rec_loss

        if self.f1_weight and (self.f1_weight > 0.0):
            f1 = 2.0 * precision * recall / (precision + recall + self.eps)
            f1_loss = 1.0 - f1
            total_pr_rec = total_pr_rec + self.f1_weight * f1_loss

        # --- Profitability Optimization ---
        profit_penalty = 0.0
        
        if trade_outcomes is not None:
            # Ensure trade_outcomes is a tensor
            if not isinstance(trade_outcomes, torch.Tensor):
                trade_outcomes = torch.tensor(trade_outcomes, dtype=torch.float32, device=logits.device)
            
            # Get predicted class for each sample
            pred_class = logits.argmax(dim=1)
            
            # Mask for samples where we predicted "trade"
            trade_pred_mask = (pred_class == self.trade_class)
            
            if trade_pred_mask.any():
                # Get outcomes and probabilities for predicted trades
                outcomes_active = trade_outcomes[trade_pred_mask]
                p_trade_active = p_trade[trade_pred_mask]
                n_trade_preds = trade_pred_mask.sum().float()
                
                # --- Normalized Reward for profitable predictions ---
                profitable_mask = (outcomes_active == 1)
                n_profitable = profitable_mask.sum().float()
                
                if n_profitable > 0:
                    p_trade_profitable = p_trade_active[profitable_mask]
                    profit_score = p_trade_profitable.sum() / (n_trade_preds + self.eps)
                    profit_loss = 1.0 - profit_score
                    profit_penalty = profit_penalty + self.profit_weight * profit_loss
                
                # --- Normalized Penalty for unprofitable predictions ---
                unprofitable_mask = (outcomes_active == -1)
                n_unprofitable = unprofitable_mask.sum().float()
                
                if n_unprofitable > 0:
                    p_trade_unprofitable = p_trade_active[unprofitable_mask]
                    loss_score = p_trade_unprofitable.sum() / (n_trade_preds + self.eps)
                    profit_penalty = profit_penalty + self.loss_penalty * loss_score
                
                # --- Normalized Direction Bonus ---
                # Reward when model correctly predicts trade for profitable outcomes
                if self.direction_bonus > 0.0 and n_profitable > 0:
                    targets_active = targets[trade_pred_mask]
                    pred_profitable = pred_class[trade_pred_mask][profitable_mask]
                    targets_profitable = targets_active[profitable_mask]
                    
                    # Reward when we predicted trade and it was actually a trade signal
                    correct_direction = (pred_profitable == targets_profitable).float()
                    p_correct = p_trade_active[profitable_mask]
                    
                    direction_score = (p_correct * correct_direction).sum() / (n_trade_preds + self.eps)
                    direction_loss = 1.0 - direction_score
                    profit_penalty = profit_penalty + self.direction_bonus * direction_loss

        # --- Combine all losses ---
        loss = mean_focal + total_pr_rec + profit_penalty
        return loss


class ProfitOnlyLoss(nn.Module):
    """Pure profit optimization loss - NO classification accuracy component.
    
    This loss function ONLY cares about maximizing profit from predicted trades.
    It completely ignores whether the model is classifying correctly.
    
    WARNING: This can be unstable and may lead to degenerate solutions!
    Recommended usage: Fine-tuning AFTER initial training with TradePRLoss or TradeProfitabilityLoss.
    
    Use cases:
    - Phase 2 of two-stage training (refine already-trained model)
    - When you want pure profit maximization regardless of classification metrics
    - Experimental/research purposes
    
    Usage:
        loss = ProfitOnlyLoss(
            profit_weight=10.0,     # reward for TP predictions
            loss_penalty=10.0,      # penalty for SL predictions
            entropy_reg=0.01        # prevent overconfident bad predictions
        )
        
        # During training:
        loss_value = loss(logits, targets, trade_outcomes=outcomes)
    """
    
    def __init__(self,
                 trade_classes=(0, 2),
                 profit_weight: float = 10.0,
                 loss_penalty: float = 10.0,
                 timeout_penalty: float = 1.0,
                 entropy_reg: float = 0.01,
                 min_trade_prob: float = 0.3,
                 eps: float = 1e-6):
        """
        Args:
            trade_classes: Which class indices represent trades (0=Short, 2=Long)
            profit_weight: Reward multiplier for profitable trades (TP)
            loss_penalty: Penalty multiplier for losing trades (SL)
            timeout_penalty: Small penalty for timeout trades (encourages selectivity)
            entropy_reg: Entropy regularization weight (prevents overconfident predictions)
            min_trade_prob: Minimum probability threshold to count as a "real" trade prediction
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.trade_classes = tuple(trade_classes)
        self.profit_weight = float(profit_weight)
        self.loss_penalty = float(loss_penalty)
        self.timeout_penalty = float(timeout_penalty)
        self.entropy_reg = float(entropy_reg)
        self.min_trade_prob = float(min_trade_prob)
        self.eps = float(eps)

    def forward(self, logits, targets, trade_outcomes=None):
        """
        Args:
            logits: Model predictions (batch_size, num_classes)
            targets: True labels (batch_size,) - not used for loss, only for logging
            trade_outcomes: Trade profitability outcomes (batch_size, 2)
                Column 0: sell/short outcomes (for class 0 predictions)
                Column 1: buy/long outcomes (for class 2 predictions)
                Values: 1 = TP, 0 = timeout, -1 = SL
        """
        if trade_outcomes is None:
            raise ValueError("ProfitOnlyLoss requires trade_outcomes to be provided!")
        
        # Ensure trade_outcomes is a tensor of shape (batch_size, 2)
        if not isinstance(trade_outcomes, torch.Tensor):
            trade_outcomes = torch.tensor(trade_outcomes, dtype=torch.float32, device=logits.device)
        
        probs = torch.softmax(logits, dim=1)
        batch_size = len(logits)
        
        # --- Calculate expected profit for each sample ---
        # For each sample, compute: P(short) * short_outcome + P(long) * long_outcome
        
        p_short = probs[:, 0]  # probability of predicting short
        p_long = probs[:, 2]   # probability of predicting long
        
        short_outcomes = trade_outcomes[:, 0]  # sell outcomes
        long_outcomes = trade_outcomes[:, 1]   # buy outcomes
        
        # Expected profit per sample (can be positive or negative)
        expected_profit = p_short * short_outcomes + p_long * long_outcomes
        
        # --- Convert to loss (we want to MAXIMIZE profit, so MINIMIZE negative profit) ---
        profit_loss = -expected_profit.mean()
        
        # --- Entropy Regularization ---
        # Prevents the model from becoming overconfident on bad predictions
        # Encourages exploration and prevents collapse
        entropy = -(probs * torch.log(probs + self.eps)).sum(dim=1).mean()
        entropy_loss = -self.entropy_reg * entropy  # negative because we want to maximize entropy
        
        # --- Optional: Penalize low-confidence "trade" predictions ---
        # This encourages the model to be decisive (either trade confidently or don't trade)
        p_trade = p_short + p_long
        confidence_mask = (p_trade > self.min_trade_prob) & (p_trade < 0.95)
        if confidence_mask.any():
            # Small penalty for wishy-washy predictions
            wishy_washy_penalty = 0.1 * confidence_mask.float().mean()
        else:
            wishy_washy_penalty = 0.0
        
        # --- Total Loss ---
        loss = profit_loss + entropy_loss + wishy_washy_penalty
        
        return loss


class TradePRLoss(nn.Module):
        """Batch-level surrogate loss that directly targets precision and recall
        for the two trade classes (default classes `0` and `2`).

        Implementation notes
        - We compute a soft positive probability `p_pos` per sample by summing
            the model's probabilities for the trade classes.
        - Using the batch, we compute soft TP/FP/FN as sums over `p_pos` and
            differentiate using those soft counts to form precision and recall
            (both are differentiable w.r.t. logits through the softmax).
        - Final loss = mean focal CE + pr_weight*(1-precision) + rec_weight*(1-recall)
            (optionally include an F1-based penalty via `f1_weight`).

        This gives the optimizer a direct signal to increase precision and recall
        for trade predictions while still keeping a per-sample focal CE term
        for stability.
        """

        def __init__(self,
                     alpha=None,
                     gamma: float = 2.0,
                     trade_classes=(0, 2),
                     pr_weight: float = 8.0,
                     rec_weight: float = 8.0,
                     f1_weight: float = 0.0,
                     eps: float = 1e-6):
            
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.trade_classes = tuple(trade_classes)
                self.pr_weight = float(pr_weight)
                self.rec_weight = float(rec_weight)
                self.f1_weight = float(f1_weight)
                self.eps = float(eps)

        def forward(self, logits, targets):
                # per-sample focal CE base (keeps training stable)
                ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
                probs = torch.softmax(logits, dim=1)
                pt = probs[torch.arange(len(targets)), targets]
                focal = ((1.0 - pt) ** self.gamma) * ce
                mean_focal = focal.mean()

                # soft positive probability for "any trade" = sum p(class) over trade classes
                trade_idx = list(self.trade_classes)
                p_pos = probs[:, trade_idx].sum(dim=1)  # shape (batch,)

                # boolean masks for true positive (true is trade) vs true negative (true is flat)
                # true positive: target in trade_classes
                tp_mask = torch.zeros_like(p_pos, dtype=torch.bool)
                for c in trade_idx:
                        tp_mask = tp_mask | (targets == c)

                tn_mask = ~tp_mask

                # soft counts across the batch
                TP = (p_pos * tp_mask.float()).sum()
                FP = (p_pos * tn_mask.float()).sum()
                FN = (((1.0 - p_pos) * tp_mask.float())).sum()

                precision = TP / (TP + FP + self.eps)
                recall = TP / (TP + FN + self.eps)

                pr_loss = 1.0 - precision
                rec_loss = 1.0 - recall

                total_pr_rec = self.pr_weight * pr_loss + self.rec_weight * rec_loss

                if self.f1_weight and (self.f1_weight > 0.0):
                        f1 = 2.0 * precision * recall / (precision + recall + self.eps)
                        f1_loss = 1.0 - f1
                        total_pr_rec = total_pr_rec + self.f1_weight * f1_loss

                # Combine per-sample loss with batch-level PR/rec penalty (broadcast scalar)
                loss = mean_focal + total_pr_rec
                return loss

class ConfidenceBasedLoss(nn.Module):
    def __init__(self,
                 alpha=None,
                 gamma=2.0,
                 flat_confidence_threshold=0.7,  # require high confidence for flat
                 flat_penalty_weight=3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.flat_confidence_threshold = flat_confidence_threshold
        self.flat_penalty_weight = flat_penalty_weight

    def forward(self, logits, targets):
        # Focal loss
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        probs = torch.softmax(logits, dim=1)
        pt = probs[torch.arange(len(targets)), targets]
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # Extra penalty: if true label is FLAT (class 1), penalize low confidence
        flat_mask = (targets == 1)
        if flat_mask.any():
            flat_probs = pt[flat_mask]
            # penalize if flat probability < threshold
            flat_confidence_penalty = torch.relu(self.flat_confidence_threshold - flat_probs)
            focal_loss[flat_mask] = focal_loss[flat_mask] + self.flat_penalty_weight * flat_confidence_penalty

        return focal_loss.mean()

class ClassSeparationLoss(nn.Module):
    def __init__(self,
                 alpha=None,
                 gamma=2.0,
                 margin_trade_vs_flat=0.5,
                 margin_flat_vs_trade=0.3,
                 penalty_weight=5.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.margin_trade_vs_flat = margin_trade_vs_flat
        self.margin_flat_vs_trade = margin_flat_vs_trade
        self.penalty_weight = penalty_weight

    def forward(self, logits, targets):
        # Focal loss
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        # full per-class probabilities
        probs = torch.softmax(logits, dim=1)
        # probability of the true class for each sample
        pt = probs[torch.arange(len(targets)), targets]
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        penalties = torch.zeros_like(focal_loss)

        # Penalty 1: Trade classes (0, 2) should be far from flat (1)
        trade_mask = (targets == 0) | (targets == 2)
        if trade_mask.any():
            trade_logits = logits[trade_mask]
            true_logits = trade_logits[torch.arange(trade_mask.sum()), targets[trade_mask]]
            flat_logits = trade_logits[:, 1]
            margin_violation = torch.relu(self.margin_trade_vs_flat - (true_logits - flat_logits))
            penalties[trade_mask] = self.penalty_weight * margin_violation

        # Penalty 2: Flat class (1) should be far from both trade classes
        flat_mask = (targets == 1)
        if flat_mask.any():
            flat_logits = logits[flat_mask]
            flat_prob = flat_logits[:, 1]
            trade_prob_max = torch.max(flat_logits[:, 0], flat_logits[:, 2])
            margin_violation = torch.relu(self.margin_flat_vs_trade - (flat_prob - trade_prob_max))
            penalties[flat_mask] = self.penalty_weight * margin_violation

        return (focal_loss + penalties).mean()

class ContrastiveClassificationLoss(nn.Module):
    def __init__(self,
                 alpha=None,
                 gamma=2.0,
                 temperature=0.5,
                 contrast_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.temperature = temperature
        self.contrast_weight = contrast_weight

    def forward(self, logits, targets, embeddings=None):
        # Standard focal loss
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        probs = torch.softmax(logits, dim=1)
        pt = probs[torch.arange(len(targets)), targets]
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # Optional: if you have access to pre-classifier embeddings,
        # add a contrastive term to push classes apart
        # (requires modifying model to return embeddings)

        return focal_loss.mean()

class ImprovedFocalMarginLoss(nn.Module):
    def __init__(self,
                 alpha=None,
                 gamma=2.0,
                 margin=0.5,
                 flat_confidence_threshold=0.75,
                 margin_penalty_weight=5.0,
                 flat_penalty_weight=3.0,
                 trade_fn_penalty_weight: float = 1.0,
                 flat_fp_threshold: float = None,
                 trade_fn_threshold: float = None,
                 trade_classes=(0, 2)):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.margin = margin
        self.flat_confidence_threshold = flat_confidence_threshold
        self.margin_penalty_weight = margin_penalty_weight
        # existing flat_penalty_weight keeps previous behaviour (penalise low p(flat))
        # also expose asymmetric penalties:
        #  - trade_fn_penalty_weight: lighter penalty when trade -> predicted FLAT
        #  - flat_fp_threshold: use for flat->trade detection (defaults to 1-flat_confidence_threshold)
        #  - trade_fn_threshold: threshold above which predicting FLAT for trade is penalised
        self.flat_penalty_weight = flat_penalty_weight
        self.trade_fn_penalty_weight = trade_fn_penalty_weight
        self.flat_fp_threshold = flat_fp_threshold if flat_fp_threshold is not None else (1.0 - flat_confidence_threshold)
        self.trade_fn_threshold = trade_fn_threshold if trade_fn_threshold is not None else flat_confidence_threshold
        self.trade_classes = trade_classes

    def forward(self, logits, targets):
        # Focal loss base
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        probs = torch.softmax(logits, dim=1)
        pt = probs[torch.arange(len(targets)), targets]
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        penalties = torch.zeros_like(focal_loss)
        flat_logit = logits[:, 1]

        # Penalty 1: Margin loss (trade vs flat)
        for cls in self.trade_classes:
            cls_mask = (targets == cls)
            if cls_mask.any():
                cls_logits = logits[cls_mask, cls]
                flat_logits = flat_logit[cls_mask]
                margin_violation = torch.relu(self.margin - (cls_logits - flat_logits))
                penalties[cls_mask] += self.margin_penalty_weight * margin_violation

        # === Penalty 2: Confidence penalty for FLAT class (harsh for flat->trade) ===
        # For examples where target == FLAT (1) we want to penalise the model
        # strongly if it assigns too much probability to either trade class.
        flat_mask = (targets == 1)
        if flat_mask.any():
            # take maximum trade probability (how strongly model favours *any* trade)
            trade_probs = torch.stack([probs[:, 0], probs[:, 2]], dim=1)
            max_trade_prob = trade_probs[flat_mask].max(dim=1).values
            # measure how much the max trade probability exceeds the allowed threshold
            # (i.e. measure propensity to predict trade when it's actually flat)
            flat_fp_violation = torch.relu(max_trade_prob - self.flat_fp_threshold)
            # add a strong penalty for flat->trade errors (configurable)
            penalties[flat_mask] += self.flat_penalty_weight * flat_fp_violation

        # === Penalty 3: *lighter* penalty when a trade example is predicted as FLAT ===
        # For trade examples (target in {0,2}), penalise only lightly if the model
        # places too much probability on FLAT. This controls the asymmetric behavior
        # you requested: prefer erring towards FLAT rather than predicting trade
        # when uncertain.
        trade_mask = (targets == 0) | (targets == 2)
        if trade_mask.any():
            p_flat = probs[trade_mask, 1]
            trade_fn_violation = torch.relu(p_flat - self.trade_fn_threshold)
            penalties[trade_mask] += self.trade_fn_penalty_weight * trade_fn_violation

        return (focal_loss + penalties).mean()
    
class FocalFlatProtectLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, margin=0.5,
                 trade_margin_weight=5.0, flat_penalty_threshold=0.6, flat_penalty_weight=5.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.margin = margin
        self.trade_margin_weight = trade_margin_weight
        self.flat_penalty_threshold = flat_penalty_threshold
        self.flat_penalty_weight = flat_penalty_weight

    def forward(self, logits, targets):
        # base focal CE
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        probs = torch.softmax(logits, dim=1)
        pt = probs[torch.arange(len(targets)), targets]
        focal = ((1.0 - pt) ** self.gamma) * ce

        total_penalty = torch.zeros_like(focal)

        # margin penalty for trade examples (existing behavior)
        flat_logits = logits[:, 1]
        for cls in (0,2):
            mask = (targets == cls)
            if mask.any():
                cls_logits = logits[mask, cls]
                flat_l = flat_logits[mask]
                margin_violation = torch.relu(self.margin - (cls_logits - flat_l))
                total_penalty[mask] += self.trade_margin_weight * margin_violation

        # NEW: penalize when true=FLAT but model has low flat prob (i.e., predicting trade)
        flat_mask = (targets == 1)
        if flat_mask.any():
            flat_probs = probs[flat_mask, 1]   # p(flat)
            # penalty = how much p(flat) is below threshold
            penalty = torch.relu(self.flat_penalty_threshold - flat_probs)
            total_penalty[flat_mask] += self.flat_penalty_weight * penalty

        return (focal + total_penalty).mean()


class TradePrecisionLoss(nn.Module):
    """Loss that prioritises precision for trade classes (0 and 2).

    Design goals:
    - Heavily penalise cases where the model assigns large probability to any
        trade class when the true label is FLAT (reduces false-positive trades).
    - Encourage confident trade predictions when the true label is a trade
        (reduce false-negative trades moderately).
    - Keep a focal/CrossEntropy base so overall training remains stable.
    """

    def __init__(self,
                    alpha=None,
                    gamma=2.0,
                    trade_fp_threshold=0.15,
                    trade_fp_weight=8.0,
                    trade_conf_threshold=0.6,
                    trade_conf_weight=2.0,
                    margin=0.5,
                    margin_weight=4.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        # threshold above which predicted trade prob on a FLAT sample is penalised
        self.trade_fp_threshold = trade_fp_threshold
        # weight for penalising flat->trade (false positive) behavior
        self.trade_fp_weight = trade_fp_weight
        # encourage trade samples to have at least this probability on the true trade
        self.trade_conf_threshold = trade_conf_threshold
        self.trade_conf_weight = trade_conf_weight
        # margin between true trade logit and flat logit
        self.margin = margin
        self.margin_weight = margin_weight

    def forward(self, logits, targets):
        # Base focal cross-entropy (per-sample)
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        probs = torch.softmax(logits, dim=1)
        pt = probs[torch.arange(len(targets)), targets]
        focal = ((1.0 - pt) ** self.gamma) * ce

        penalties = torch.zeros_like(focal)

        # 1) Penalise FLAT targets that have high trade probability (flat -> predicted trade)
        flat_mask = (targets == 1)
        if flat_mask.any():
            # maximum probability the model assigns to any trade class (0 or 2)
            trade_probs = torch.stack([probs[:, 0], probs[:, 2]], dim=1)
            max_trade_prob = trade_probs[flat_mask].max(dim=1).values
            # amount by which max_trade_prob exceeds the allowed FP threshold
            fp_violation = torch.relu(max_trade_prob - self.trade_fp_threshold)
            penalties[flat_mask] += self.trade_fp_weight * fp_violation

        # 2) Encourage true trade examples (target in {0,2}) to be confident
        trade_mask = (targets == 0) | (targets == 2)
        if trade_mask.any():
            # probability assigned to the correct trade class for trade samples
            p_true_trade = probs[trade_mask, targets[trade_mask]]
            # penalise low confidence (below trade_conf_threshold)
            low_confidence = torch.relu(self.trade_conf_threshold - p_true_trade)
            penalties[trade_mask] += self.trade_conf_weight * low_confidence

            # enforce margin between true-trade logit and flat logit
            flat_logits = logits[trade_mask, 1]
            true_logits = logits[trade_mask, targets[trade_mask]]
            margin_violation = torch.relu(self.margin - (true_logits - flat_logits))
            penalties[trade_mask] += self.margin_weight * margin_violation

        return (focal + penalties).mean()