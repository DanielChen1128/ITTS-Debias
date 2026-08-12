"""Dependency-light linear probes for representation screening.

A small L2-regularised logistic regression trained with full-batch gradient
descent, plus a self-contained ROC-AUC estimator. Kept in pure torch so the
screening pipeline introduces no scikit-learn dependency.
"""

import torch


def roc_auc(scores, labels):
    """Return the ROC-AUC of ``scores`` against binary ``labels`` (0/1).

    Computed via the rank-sum (Mann-Whitney U) identity with average ranks for
    ties, so it is exact and needs no threshold sweep.
    """
    scores = scores.detach().to(dtype=torch.float64).flatten()
    labels = labels.detach().flatten()
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC-AUC requires at least one positive and one negative label")

    order = torch.argsort(scores)
    sorted_scores = scores[order]
    ranks = torch.empty_like(sorted_scores)
    n = sorted_scores.numel()
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        # Average rank (1-indexed) across the tied block.
        ranks[i:j] = (i + j - 1) / 2.0 + 1.0
        i = j
    rank_of_positive = torch.empty_like(ranks)
    rank_of_positive[order] = ranks
    sum_pos_ranks = rank_of_positive[positives].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


class LinearProbe:
    """Standardising L2-regularised logistic regression probe."""

    def __init__(self, weight, bias, mean, std):
        self.weight = weight
        self.bias = bias
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, features, labels, *, l2=1e-2, epochs=300, lr=0.5, seed=0):
        torch.manual_seed(seed)
        x = features.detach().to(dtype=torch.float64)
        y = labels.detach().to(dtype=torch.float64).flatten()
        if x.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("features must be [samples, dim] aligned with labels")
        if set(int(v) for v in y.unique().tolist()) - {0, 1}:
            raise ValueError("labels must be binary 0/1")

        mean = x.mean(dim=0)
        std = x.std(dim=0).clamp_min(1e-8)
        xs = (x - mean) / std

        weight = torch.zeros(xs.shape[1], dtype=torch.float64, requires_grad=True)
        bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([weight, bias], lr=lr, max_iter=epochs, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            logits = xs @ weight + bias
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss = loss + l2 * weight.pow(2).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        return cls(weight.detach(), bias.detach(), mean, std)

    def scores(self, features):
        x = features.detach().to(dtype=torch.float64)
        xs = (x - self.mean) / self.std
        return xs @ self.weight + self.bias

    def evaluate(self, features, labels):
        scores = self.scores(features)
        labels = labels.detach().flatten()
        auc = roc_auc(scores, labels)
        accuracy = float(((scores > 0).long() == labels.long()).float().mean())
        return {"auc": auc, "accuracy": accuracy}
