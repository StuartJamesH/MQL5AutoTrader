import numpy as np
import torch
from sklearn.model_selection import train_test_split

def split(X, y, shuffle, test_size):
    idx = np.arange(X.shape[0]).reshape(-1,1)
    X_idx = np.hstack([idx, X])

    X_train, X_test, y_train, y_test = train_test_split(X_idx, y, shuffle=shuffle, test_size=test_size)

    train_idx = X_train[:,0]
    test_idx = X_test[:,0]

    return X_train[:,1:], X_test[:,1:], y_train, y_test, train_idx, test_idx

def conservative_predict(logits, trade_min_prob=0.6, gap=0.15):
    probs = torch.softmax(logits, dim=-1)
    p0, p1, p2 = probs[:,0], probs[:,1], probs[:,2]
    preds = torch.ones(len(probs), dtype=torch.long)  # default FLAT
    # only set to 2 when p2 > trade_min_prob and p2 > p1 + gap
    preds[(p2 > trade_min_prob) & (p2 > p1 + gap)] = 2
    preds[(p0 > trade_min_prob) & (p0 > p1 + gap)] = 0
    return preds