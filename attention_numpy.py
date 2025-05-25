# attention_numpy.py

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax ligne à ligne pour une matrice 2D"""
    x_max = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - x_max)  # stabilité numérique
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def attention_numpy(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Attention classique : softmax(QKᵀ / sqrt(dₖ)) V
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores)
    return weights @ V
