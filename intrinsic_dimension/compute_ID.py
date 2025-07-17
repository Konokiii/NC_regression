import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt


def compute_intrinsic_dimension(M, batch_size=1000, epsilon=0.):
    x, y, _ = compute_twonn_data_batched(M, batch_size=batch_size, epsilon=epsilon)
    return fit_line_through_origin(x, y)


def fit_line_through_origin(x, y, discard_fraction=0.0):
    n_points = x.shape[0]
    n_keep = int(n_points * (1 - discard_fraction))

    x_subset = x[:n_keep]
    y_subset = y[:n_keep]

    slope = torch.sum(x_subset * y_subset) / torch.sum(x_subset ** 2)
    return slope


def compute_twonn_data_batched(X, batch_size=1000, epsilon=0.):
    N = X.shape[0]
    mu = torch.empty((1, N), device=X.device, dtype=X.dtype)
    X_norm = (X ** 2).sum(dim=1).view(-1, 1)  # (N, 1)

    for i in range(0, N, batch_size):
        j = min(i + batch_size, N)
        X_batch = X[i:j]  # (B, D)
        X_batch_norm = (X_batch ** 2).sum(dim=1).view(-1, 1)  # (B, 1)

        # Compute distance^2: ||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b
        dists = torch.clamp(X_batch_norm + X_norm.t() - 2.0 * X_batch @ X.t(), min=0.0).sqrt()
        dists, _ = torch.sort(dists, dim=1)
        # dists[: 0] is self-distance, which is 0
        if (dists[:, 1] < 1e-6).sum().item() > 0:
            mu[0, i:j] = (dists[:, 2] + epsilon) / (dists[:, 1] + epsilon)
        else:
            mu[0, i:j] = dists[:, 2] / dists[:, 1]

    mu = mu.flatten()
    mu, _ = torch.sort(mu)
    log_mu = torch.log(mu[:-1])
    F_emp = torch.arange(1, N, dtype=X.dtype, device=X.device) / N
    log_one_minus_F = - torch.log(1 - F_emp)

    return log_mu, log_one_minus_F, mu

