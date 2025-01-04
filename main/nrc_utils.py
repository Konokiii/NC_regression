import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA


def gram_schmidt(W):
    dim = W.shape[0]

    U = torch.empty_like(W)
    U[0, :] = W[0, :] / torch.norm(W[0, :], p=2)

    for i in range(1, dim):
        j = i - 1
        ortho_vector = W[i, :]
        while j >= 0:
            proj = torch.dot(U[j, :], W[i, :]) * U[j, :]
            ortho_vector -= proj
            j -= 1
        U[i, :] = ortho_vector / torch.norm(ortho_vector, p=2)

    return U


def compute_metrics(data, device, info=None):
    result = {}
    y = data['targets']  # (B,2)
    yhat = data['predictions']  # (B,2)
    W = data['weights']  # (2,256)
    H = data['features']  # (B,256)
    H_normalized = H / (torch.norm(H, dim=1, keepdim=True) + 1e-8)
    B = H.shape[0]
    y_dim = y.shape[1]

    # result['prediction_error'] = torch.sum((y-yhat)**2).item() / B

    # R^2
    SS_res = torch.sum((y-yhat)**2).item()
    SS_tot = torch.sum((y-y.mean(dim=0))**2).item()
    result['R_sq'] = 1 - SS_res/SS_tot

    result['H_norm'] = torch.norm(H, p=2, dim=1).mean().item()
    result['W_norm'] = torch.norm(W, p=2, dim=1).mean().item()

    # WWT
    WWT = W @ W.T
    if y_dim == 2:
        result['W11'] = WWT[0, 0].item()
        result['W22'] = WWT[1, 1].item()
        result['W12'] = WWT[0, 1].item()
        result['cosSIM_W12'] = F.cosine_similarity(W[0], W[1], dim=0).item()
    elif y_dim == 3:
        result['W11'] = WWT[0, 0].item()
        result['W22'] = WWT[1, 1].item()
        result['W33'] = WWT[2, 2].item()
        result['W12'] = WWT[0, 1].item()
        result['W13'] = WWT[0, 2].item()
        result['W23'] = WWT[1, 2].item()
        result['cosSIM_W12'] = F.cosine_similarity(W[0], W[1], dim=0).item()
        result['cosSIM_W13'] = F.cosine_similarity(W[0], W[2], dim=0).item()
        result['cosSIM_W23'] = F.cosine_similarity(W[1], W[2], dim=0).item()
    result['WWT_norm'] = torch.norm(WWT).item()

    # NRC1 & explained variance ratio
    H_np = H.cpu().numpy()
    n_components = max(y_dim, 5)
    pca_for_H = PCA(n_components=n_components)
    try:
        pca_for_H.fit(H_np)
    except Exception as e:
        print('Initial PCA failed with error:', e)
        print('Try PCA with full SVD solver.')
        pca_for_H = PCA(n_components=n_components, svd_solver='full')
        pca_for_H.fit(H_np)
        # for k in range(n_components):
        #     result[f'NRC1_pca{k+1}'] = -1
        #     result[f'NRC1n_pca{k + 1}'] = -1
    finally:
        H_pca = torch.tensor(pca_for_H.components_[:n_components, :], device=device)
        H_U = gram_schmidt(H_pca)
        for k in range(n_components):
            result[f'EVR{k+1}'] = pca_for_H.explained_variance_ratio_[k]

            H_P = torch.mm(H_U[:k+1, :].T, H_U[:k+1, :])
            # H_proj_PCA = torch.mm(H, H_P)
            # result[f'NRC1_pca{k + 1}'] = torch.norm(H_proj_PCA - H).item() ** 2 / B

            H_proj_PCA_normalized = torch.mm(H_normalized, H_P)
            result[f'NRC1_pca{k + 1}'] = torch.norm(H_proj_PCA_normalized - H_normalized).item() ** 2 / B
    del H_pca, H_U, H_P, H_proj_PCA_normalized, pca_for_H, H_np

    result['NRC1'] = result[f'NRC1_pca{y_dim}']

    # NRC2 with Gram-Schmidt
    U = gram_schmidt(W)
    P_E = torch.mm(U.T, U)
    # H_proj_W = torch.mm(H, P_E)
    # result['NRC2'] = torch.norm(H - H_proj_W).item() ** 2 / B
    H_proj_W_normalized = torch.mm(H_normalized, P_E)
    result['NRC2'] = torch.norm(H_normalized - H_proj_W_normalized).item() ** 2 / B
    del H_proj_W_normalized

    if info:
        assert isinstance(info, dict), 'Extra info used to compute should be a dictionary.'
        NRC3_target = info['NRC3_target']
        NRC3n_target = info['NRC3n_target']

        # NRC3 with UFM assumption
        result['NRC3n_ufm'] = torch.norm(WWT / torch.norm(WWT) - torch.tensor(NRC3n_target, device=device)).item() ** 2
        result['NRC3_ufm'] = torch.norm(WWT - torch.tensor(NRC3_target, device=device)).item() ** 2

    # NRC2: Project H to span(w1, w2)
    # try:
    #     inverse_mat = torch.inverse(W @ W.T)
    # except Exception as e:
    #     print(e)
    #     result['NRC2_old'] = -1
    # else:
    #     H_proj_W = H @ W.T @ inverse_mat @ W
    #     result['NRC2_old'] = (torch.norm(H - H_proj_W)**2 / B).item()
    #     del H_proj_W

    return result