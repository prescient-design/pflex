"""
Maximum mean discrepancy (MMD) distance with various kernels

See:
Gretton, Arthur, et al. "A kernel two-sample test." JMLR 13.1 (2012): 723-773.

Every function here returns the square of the MMD.

"""
import torch
import torch.nn.functional as F


def analytic_rbf_mmd(hypothesis, gamma=1.0):
    """
    Closed-form expression for the distance of an RV (given by its samples)
    from the standard Gaussian

    References
    ----------
    Rustamov, Raif M. "Closed-form expressions for maximum mean discrepancy
    with applications to Wasserstein auto-encoders." Stat 10.1 (2021): e329.

    """
    dim = hypothesis.shape[-1]
    sigma2= 0.5/gamma
    term1 = (sigma2/(2.0 + sigma2))**(dim*0.5)
    term2 = -2.0 * (sigma2/(1.0 + sigma2))**(dim*0.5)
    l2_hypothesis = (hypothesis**2.0).sum(dim=-1)  # [num_samples]
    term2 *= torch.exp(-0.5*l2_hypothesis/(1.0 + sigma2)).mean()
    pairwise_dist_sq = torch.cdist(hypothesis, hypothesis, p=2.0)**2.0
    term3 = torch.exp(-gamma*pairwise_dist_sq).mean()
    return (term1 + term2 + term3).item()


def rbf_kernel(a, b, gamma=1.0):
    """
    Radial basis function kernel

    Note
    ----
    gamma is 0.5/sigma^2

    """
    return torch.exp(-gamma * torch.cdist(a, b, p=2.0)**2.0)


def get_mmd(truth, hypothesis, kernel_fn, **kernel_fn_kwargs):
    cdists = kernel_fn(truth, hypothesis, **kernel_fn_kwargs).mean()
    tdists = kernel_fn(truth, truth, **kernel_fn_kwargs).mean()
    hdists = kernel_fn(hypothesis, hypothesis, **kernel_fn_kwargs).mean()
    return (hdists + tdists - 2 * cdists).item()


def rbf_kernel_wrapped(a, b, gamma=1.0):
    """
    Radial basis function kernel, wrapped for angular distance differences

    Parameters
    ----------
    a: torch.tensor
        ~ [P, dim]
    b: torch.tensor
        ~ [Q, dim]

    Note
    ----
    gamma is 0.5/sigma^2

    """
    eps = 1.e-5
    cdist = torch.cdist(a, b, p=2.0)  # [P, Q]
    cdist = torch.clamp(torch.sin(cdist), min=-1 + eps, max=1 - eps)
    cdist = torch.asin(cdist)
    return torch.exp(-gamma * cdist**2.0)


def rbf_mmd(truth, hypothesis, gamma=1):
    """
    Get the square of the RBF-kernelized MMD distance between two sets of
    samples

    Parameters
    ----------
    truth: torch.tensor
        ~ [num_samples, dim]
    hypothesis: torch.tensor
        ~ [num_samples, dim]
    gamma: float
        Kernel parameter (0.5/sigma^2)

    Returns
    -------
    float
        MMD distance

    """
    cdists = rbf_kernel(truth, hypothesis, gamma).mean()
    tdists = rbf_kernel(truth, truth, gamma).mean()
    hdists = rbf_kernel(hypothesis, hypothesis, gamma).mean()
    return (hdists + tdists - 2 * cdists).item()


def hyperbolic_rbf_mmd(truth, hypothesis, temp=1e-2):
    truth = hyperbolic_transform(truth, temp)
    hypothesis = hyperbolic_transform(hypothesis, temp)
    return rbf_mmd(truth, hypothesis)


def hyperbolic_transform(x, c):
    # input x of shape [T, D]
    x_norm = x.norm(2, -1)[:].unsqueeze(-1)
    sqrt_c = c**0.5
    return (x / (sqrt_c * x_norm)) * torch.atanh(sqrt_c * x_norm)


def pair_mmd_cos_distance(hypothesis, truth):
    hypothesis = F.normalize(hypothesis)
    truth = F.normalize(truth)
    return pair_mmd_dot_distance(hypothesis, truth)


def pair_mmd_dot_distance(hypothesis, truth):
    hypo_k = torch.mean(torch.mm(hypothesis, hypothesis.T))
    truth_k = torch.mean(torch.mm(truth, truth.T))
    hypo_truth_k = torch.mean(torch.mm(hypothesis, truth.T))
    return (hypo_k + truth_k - 2 * hypo_truth_k).item()


def pair_mmd_l2_distance(hypothesis, truth):
    hypo_k = torch.mean(torch.exp(-torch.cdist(hypothesis, hypothesis)**2.0))
    truth_k = torch.mean(torch.exp(-torch.cdist(truth, truth)**2.0))
    hypo_truth_k = torch.mean(torch.exp(-torch.cdist(hypothesis, truth)**2.0))
    return (hypo_k + truth_k - 2 * hypo_truth_k).mean().item()


def pair_cos_distance(hypothesis, truth, temp=1):
    norm_hypo = F.normalize(hypothesis)
    norm_truth = F.normalize(truth)
    # [0, 1]: 0 most similar, 1 least similar
    dists = -torch.matmul(norm_hypo, norm_truth.T)
    dists = (dists + 1) / 2
    # softmax over truth embeddings
    weights = F.softmax(-dists / temp, dim=-1)
    # average over hypothesis sequence length
    return torch.mean(torch.sum(weights * dists, dim=-1)).item()
